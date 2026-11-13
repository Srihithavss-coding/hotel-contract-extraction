import json
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import extract
import pymupdf
import pytest

from extract import (
    AdvancePaymentCandidate,
    CancellationCandidate,
    CandidateRecord,
    EstimatedCostLimitError,
    EvidenceSource,
    ExtractionConfig,
    ExtractionError,
    MAX_COMPLETION_TOKENS,
    MAX_ESTIMATED_API_COST_USD,
    ModelPricingConfigurationError,
    RoomRateCandidate,
    _assign_payment_schedule_ids,
    _quote_contains_price,
    _create_client,
    extract_folder,
)
from ingest import PageContent, PageLine

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"


class FakeGeminiModels:
    def __init__(self, make_result):
        self.make_result = make_result
        self.requests = []

    def generate_content(self, model, contents, config):
        pages = json.loads(contents)["pages"]
        self.requests.append(
            {"model": model, "contents": contents, "config": config, "pages": pages}
        )
        parsed = self.make_result(pages, extract.ExtractionResult)
        return SimpleNamespace(
            text=parsed.model_dump_json() if parsed is not None else None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=1000,
                candidates_token_count=200,
                total_token_count=1200,
            ),
        )


def _client(make_result):
    models = FakeGeminiModels(make_result)
    return SimpleNamespace(models=models), models


def _mock_gemini_api_error(status, code, message):
    from google.genai.errors import ClientError

    return ClientError(
        status,
        {"error": {"status": code, "message": message}},
    )


def _request_with_gemini_error(provider_error):
    class FailingModels:
        def generate_content(self, **kwargs):
            raise provider_error

    client = SimpleNamespace(models=FailingModels())
    with pytest.raises(ExtractionError) as raised:
        extract._request_extraction(client, "test-model", "prompt", "page message")
    return raised.value


def _empty_result(pages, response_format):
    return response_format()


def _config(**overrides):
    values = {
        "model": "test-model",
        "pricing_model": "test-model",
        "input_cost_per_million": 0.15,
        "output_cost_per_million": 0.60,
    }
    values.update(overrides)
    return ExtractionConfig(**values)


def _normalise(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def test_all_sample_contract_folders_are_sent_as_structured_pages():
    client, models = _client(_empty_result)
    config = _config()

    for folder in sorted(path for path in CONTRACTS.iterdir() if path.is_dir()):
        run = extract_folder(folder, config=config, client=client)
        request = models.requests[-1]
        expected_sources = set()
        for pdf_path in sorted(folder.glob("*.pdf")):
            with pymupdf.open(pdf_path) as document:
                expected_sources.update(
                    (pdf_path.name, page_number)
                    for page_number in range(1, document.page_count + 1)
                )
        received_sources = {(page["file"], page["page"]) for page in request["pages"]}

        assert received_sources == expected_sources
        assert all(page["text"] for page in request["pages"])
        assert run.candidate_records == ()
        assert run.model == "test-model"

    assert len(models.requests) == 5
    config = models.requests[0]["config"]
    assert "untrusted" in config.system_instruction.lower()
    assert config.response_mime_type == "application/json"
    assert "exclusiveMinimum" not in config.response_schema["$defs"][
        "RoomRateCandidate"
    ]["properties"]["price"]
    amount_schema = config.response_schema["$defs"]["CommercialTermCandidate"][
        "properties"
    ]["amount"]
    assert all(
        "exclusiveMinimum" not in variant
        for variant in amount_schema["anyOf"]
    )
    original_schema = extract.ExtractionResult.model_json_schema()
    assert (
        original_schema["$defs"]["RoomRateCandidate"]["properties"]["price"][
            "exclusiveMinimum"
        ]
        == 0
    )
    assert (
        original_schema["$defs"]["CommercialTermCandidate"]["properties"]["amount"][
            "anyOf"
        ][0]["exclusiveMinimum"]
        == 0
    )
    assert config.max_output_tokens == MAX_COMPLETION_TOKENS


@pytest.mark.parametrize(
    ("category", "candidate"),
    [
        (
            "room_rates",
            {
                "occupancy": "single",
                "price": 0,
                "rate_type": "individual",
                "source": {"file": "contract.pdf", "page": 1, "quote": "zero"},
            },
        ),
        (
            "room_rates",
            {
                "occupancy": "single",
                "price": -1,
                "rate_type": "individual",
                "source": {"file": "contract.pdf", "page": 1, "quote": "negative"},
            },
        ),
        (
            "commercial_terms",
            {
                "kind": "other",
                "description": "zero amount",
                "amount": 0,
                "source": {"file": "contract.pdf", "page": 1, "quote": "zero"},
            },
        ),
        (
            "commercial_terms",
            {
                "kind": "other",
                "description": "negative amount",
                "amount": -1,
                "source": {"file": "contract.pdf", "page": 1, "quote": "negative"},
            },
        ),
    ],
)
def test_gemini_response_keeps_strict_positive_price_validation(category, candidate):
    payload = {field: [] for field in extract.ExtractionResult.model_fields}
    payload[category] = [candidate]

    with pytest.raises(ExtractionError, match="did not match the evidence schema"):
        extract._read_response(SimpleNamespace(text=json.dumps(payload)))


def test_c01_hotel_fact_returns_as_a_source_backed_candidate():
    def make_hotel(pages, response_format):
        page = next(page for page in pages if page["file"] == "rate-letter.pdf")
        quote = next(line["text"] for line in page["lines"] if line["text"].strip())
        return response_format(
            contract_facts=[
                {
                    "field": "hotel_name",
                    "value": quote,
                    "source": {
                        "file": page["file"],
                        "page": page["page"],
                        "quote": quote,
                    },
                }
            ]
        )

    client, _ = _client(make_hotel)
    run = extract_folder(
        CONTRACTS / "c01",
        config=_config(),
        client=client,
    )

    candidate = next(
        record.candidate
        for record in run.candidate_records
        if record.category == "contract_fact"
    )
    assert candidate.field == "hotel_name"
    assert candidate.value == candidate.source.quote
    assert candidate.source.file == "rate-letter.pdf"
    assert candidate.source.page == 1


def test_c02_rate_candidate_keeps_a_source_backed_scanned_table_row():
    def make_rates(pages, response_format):
        scanned_page = next(page for page in pages if page["page"] == 2)
        room, meal_plan = "Garden Villa", "CP"
        line = next(
            line["text"]
            for line in scanned_page["lines"]
            if _normalise(room) in _normalise(line["text"])
            and _normalise(meal_plan) in _normalise(line["text"])
        )
        prices = re.findall(r"(?<!\d)\d[\d,]*", line)
        return response_format(
            room_rates=[
                RoomRateCandidate(
                    room=room,
                    meal_plan=meal_plan,
                    occupancy="single",
                    price=float(prices[0].replace(",", "")),
                    rate_type="individual",
                    source=EvidenceSource(
                        file=scanned_page["file"],
                        page=scanned_page["page"],
                        quote=line,
                    ),
                )
            ]
        )

    client, _ = _client(make_rates)
    run = extract_folder(
        CONTRACTS / "c02",
        config=_config(),
        client=client,
    )

    rate_candidates = [
        record.candidate
        for record in run.candidate_records
        if record.category == "room_rate"
    ]
    assert len(rate_candidates) == 1
    candidate = rate_candidates[0]
    assert candidate.room == "Garden Villa"
    assert candidate.meal_plan == "CP"
    assert f"{candidate.price:,.0f}" in candidate.source.quote
    assert candidate.source.page == 2
    assert _normalise("Garden Villa") in _normalise(candidate.source.quote)
    assert all(
        f"{candidate.price:,.0f}" in candidate.source.quote
        for candidate in rate_candidates
    )
    assert run.usage.input_tokens == 1000
    assert run.usage.output_tokens == 200
    assert run.usage.total_tokens == 1200
    assert run.usage.estimated_cost_usd == pytest.approx(0.00027)
    assert run.usage.preflight_input_token_bound > 0
    expected_preflight = (
        run.usage.preflight_input_token_bound * 0.15
        + MAX_COMPLETION_TOKENS * 0.60
    ) / 1_000_000
    assert run.usage.preflight_estimated_cost_usd == pytest.approx(
        expected_preflight
    )
    assert run.usage.preflight_estimated_cost_usd <= MAX_ESTIMATED_API_COST_USD


def test_extraction_timing_aggregates_ingestion_stages_and_total(monkeypatch):
    page = PageContent(
        file="timed.pdf",
        page=1,
        text="Hotel details",
        extraction_method="text",
        ocr_confidence=None,
        lines=[PageLine(text="Hotel details")],
    )

    def timed_ingest(folder, timings=None):
        timings.pdf_discovery_seconds = 0.004
        timings.pdf_ingestion_seconds = 0.012
        timings.rendering_seconds = 0.003
        timings.ocr_seconds = 0.006
        return [page]

    monkeypatch.setattr(extract, "ingest_folder", timed_ingest)
    client, _ = _client(_empty_result)
    run = extract_folder("unused", config=_config(), client=client)

    timings = run.timings
    assert timings.pdf_discovery_ms == pytest.approx(4)
    assert timings.pdf_ingestion_ms == pytest.approx(12)
    assert timings.rendering_ms == pytest.approx(3)
    assert timings.ocr_ms == pytest.approx(6)
    assert timings.prompt_construction_preflight_ms > 0
    assert timings.model_api_request_response_ms > 0
    assert timings.pydantic_response_parsing_ms >= 0
    assert timings.evidence_validation_ms >= 0
    assert timings.total_extraction_ms >= timings.model_api_request_response_ms
    assert timings.total_extraction_ms > 0
    assert timings.rendering_ms <= timings.pdf_ingestion_ms
    assert timings.ocr_ms <= timings.pdf_ingestion_ms


def test_payment_schedule_identity_links_adjacent_sentences_but_separates_labels():
    advance_quote = "50% of the booking value is payable at the time of confirmation."
    balance_quote = "The balance 50% is payable 21 days before arrival."
    page_text = "\n".join(
        [
            "Payment",
            advance_quote,
            balance_quote,
            "Schedule A",
            "25% is payable at booking.",
            "Schedule B",
            "100% is payable 30 days before arrival.",
        ]
    )
    page = PageContent(
        file="rate-letter.pdf",
        page=1,
        text=page_text,
        extraction_method="text",
        ocr_confidence=None,
        lines=[PageLine(text=line) for line in page_text.splitlines()],
        document_id="contract/rate-letter.pdf",
    )

    def record(candidate_id, quote, percentage, timing, balance_wording=None):
        return CandidateRecord(
            candidate_id=candidate_id,
            category="advance_payment",
            candidate=AdvancePaymentCandidate(
                percentage=percentage,
                timing=timing,
                balance_wording=balance_wording,
                source=EvidenceSource(file=page.file, page=page.page, quote=quote),
            ),
        )

    records = _assign_payment_schedule_ids(
        (
            record("advance", advance_quote, 50, "at confirmation"),
            record(
                "balance",
                balance_quote,
                None,
                None,
                balance_wording=balance_quote,
            ),
            record("schedule-a", "25% is payable at booking.", 25, "at booking"),
            record(
                "schedule-b",
                "100% is payable 30 days before arrival.",
                100,
                "30 days before arrival",
            ),
        ),
        [page],
    )
    schedule_ids = {item.candidate_id: item.schedule_id for item in records}

    assert schedule_ids["advance"] == schedule_ids["balance"]
    assert schedule_ids["schedule-a"] != schedule_ids["schedule-b"]


def test_c03_conflicting_cancellation_candidates_remain_separate():
    def make_cancellations(pages, response_format):
        candidates = []
        for page in pages:
            for line in page["text"].splitlines():
                match = re.search(
                    r"Cancellation:.*?within (\d+) days of arrival\.",
                    line,
                    flags=re.IGNORECASE,
                )
                if match:
                    candidates.append(
                        CancellationCandidate(
                            charge_percent=100,
                            days_max=int(match.group(1)),
                            source=EvidenceSource(
                                file=page["file"],
                                page=page["page"],
                                quote=line,
                            ),
                        )
                    )
        return response_format(cancellations=candidates)

    client, _ = _client(make_cancellations)
    run = extract_folder(
        CONTRACTS / "c03",
        config=_config(),
        client=client,
    )

    assert {
        (candidate.charge_percent, candidate.days_max, candidate.source.file)
        for record in run.candidate_records
        if record.category == "cancellation"
        for candidate in [record.candidate]
    } == {
        (100, 21, "rate-letter.pdf"),
        (100, 30, "email.pdf"),
    }


def test_rate_quote_validator_recognises_localised_price_grouping():
    assert _quote_contains_price("Garden Villa CP 7,000", 7000)
    assert _quote_contains_price("Banyan Suite 1,00,000", 100000)
    assert not _quote_contains_price("Garden Villa CP 7,800", 7000)


def test_non_verbatim_source_quote_is_preserved_as_invalid_evidence():
    def invented_quote(pages, response_format):
        return response_format(
            contract_facts=[
                {
                    "field": "hotel_name",
                    "value": "Made Up Hotel",
                    "source": {
                        "file": pages[0]["file"],
                        "page": pages[0]["page"],
                        "quote": "This quote is not in the document",
                    },
                }
            ]
        )

    client, _ = _client(invented_quote)
    run = extract_folder(
        CONTRACTS / "c01",
        config=_config(),
        client=client,
    )

    assert len(run.candidate_records) == 1
    assert run.evidence_assessments[0].status == "invalid"
    assert "exact substring" in run.evidence_assessments[0].issues[0]


def test_gemini_api_key_is_required_for_a_real_client(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ExtractionError, match="GEMINI_API_KEY"):
        _create_client()


def test_gemini_client_disables_sdk_retries(monkeypatch):
    from google import genai

    captured = {}

    class FakeGeminiClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    monkeypatch.setattr(genai, "Client", FakeGeminiClient)

    client = _create_client(timeout_seconds=12)

    assert isinstance(client, FakeGeminiClient)
    assert captured["api_key"] == "offline-test-key"
    assert captured["http_options"].timeout == 12_000
    assert captured["http_options"].retry_options.attempts == 1


def test_gemini_tpm_429_preserves_status_and_classification():
    provider_error = _mock_gemini_api_error(
        429,
        "RESOURCE_EXHAUSTED",
        "Rate limit reached for tokens per minute (TPM).",
    )

    error = _request_with_gemini_error(provider_error)

    assert error.status_code == 429
    assert error.classification == "tokens_per_minute_limit"
    assert error.provider_code == "RESOURCE_EXHAUSTED"
    assert error.details == {
        "status_code": 429,
        "classification": "tokens_per_minute_limit",
        "provider_code": "RESOURCE_EXHAUSTED",
    }
    assert error.__cause__ is provider_error


def test_gemini_rpm_429_is_classified_from_provider_message():
    provider_error = _mock_gemini_api_error(
        429,
        "RESOURCE_EXHAUSTED",
        "Rate limit reached for requests per minute (RPM).",
    )

    error = _request_with_gemini_error(provider_error)

    assert error.status_code == 429
    assert error.classification == "requests_per_minute_limit"

def test_gemini_quota_429_is_not_reported_as_a_minute_rate_limit():
    provider_error = _mock_gemini_api_error(
        429,
        "RESOURCE_EXHAUSTED",
        "Daily quota exceeded for the configured model.",
    )

    error = _request_with_gemini_error(provider_error)

    assert error.status_code == 429
    assert error.classification == "quota_exceeded"


def test_gemini_authentication_error_is_reported_without_echoing_key():
    provider_error = _mock_gemini_api_error(
        400,
        "API_KEY_INVALID",
        "API key not valid. api-key=not-a-real-secret",
    )

    error = _request_with_gemini_error(provider_error)

    assert error.status_code == 400
    assert error.classification == "authentication_error"
    assert "not-a-real-secret" not in str(error)


def test_gemini_network_failure_is_classified_without_retry():
    from httpx2 import ConnectError, Request

    provider_error = ConnectError(
        "connection failed",
        request=Request("POST", "https://generativelanguage.googleapis.com"),
    )
    error = _request_with_gemini_error(provider_error)

    assert error.status_code is None
    assert error.classification == "network_error"
    assert error.__cause__ is provider_error


def test_other_gemini_api_error_keeps_status_and_classification():
    provider_error = _mock_gemini_api_error(
        503,
        "UNAVAILABLE",
        "The provider is temporarily unavailable.",
    )

    error = _request_with_gemini_error(provider_error)

    assert error.status_code == 503
    assert error.classification == "api_error"
    assert error.details["status_code"] == 503


def test_malformed_gemini_json_is_rejected_without_fabricating_candidates():
    with pytest.raises(ExtractionError, match="malformed JSON") as raised:
        extract._read_response(SimpleNamespace(text="{broken", usage_metadata=None))

    assert raised.value.classification == "invalid_response"


def test_gemini_json_with_missing_candidate_categories_is_rejected():
    with pytest.raises(ExtractionError, match="omitted required candidate categories") as raised:
        extract._read_response(
            SimpleNamespace(text='{"contract_facts": []}', usage_metadata=None)
        )

    assert raised.value.classification == "invalid_response"


def test_gemini_json_with_unexpected_properties_is_rejected():
    payload = {name: [] for name in extract.ExtractionResult.model_fields}
    payload["unexpected"] = True

    with pytest.raises(ExtractionError, match="did not match the evidence schema") as raised:
        extract._read_response(
            SimpleNamespace(text=json.dumps(payload), usage_metadata=None)
        )

    assert raised.value.classification == "invalid_response"


def test_missing_structured_response_is_rejected():
    client, _ = _client(lambda pages, response_format: None)
    with pytest.raises(ExtractionError, match="structured candidate JSON"):
        extract_folder(
            CONTRACTS / "c01",
            config=_config(),
            client=client,
        )


def _single_page():
    return PageContent(
        file="contract.pdf",
        page=1,
        text="Hotel rates apply throughout the season.",
        extraction_method="text",
        ocr_confidence=None,
        lines=[PageLine(text="Hotel rates apply throughout the season.")],
    )


@pytest.mark.parametrize(
    "config",
    [
        ExtractionConfig(model="unknown-model"),
        ExtractionConfig(
            model="test-model",
            pricing_model="another-model",
            input_cost_per_million=0.1,
            output_cost_per_million=0.2,
        ),
    ],
)
def test_missing_or_mismatched_model_prices_fail_before_request(config):
    client, models = _client(_empty_result)

    with pytest.raises(ModelPricingConfigurationError):
        extract_folder(
            CONTRACTS / "c01",
            config=config,
            client=client,
            pages=[_single_page()],
        )

    assert models.requests == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GEMINI_INPUT_COST_PER_MILLION", "not-a-number"),
        ("GEMINI_OUTPUT_COST_PER_MILLION", "not-a-number"),
        ("GEMINI_INPUT_COST_PER_MILLION", "NaN"),
        ("GEMINI_OUTPUT_COST_PER_MILLION", "Infinity"),
        ("GEMINI_INPUT_COST_PER_MILLION", "-Infinity"),
        ("GEMINI_INPUT_COST_PER_MILLION", "-0.01"),
        ("GEMINI_OUTPUT_COST_PER_MILLION", "-0.01"),
    ],
)
def test_invalid_environment_prices_fail_before_client_creation(
    monkeypatch, name, value
):
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_PRICING_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_INPUT_COST_PER_MILLION", "0.15")
    monkeypatch.setenv("GEMINI_OUTPUT_COST_PER_MILLION", "0.60")
    monkeypatch.setenv(name, value)
    client_creations = []
    monkeypatch.setattr(
        "extract._create_client",
        lambda: client_creations.append("created"),
    )

    with pytest.raises(ModelPricingConfigurationError):
        config = ExtractionConfig.from_env()
        extract_folder(
            CONTRACTS / "c01",
            config=config,
            pages=[_single_page()],
        )

    assert client_creations == []


def test_matching_model_pricing_environment_allows_extraction(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_PRICING_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_INPUT_COST_PER_MILLION", "0.15")
    monkeypatch.setenv("GEMINI_OUTPUT_COST_PER_MILLION", "0.60")
    client, models = _client(_empty_result)

    config = ExtractionConfig.from_env()
    run = extract_folder(
        CONTRACTS / "c01",
        config=config,
        client=client,
        pages=[_single_page()],
    )

    assert config.require_model_prices() == (0.15, 0.60)
    assert config.model == config.pricing_model == "test-model"
    assert run.model == "test-model"
    assert len(models.requests) == 1
    assert models.requests[0]["model"] == "test-model"


def test_high_preflight_estimate_is_rejected_before_request():
    client, models = _client(_empty_result)
    config = _config(
        input_cost_per_million=1000,
        output_cost_per_million=1000,
    )

    with pytest.raises(EstimatedCostLimitError, match="no request was sent"):
        extract_folder(
            CONTRACTS / "c01",
            config=config,
            client=client,
            pages=[_single_page()],
        )

    assert models.requests == []
