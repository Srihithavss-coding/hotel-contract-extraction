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


class FakeCompletions:
    def __init__(self, make_result):
        self.make_result = make_result
        self.requests = []

    def parse(self, model, messages, response_format, **kwargs):
        pages = json.loads(messages[1]["content"])["pages"]
        self.requests.append(
            {"model": model, "messages": messages, "pages": pages, **kwargs}
        )
        parsed = self.make_result(pages, response_format)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
        )


def _client(make_result):
    completions = FakeCompletions(make_result)
    client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    return client, completions


def _mock_openai_status_error(status, code, message, headers=None, request_headers=None):
    from httpx2 import Request, Response

    response = Response(
        status,
        headers=headers or {},
        request=Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers=request_headers or {},
        ),
    )
    body = {"error": {"code": code, "message": message}}
    if status == 429:
        from openai import RateLimitError

        return RateLimitError(message, response=response, body=body)
    from openai import APIStatusError

    return APIStatusError(message, response=response, body=body)


def _request_with_openai_error(provider_error):
    class FailingCompletions:
        def parse(self, **kwargs):
            raise provider_error

    client = SimpleNamespace(
        beta=SimpleNamespace(
            chat=SimpleNamespace(completions=FailingCompletions())
        )
    )
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
    client, completions = _client(_empty_result)
    config = _config()

    for folder in sorted(path for path in CONTRACTS.iterdir() if path.is_dir()):
        run = extract_folder(folder, config=config, client=client)
        request = completions.requests[-1]
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

    assert len(completions.requests) == 5
    assert "untrusted" in completions.requests[0]["messages"][0]["content"].lower()


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
    assert timings.openai_api_request_response_ms > 0
    assert timings.pydantic_response_parsing_ms >= 0
    assert timings.evidence_validation_ms >= 0
    assert timings.total_extraction_ms >= timings.openai_api_request_response_ms
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


def test_api_key_is_required_for_a_real_client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ExtractionError, match="OPENAI_API_KEY"):
        _create_client()


def test_openai_client_disables_sdk_retries(monkeypatch):
    import openai

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    _create_client(timeout_seconds=12)

    assert captured == {"max_retries": 0, "timeout": 12}


def test_tpm_429_preserves_status_classification_and_available_headers():
    response_headers = {
        "x-ratelimit-limit-requests": "10",
        "x-ratelimit-remaining-requests": "9",
        "x-ratelimit-reset-requests": "1s",
        "x-ratelimit-limit-tokens": "60000",
        "x-ratelimit-remaining-tokens": "0",
        "x-ratelimit-reset-tokens": "6s",
        "retry-after": "2",
        "x-request-id": "req_test_tpm",
        "x-ratelimit-limit-project-tokens": "50000",
        "x-ratelimit-api-key": "rate-limit-api-secret",
        "authorization": "Bearer sk-response-secret",
        "x-api-key": "response-api-secret",
    }
    provider_error = _mock_openai_status_error(
        429,
        "rate_limit_exceeded",
        "Rate limit reached for tokens per minute (TPM).",
        headers=response_headers,
        request_headers={"Authorization": "Bearer sk-request-secret"},
    )

    error = _request_with_openai_error(provider_error)

    assert error.status_code == 429
    assert error.classification == "tokens_per_minute_limit"
    assert error.provider_code == "rate_limit_exceeded"
    assert error.details == {
        "status_code": 429,
        "classification": "tokens_per_minute_limit",
        "provider_code": "rate_limit_exceeded",
        "rate_limit_headers": {
            "retry-after": "2",
            "x-ratelimit-limit-project-tokens": "50000",
            "x-ratelimit-limit-requests": "10",
            "x-ratelimit-limit-tokens": "60000",
            "x-ratelimit-remaining-requests": "9",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-requests": "1s",
            "x-ratelimit-reset-tokens": "6s",
            "x-request-id": "req_test_tpm",
        },
    }
    assert error.__cause__ is provider_error


def test_rpm_429_is_classified_from_provider_message_and_headers():
    provider_error = _mock_openai_status_error(
        429,
        "rate_limit_exceeded",
        "Rate limit reached for requests per minute (RPM).",
        headers={
            "x-ratelimit-limit-requests": "10",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-remaining-tokens": "50000",
        },
    )

    error = _request_with_openai_error(provider_error)

    assert error.status_code == 429
    assert error.classification == "requests_per_minute_limit"
    assert error.rate_limit_headers["x-ratelimit-remaining-requests"] == "0"


def test_429_limit_type_can_be_inferred_from_remaining_token_header():
    provider_error = _mock_openai_status_error(
        429,
        "rate_limit_exceeded",
        "Rate limit exceeded.",
        headers={
            "x-ratelimit-remaining-requests": "8",
            "x-ratelimit-remaining-tokens": "0",
        },
    )

    error = _request_with_openai_error(provider_error)

    assert error.classification == "tokens_per_minute_limit"


def test_insufficient_credit_429_is_not_classified_as_a_rate_limit():
    provider_error = _mock_openai_status_error(
        429,
        "insufficient_quota",
        "You exceeded your current quota.",
    )

    error = _request_with_openai_error(provider_error)

    assert error.status_code == 429
    assert error.classification == "insufficient_quota"
    assert error.rate_limit_headers == {}


def test_429_without_response_headers_keeps_classification_and_empty_headers():
    provider_error = _mock_openai_status_error(
        429,
        "rate_limit_exceeded",
        "Rate limit reached for tokens per minute (TPM).",
    )

    error = _request_with_openai_error(provider_error)

    assert error.status_code == 429
    assert error.classification == "tokens_per_minute_limit"
    assert error.rate_limit_headers == {}


def test_non_429_openai_errors_keep_status_and_other_api_classification():
    provider_error = _mock_openai_status_error(
        503,
        "server_error",
        "The provider is temporarily unavailable.",
    )

    error = _request_with_openai_error(provider_error)

    assert error.status_code == 503
    assert error.classification == "other_api_error"
    assert error.details["status_code"] == 503


def test_missing_structured_response_is_rejected():
    client, _ = _client(lambda pages, response_format: None)
    with pytest.raises(ExtractionError, match="structured candidate data"):
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
    client, completions = _client(_empty_result)

    with pytest.raises(ModelPricingConfigurationError):
        extract_folder(
            CONTRACTS / "c01",
            config=config,
            client=client,
            pages=[_single_page()],
        )

    assert completions.requests == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENAI_INPUT_COST_PER_MILLION", "not-a-number"),
        ("OPENAI_OUTPUT_COST_PER_MILLION", "not-a-number"),
        ("OPENAI_INPUT_COST_PER_MILLION", "NaN"),
        ("OPENAI_OUTPUT_COST_PER_MILLION", "Infinity"),
        ("OPENAI_INPUT_COST_PER_MILLION", "-Infinity"),
        ("OPENAI_INPUT_COST_PER_MILLION", "-0.01"),
        ("OPENAI_OUTPUT_COST_PER_MILLION", "-0.01"),
    ],
)
def test_invalid_environment_prices_fail_before_client_creation(
    monkeypatch, name, value
):
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_PRICING_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_INPUT_COST_PER_MILLION", "0.15")
    monkeypatch.setenv("OPENAI_OUTPUT_COST_PER_MILLION", "0.60")
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
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_PRICING_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_INPUT_COST_PER_MILLION", "0.15")
    monkeypatch.setenv("OPENAI_OUTPUT_COST_PER_MILLION", "0.60")
    client, completions = _client(_empty_result)

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
    assert len(completions.requests) == 1
    assert completions.requests[0]["model"] == "test-model"


def test_high_preflight_estimate_is_rejected_before_request():
    client, completions = _client(_empty_result)
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

    assert completions.requests == []
