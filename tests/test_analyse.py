import json
from pathlib import Path

import pytest

import analyse
from extract import (
    CandidateRecord,
    ContractFactCandidate,
    EvidenceSource,
    ExtractionError,
    ExtractionRun,
    TokenUsage,
)
from ingest import PageContent
from ingest import IngestionTimeoutError
from reconcile import ReconciliationError


def _empty_run():
    return ExtractionRun(
        model="offline-test-model",
        candidate_records=(),
        evidence_assessments=(),
        usage=TokenUsage(
            input_tokens=12,
            output_tokens=8,
            total_tokens=20,
            estimated_cost_usd=0.00001,
        ),
    )


def _prepare_folder(tmp_path: Path, monkeypatch, extraction=None):
    folder = tmp_path / "contract folder with spaces"
    folder.mkdir()
    pdf = folder / "contract.pdf"
    pdf.write_bytes(b"mock PDF placeholder")
    page = PageContent(
        file=pdf.name,
        page=1,
        text="Contract terms for offline CLI test.",
        extraction_method="text",
        ocr_confidence=None,
    )
    monkeypatch.setattr(
        analyse.ExtractionConfig,
        "from_env",
        classmethod(
            lambda cls: cls(
                model="offline-test-model",
                pricing_model="offline-test-model",
                input_cost_per_million=0.15,
                output_cost_per_million=0.60,
            )
        ),
    )
    monkeypatch.setattr(analyse, "_create_client", lambda timeout_seconds: object())
    monkeypatch.setattr(
        analyse, "ingest_folder_bounded", lambda path, deadline: [page]
    )
    if extraction is None:
        extraction = lambda folder, config, client, pages: _empty_run()
    monkeypatch.setattr(analyse, "extract_folder", extraction)
    return folder


def test_success_emits_only_schema_valid_json_and_uses_pipeline_functions(
    tmp_path, monkeypatch, capsys
):
    calls = {"extract": 0, "reconcile": 0, "validate": 0}
    folder = _prepare_folder(
        tmp_path,
        monkeypatch,
        extraction=lambda folder, config, client, pages: (
            calls.__setitem__("extract", calls["extract"] + 1) or _empty_run()
        ),
    )
    real_reconcile = analyse.reconcile_run
    real_validate = analyse.validate_output

    def tracked_reconcile(run):
        calls["reconcile"] += 1
        return real_reconcile(run)

    def tracked_validate(output):
        calls["validate"] += 1
        return real_validate(output)

    monkeypatch.setattr(analyse, "reconcile_run", tracked_reconcile)
    monkeypatch.setattr(analyse, "validate_output", tracked_validate)

    assert analyse.main([str(folder)]) == 0
    captured = capsys.readouterr()

    assert json.loads(captured.out) == {
        "contract": {},
        "rates": [],
        "policies": {},
        "questions": [],
        "not_loaded": [],
    }
    assert captured.out.lstrip().startswith("{")
    assert "Discovering and ingesting" not in captured.out
    assert "Provider-reported token usage" not in captured.out
    assert "Discovering and ingesting" in captured.err
    assert "Provider-reported token usage" in captured.err
    assert "Estimated API cost" in captured.err
    assert calls == {"extract": 1, "reconcile": 1, "validate": 1}


def test_invalid_directory_fails_without_stdout(tmp_path, capsys):
    missing = tmp_path / "missing"

    assert analyse.main([str(missing)]) != 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "does not exist" in captured.err


def test_empty_pdf_folder_fails_without_stdout(tmp_path, capsys):
    folder = tmp_path / "empty"
    folder.mkdir()

    assert analyse.main([str(folder)]) != 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "No PDF documents found" in captured.err


def test_ingestion_timeout_fails_without_partial_stdout(
    tmp_path, monkeypatch, capsys
):
    folder = _prepare_folder(tmp_path, monkeypatch)
    calls = {"client": 0, "extract": 0}

    def time_out(folder, deadline):
        raise IngestionTimeoutError(
            "The runtime deadline expired while rendering scanned page 2."
        )

    def unexpected_client(timeout_seconds):
        calls["client"] += 1
        raise AssertionError("client creation must not follow ingestion timeout")

    def unexpected_extraction(*args, **kwargs):
        calls["extract"] += 1
        raise AssertionError("partial pages must not be sent for extraction")

    monkeypatch.setattr(analyse, "ingest_folder_bounded", time_out)
    monkeypatch.setattr(analyse, "_create_client", unexpected_client)
    monkeypatch.setattr(analyse, "extract_folder", unexpected_extraction)

    assert analyse.main([str(folder)]) != 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "deadline exceeded during ingestion" in captured.err.lower()
    assert "page 2" in captured.err
    assert calls == {"client": 0, "extract": 0}


def test_extraction_failure_does_not_emit_partial_json(tmp_path, monkeypatch, capsys):
    def fail_extraction(folder, config, client, pages):
        print("extractor diagnostic on stdout")
        raise ExtractionError("synthetic API failure")

    folder = _prepare_folder(tmp_path, monkeypatch, extraction=fail_extraction)

    assert analyse.main([str(folder)]) != 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "extractor diagnostic on stdout" in captured.err
    assert "Extraction failed" in captured.err


def test_missing_assessment_fails_closed_before_final_output(
    tmp_path, monkeypatch, capsys
):
    folder = _prepare_folder(tmp_path, monkeypatch)
    candidate = ContractFactCandidate(
        field="hotel_name",
        value="Example Hotel",
        source=EvidenceSource(
            file="contract.pdf",
            page=1,
            quote="Contract terms for offline CLI test.",
        ),
    )
    record = CandidateRecord(category="contract_fact", candidate=candidate)
    malformed_run = ExtractionRun.model_construct(
        model="offline-test-model",
        candidate_records=(record,),
        evidence_assessments=(),
        usage=TokenUsage(),
    )
    monkeypatch.setattr(
        analyse, "extract_folder", lambda folder, config, client, pages: malformed_run
    )

    assert analyse.main([str(folder)]) != 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "assessment integrity failure" in captured.err.lower()


def test_missing_model_pricing_fails_before_client_or_model_request(
    tmp_path, monkeypatch, capsys
):
    folder = _prepare_folder(tmp_path, monkeypatch)
    calls = {"client": 0, "extract": 0}
    monkeypatch.setattr(
        analyse.ExtractionConfig,
        "from_env",
        classmethod(lambda cls: cls(model="unpriced-model")),
    )
    monkeypatch.setattr(
        analyse,
        "_create_client",
        lambda timeout_seconds: calls.__setitem__("client", calls["client"] + 1),
    )
    monkeypatch.setattr(
        analyse,
        "extract_folder",
        lambda *args, **kwargs: calls.__setitem__("extract", calls["extract"] + 1),
    )

    assert analyse.main([str(folder)]) != 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "model pricing configuration error" in captured.err.lower()
    assert calls == {"client": 0, "extract": 0}


def test_malformed_environment_price_fails_before_client_creation(
    tmp_path, monkeypatch, capsys
):
    original_from_env = analyse.ExtractionConfig.from_env
    folder = _prepare_folder(tmp_path, monkeypatch)
    calls = {"client": 0, "extract": 0}
    monkeypatch.setenv("GEMINI_MODEL", "offline-test-model")
    monkeypatch.setenv("GEMINI_PRICING_MODEL", "offline-test-model")
    monkeypatch.setenv("GEMINI_INPUT_COST_PER_MILLION", "0.15")
    monkeypatch.setenv("GEMINI_OUTPUT_COST_PER_MILLION", "not-a-number")
    monkeypatch.setattr(
        analyse.ExtractionConfig,
        "from_env",
        classmethod(lambda cls: original_from_env()),
    )
    monkeypatch.setattr(
        analyse,
        "_create_client",
        lambda timeout_seconds: calls.__setitem__("client", calls["client"] + 1),
    )
    monkeypatch.setattr(
        analyse,
        "extract_folder",
        lambda *args, **kwargs: calls.__setitem__("extract", calls["extract"] + 1),
    )

    assert analyse.main([str(folder)]) != 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "model pricing configuration error" in captured.err.lower()
    assert "GEMINI_OUTPUT_COST_PER_MILLION" in captured.err
    assert calls == {"client": 0, "extract": 0}


def test_usage_based_cost_over_budget_is_reported_without_final_json(
    tmp_path, monkeypatch, capsys
):
    folder = _prepare_folder(tmp_path, monkeypatch)
    over_budget_run = ExtractionRun(
        model="offline-test-model",
        candidate_records=(),
        evidence_assessments=(),
        usage=TokenUsage(
            input_tokens=1_000_000,
            output_tokens=0,
            total_tokens=1_000_000,
            estimated_cost_usd=0.150000001,
            preflight_input_token_bound=100,
            preflight_estimated_cost_usd=0.01,
        ),
    )
    monkeypatch.setattr(
        analyse,
        "extract_folder",
        lambda folder, config, client, pages: over_budget_run,
    )

    assert analyse.main([str(folder)]) != 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "provider-reported token usage" in captured.err.lower()
    assert "estimated api cost limit exceeded" in captured.err.lower()


def test_schema_validation_failure_fails_without_stdout(
    tmp_path, monkeypatch, capsys
):
    folder = _prepare_folder(tmp_path, monkeypatch)

    def fail_validation(output):
        raise ReconciliationError("synthetic schema failure")

    monkeypatch.setattr(analyse, "validate_output", fail_validation)

    assert analyse.main([str(folder)]) != 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "schema validation failed" in captured.err.lower()


def test_missing_provider_usage_is_reported_as_unavailable(
    tmp_path, monkeypatch, capsys
):
    folder = _prepare_folder(tmp_path, monkeypatch)
    run_without_usage = ExtractionRun(
        model="offline-test-model",
        candidate_records=(),
        evidence_assessments=(),
        usage=TokenUsage(),
    )
    monkeypatch.setattr(
        analyse,
        "extract_folder",
        lambda folder, config, client, pages: run_without_usage,
    )

    assert analyse.main([str(folder)]) == 0
    captured = capsys.readouterr()

    assert json.loads(captured.out)["rates"] == []
    assert "input=unavailable" in captured.err
    assert "Estimated API cost unavailable" in captured.err


@pytest.mark.parametrize(
    ("status_code", "classification", "expected"),
    [
        (429, "quota_exceeded", "quota is exhausted"),
        (429, "rate_limit", "rate limit reached"),
        (401, "authentication_error", "authentication failed"),
        (400, "request_size_limit", "context/request-size limit"),
    ],
)
def test_api_quota_and_rate_limit_errors_are_distinguished(
    tmp_path, monkeypatch, capsys, status_code, classification, expected
):
    def fail_extraction(folder, config, client, pages):
        raise ExtractionError(
            "Gemini structured extraction failed",
            status_code=status_code,
            classification=classification,
        )

    folder = _prepare_folder(tmp_path, monkeypatch, extraction=fail_extraction)

    assert analyse.main([str(folder)]) != 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert expected in captured.err.lower()


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        ("quota_exceeded", "quota is exhausted"),
        ("requests_per_minute_limit", "requests-per-minute limit reached"),
        ("tokens_per_minute_limit", "tokens-per-minute limit reached"),
        ("rate_limit", "limit type could not be identified"),
    ],
)
def test_429_diagnostic_reports_classification_without_echoing_credentials(
    tmp_path, monkeypatch, capsys, classification, expected
):
    error = ExtractionError(
        "Gemini structured extraction failed: Authorization: Bearer sk-test-secret",
        status_code=429,
        classification=classification,
        provider_code="RESOURCE_EXHAUSTED",
    )

    def fail_extraction(folder, config, client, pages):
        raise error

    folder = _prepare_folder(tmp_path, monkeypatch, extraction=fail_extraction)

    assert analyse.main([str(folder)]) != 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert expected in captured.err.lower()
    assert "sk-test-secret" not in captured.err


def test_network_failure_is_reported_without_request_details(
    tmp_path, monkeypatch, capsys
):
    error = ExtractionError(
        "Gemini network request failed: Authorization: Bearer sk-network-secret",
        classification="network_error",
    )

    def fail_extraction(folder, config, client, pages):
        raise error

    folder = _prepare_folder(tmp_path, monkeypatch, extraction=fail_extraction)

    assert analyse.main([str(folder)]) != 0
    captured = capsys.readouterr()

    assert "network request failed" in captured.err.lower()
    assert "sk-network-secret" not in captured.err


def test_invalid_file_path_is_rejected(tmp_path, capsys):
    file_path = tmp_path / "not-a-folder.pdf"
    file_path.write_text("not a folder", encoding="utf-8")

    assert analyse.main([str(file_path)]) != 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "not a directory" in captured.err
