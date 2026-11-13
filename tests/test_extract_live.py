import json
import os
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path
from time import perf_counter

import pytest

from extract import (
    CandidateEvidenceAssessment,
    CandidateRecord,
    EvidenceSource,
    ExtractionError,
    ExtractionTimings,
    RoomRateCandidate,
    _create_client,
    extract_folder,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
EXPECTED_SEASONS = {
    ("2026-10-01", "2027-01-15"): "Peak season",
    ("2027-01-16", "2027-09-30"): "Lean season",
}


def _normalise_room_name(room):
    if room is None:
        return None
    normalized = " ".join(room.casefold().split())
    start = 0
    end = len(normalized)
    while start < end and unicodedata.category(normalized[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(normalized[end - 1]).startswith("P"):
        end -= 1
    return normalized[start:end].strip()


def _expected_rate_key(rate):
    season = EXPECTED_SEASONS[(rate["start"], rate["end"])]
    return (
        season,
        rate["start"],
        rate["end"],
        _normalise_room_name(rate["room"]),
        rate["meal_plan"],
        rate["occupancy"],
        rate["price"],
    )


def _actual_rate_key(rate):
    start = rate.start_date.isoformat() if rate.start_date else None
    end = rate.end_date.isoformat() if rate.end_date else None
    return (
        rate.season,
        start,
        end,
        _normalise_room_name(rate.room),
        rate.meal_plan,
        rate.occupancy,
        rate.price,
    )


def _rate_report_record(rate):
    key = _actual_rate_key(rate)
    return {
        "season": key[0],
        "start_date": key[1],
        "end_date": key[2],
        "room": key[3],
        "meal_plan": key[4],
        "occupancy": key[5],
        "price": key[6],
        "rate_type": rate.rate_type,
        "source": rate.source.model_dump(mode="json"),
    }


def _candidate_diagnostic_record(record, assessment, expected_keys):
    rate = record.candidate
    rate_key = _actual_rate_key(rate)
    return {
        "candidate_id": record.candidate_id,
        "matches_expected_fixture_values": rate_key in expected_keys,
        "room": rate.room,
        "season": rate.season,
        "start_date": rate.start_date.isoformat() if rate.start_date else None,
        "end_date": rate.end_date.isoformat() if rate.end_date else None,
        "meal_plan": rate.meal_plan,
        "occupancy": rate.occupancy,
        "price": rate.price,
        "rate_type": rate.rate_type,
        "evidence_assessment_status": assessment.status,
        "validation_issues": list(assessment.issues),
        "source_file": rate.source.file,
        "source_page": rate.source.page,
    }


def _rate_diagnostics(rate_records, assessments, expected_rates, expected_sources):
    expected_keys = {_expected_rate_key(rate) for rate in expected_rates}
    expected_table_candidates = []
    other_candidates = []
    for record in rate_records:
        rate = record.candidate
        is_expected_table_source = (
            (rate.source.file, rate.source.page) in expected_sources
            and rate.rate_type not in {"group", "rack"}
        )
        report_record = _candidate_diagnostic_record(
            record,
            assessments[record.candidate_id],
            expected_keys,
        )
        if is_expected_table_source:
            expected_table_candidates.append(report_record)
        else:
            other_candidates.append(report_record)

    return {
        "expected_table_rate_candidates": expected_table_candidates,
        "other_commercial_rate_candidates": other_candidates,
        "expected_table_candidates_not_verified": [
            record
            for record in expected_table_candidates
            if record["evidence_assessment_status"] != "verified"
        ],
    }


def _timing_report(timings):
    return {
        "unit": "milliseconds",
        **timings.model_dump(mode="json"),
    }


def _compare_rates(expected_rates, actual_rates):
    expected_keys = [_expected_rate_key(rate) for rate in expected_rates]
    actual_keys = [_actual_rate_key(rate) for rate in actual_rates]
    expected_counts = Counter(expected_keys)
    remaining_expected = expected_counts.copy()
    seen_actual = Counter()
    exact_matches = []
    incorrect_candidates = []
    duplicate_candidates = []

    for candidate, key in zip(actual_rates, actual_keys):
        if remaining_expected[key] > 0:
            exact_matches.append(_rate_report_record(candidate))
            remaining_expected[key] -= 1
        elif seen_actual[key] > 0:
            duplicate_candidates.append(_rate_report_record(candidate))
        else:
            incorrect_candidates.append(_rate_report_record(candidate))
        seen_actual[key] += 1

    missing_candidates = []
    for expected_rate, key in zip(expected_rates, expected_keys):
        if remaining_expected[key] > 0:
            missing_candidates.append(expected_rate)
            remaining_expected[key] -= 1

    return {
        "exact_matches": exact_matches,
        "missing_candidates": missing_candidates,
        "incorrect_candidates": incorrect_candidates,
        "duplicate_candidates": duplicate_candidates,
    }


def test_rate_comparison_catches_misassociations_and_duplicates():
    source = {
        "file": "synthetic-rate-sheet.pdf",
        "page": 2,
        "quote": "Garden Villa CP 7,000 single 7,800 double",
    }
    expected_rates = [
        {
            "room": "Garden Villa",
            "meal_plan": "CP",
            "occupancy": "single",
            "price": 7000,
            "start": "2026-10-01",
            "end": "2027-01-15",
            "source": source,
        },
        {
            "room": "Garden Villa",
            "meal_plan": "CP",
            "occupancy": "double",
            "price": 7800,
            "start": "2026-10-01",
            "end": "2027-01-15",
            "source": source,
        },
    ]

    def candidate_from_expected(rate, **changes):
        values = {
            "room": rate["room"],
            "meal_plan": rate["meal_plan"],
            "occupancy": rate["occupancy"],
            "price": rate["price"],
            "season": "Peak season",
            "start_date": rate["start"],
            "end_date": rate["end"],
            "rate_type": "individual",
            "source": rate["source"],
        }
        values.update(changes)
        values["start_date"] = date.fromisoformat(values["start_date"])
        values["end_date"] = date.fromisoformat(values["end_date"])
        values["source"] = EvidenceSource(**values["source"])
        return RoomRateCandidate(**values)

    wrong_occupancy = candidate_from_expected(expected_rates[1], occupancy="single")
    actual_rates = [
        candidate_from_expected(expected_rates[0]),
        candidate_from_expected(expected_rates[0]),
        wrong_occupancy,
    ]

    comparison = _compare_rates(expected_rates, actual_rates)

    assert len(comparison["exact_matches"]) == 1
    assert len(comparison["missing_candidates"]) == 1
    assert len(comparison["incorrect_candidates"]) == 1
    assert len(comparison["duplicate_candidates"]) == 1


@pytest.mark.parametrize(
    "actual_room",
    ["garden villa", "  Garden   Villa  ", "(Garden Villa),"],
)
def test_rate_comparison_normalizes_room_case_spacing_and_surrounding_punctuation(
    actual_room,
):
    source = {
        "file": "rate-letter.pdf",
        "page": 2,
        "quote": f"{actual_room} CP 7,000 single",
    }
    expected_rates = [
        {
            "room": "Garden Villa",
            "meal_plan": "CP",
            "occupancy": "single",
            "price": 7000,
            "start": "2026-10-01",
            "end": "2027-01-15",
            "source": source,
        }
    ]
    actual_rates = [
        RoomRateCandidate(
            room=actual_room,
            meal_plan="CP",
            occupancy="single",
            price=7000,
            season="Peak season",
            start_date=date(2026, 10, 1),
            end_date=date(2027, 1, 15),
            rate_type="individual",
            source=EvidenceSource(**source),
        )
    ]

    comparison = _compare_rates(expected_rates, actual_rates)

    assert len(comparison["exact_matches"]) == 1
    assert comparison["missing_candidates"] == []
    assert comparison["incorrect_candidates"] == []
    assert actual_rates[0].room == actual_room
    assert actual_rates[0].source.quote == source["quote"]


def test_rate_comparison_does_not_merge_distinct_room_types():
    source = {
        "file": "rate-letter.pdf",
        "page": 2,
        "quote": "Garden Villa Deluxe CP 7,000 single",
    }
    expected_rates = [
        {
            "room": "Garden Villa",
            "meal_plan": "CP",
            "occupancy": "single",
            "price": 7000,
            "start": "2026-10-01",
            "end": "2027-01-15",
            "source": source,
        }
    ]
    actual_rates = [
        RoomRateCandidate(
            room="Garden Villa Deluxe",
            meal_plan="CP",
            occupancy="single",
            price=7000,
            season="Peak season",
            start_date=date(2026, 10, 1),
            end_date=date(2027, 1, 15),
            rate_type="individual",
            source=EvidenceSource(**source),
        )
    ]

    comparison = _compare_rates(expected_rates, actual_rates)

    assert comparison["exact_matches"] == []
    assert len(comparison["missing_candidates"]) == 1
    assert len(comparison["incorrect_candidates"]) == 1


def _insufficient_quota_reason(error):
    if error.classification == "quota_exceeded":
        return "Gemini reports exhausted quota."
    return None


def _report_extraction_error(error, elapsed):
    quota_reason = _insufficient_quota_reason(error)
    if quota_reason:
        message = (
            "Live c02 evaluation blocked before extraction completed. "
            f"{quota_reason} No rate candidates were compared; this is not an "
            "extraction-accuracy result. Add credits or resolve billing, then rerun."
        )
        print(message)
        pytest.skip(message)

    pytest.fail(
        "Live c02 extraction did not complete; no fixture comparison was made. "
        f"Elapsed: {elapsed}s. Error: {error}",
        pytrace=False,
    )


def test_insufficient_credit_429_is_reported_as_blocked_not_inaccurate(capsys):
    error = ExtractionError(
        "Gemini structured extraction failed",
        status_code=429,
        classification="quota_exceeded",
    )

    with pytest.raises(pytest.skip.Exception, match="blocked before extraction"):
        _report_extraction_error(error, elapsed=1.25)

    output = capsys.readouterr().out.lower()
    assert "no rate candidates were compared" in output
    assert "not an extraction-accuracy result" in output


def test_ordinary_rate_limit_429_is_not_mislabeled_as_insufficient_credits():
    error = ExtractionError(
        "Gemini structured extraction failed",
        status_code=429,
        classification="rate_limit",
    )

    assert _insufficient_quota_reason(error) is None


def test_rate_diagnostics_report_uncertain_table_reasons_and_group_rates():
    source = EvidenceSource(
        file="rate-letter.pdf",
        page=2,
        quote="Garden Villa CP 7,000 single 7,800 double",
    )
    table_rate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="single",
        price=7000,
        season="Peak season",
        start_date=date(2026, 10, 1),
        end_date=date(2027, 1, 15),
        rate_type="individual",
        source=source,
    )
    group_rate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="double",
        price=6500,
        season=None,
        rate_type="group",
        source=EvidenceSource(
            file="group-rates.pdf",
            page=1,
            quote="Garden Villa group rate CP 6,500",
        ),
    )
    table_record = CandidateRecord(
        candidate_id="table-uncertain",
        category="room_rate",
        candidate=table_rate,
    )
    group_record = CandidateRecord(
        candidate_id="group-verified",
        category="room_rate",
        candidate=group_rate,
    )
    assessments = {
        "table-uncertain": CandidateEvidenceAssessment(
            candidate_id="table-uncertain",
            category="room_rate",
            status="uncertain",
            issues=("Rate type is not explicit in the cited context.",),
        ),
        "group-verified": CandidateEvidenceAssessment(
            candidate_id="group-verified",
            category="room_rate",
            status="verified",
        ),
    }
    expected_rates = [
        {
            "room": "Garden Villa",
            "meal_plan": "CP",
            "occupancy": "single",
            "price": 7000,
            "start": "2026-10-01",
            "end": "2027-01-15",
            "source": source.model_dump(mode="json"),
        }
    ]

    report = _rate_diagnostics(
        [table_record, group_record],
        assessments,
        expected_rates,
        {("rate-letter.pdf", 2)},
    )

    assert [item["candidate_id"] for item in report["expected_table_rate_candidates"]] == [
        "table-uncertain"
    ]
    assert [item["candidate_id"] for item in report["other_commercial_rate_candidates"]] == [
        "group-verified"
    ]
    uncertain = report["expected_table_candidates_not_verified"][0]
    assert uncertain["matches_expected_fixture_values"] is True
    assert uncertain["candidate_id"] == "table-uncertain"
    assert uncertain["room"] == "Garden Villa"
    assert uncertain["season"] == "Peak season"
    assert (uncertain["start_date"], uncertain["end_date"]) == (
        "2026-10-01",
        "2027-01-15",
    )
    assert uncertain["meal_plan"] == "CP"
    assert uncertain["occupancy"] == "single"
    assert uncertain["price"] == 7000
    assert uncertain["rate_type"] == "individual"
    assert uncertain["evidence_assessment_status"] == "uncertain"
    assert uncertain["validation_issues"] == [
        "Rate type is not explicit in the cited context."
    ]
    assert (uncertain["source_file"], uncertain["source_page"]) == (
        "rate-letter.pdf",
        2,
    )


def test_timing_report_uses_milliseconds_and_preserves_total():
    timings = ExtractionTimings(
        pdf_discovery_ms=4.5,
        pdf_ingestion_ms=120,
        rendering_ms=25,
        ocr_ms=80,
        prompt_construction_preflight_ms=12,
        model_api_request_response_ms=8000,
        pydantic_response_parsing_ms=3,
        evidence_validation_ms=5,
        total_extraction_ms=8145,
    )

    report = _timing_report(timings)

    assert report["unit"] == "milliseconds"
    assert report["pdf_discovery_ms"] == 4.5
    assert report["ocr_ms"] == 80
    assert report["total_extraction_ms"] == 8145
    assert report["total_extraction_ms"] >= sum(
        report[key]
        for key in (
            "pdf_discovery_ms",
            "pdf_ingestion_ms",
            "prompt_construction_preflight_ms",
            "model_api_request_response_ms",
            "pydantic_response_parsing_ms",
            "evidence_validation_ms",
        )
    )


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_EXTRACTION_C02") != "1",
    reason="Set RUN_LIVE_EXTRACTION_C02=1 to call the configured live model",
)
def test_live_c02_rate_extraction_matches_expected_candidates():
    started = perf_counter()
    try:
        if not os.environ.get("GEMINI_API_KEY", "").strip():
            pytest.fail(
                "RUN_LIVE_EXTRACTION_C02=1 requires GEMINI_API_KEY in the environment "
                "or .env; the key will not be displayed.",
                pytrace=False,
            )

        run = extract_folder(CONTRACTS / "c02", client=_create_client())
    except ExtractionError as exc:
        elapsed = round(perf_counter() - started, 3)
        _report_extraction_error(exc, elapsed)
    elapsed = run.timings.total_extraction_ms / 1000

    # Read expected answers only after the actual model extraction has completed.
    expected_path = ROOT / "expected" / "c02.json"
    expected_rates = json.loads(expected_path.read_text(encoding="utf-8"))["rates"]
    expected_sources = {
        (rate["source"]["file"], rate["source"]["page"])
        for rate in expected_rates
    }
    rate_records = [
        record for record in run.candidate_records if record.category == "room_rate"
    ]
    assessments = {
        assessment.candidate_id: assessment
        for assessment in run.evidence_assessments
    }
    table_records = [
        record
        for record in rate_records
        if (record.candidate.source.file, record.candidate.source.page)
        in expected_sources
    ]
    other_records = [
        record
        for record in rate_records
        if (record.candidate.source.file, record.candidate.source.page)
        not in expected_sources
    ]
    table_candidates = [record.candidate for record in table_records]
    other_candidates = [record.candidate for record in other_records]
    comparison = _compare_rates(expected_rates, table_candidates)
    rate_type_errors = [
        _rate_report_record(rate)
        for rate in table_candidates
        if rate.rate_type != "individual"
    ]
    unexpected_source_candidates = [
        _rate_report_record(rate)
        for rate in other_candidates
        if rate.rate_type in {"individual", "unspecified"}
    ]
    incorrect_candidates = [
        *comparison["incorrect_candidates"],
        *unexpected_source_candidates,
    ]

    rate_status_counts = Counter(
        assessments[record.candidate_id].status for record in rate_records
    )
    rate_diagnostics = _rate_diagnostics(
        rate_records,
        assessments,
        expected_rates,
        expected_sources,
    )
    report = {
        "model": run.model,
        "expected_candidate_count": len(expected_rates),
        "extracted_candidate_count": len(rate_records),
        "verified_candidate_count": rate_status_counts["verified"],
        "uncertain_candidate_count": rate_status_counts["uncertain"],
        "invalid_candidate_count": rate_status_counts["invalid"],
        "compared_table_candidate_count": len(table_candidates),
        "exact_match_count": len(comparison["exact_matches"]),
        "exact_matches": comparison["exact_matches"],
        "missing_candidates": comparison["missing_candidates"],
        "incorrect_candidates": incorrect_candidates,
        "duplicate_candidates": comparison["duplicate_candidates"],
        "rate_type_errors": rate_type_errors,
        **rate_diagnostics,
        "fixture_match_note": (
            "A fixture-value match is independent of evidence assessment; it does "
            "not imply that a candidate is verified."
        ),
        "other_rate_candidates": [
            _rate_report_record(rate)
            for rate in other_candidates
            if rate.rate_type in {"group", "rack"}
        ],
        "source_citation_validity_counts": dict(rate_status_counts),
        "expected_table_page_validity": (
            "valid" if not unexpected_source_candidates else "invalid"
        ),
        "token_usage": {
            "input_tokens": run.usage.input_tokens,
            "output_tokens": run.usage.output_tokens,
            "total_tokens": run.usage.total_tokens,
            "estimated_cost_usd": run.usage.estimated_cost_usd,
        },
        "stage_timings": _timing_report(run.timings),
        "timing_notes": [
            "Gemini generation runs inside the API call; local JSON and Pydantic "
            "response parsing is measured separately.",
            "pydantic_response_parsing_ms measures application-side response "
            "inspection and any validation after the SDK returns.",
            "rendering_ms and ocr_ms are subtotals of pdf_ingestion_ms.",
        ],
        "elapsed_seconds": elapsed,
    }
    print("LIVE C02 EXTRACTION EVALUATION")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    assert not comparison["missing_candidates"], json.dumps(report, indent=2)
    assert not incorrect_candidates, json.dumps(report, indent=2)
    assert not comparison["duplicate_candidates"], json.dumps(report, indent=2)
    assert not rate_type_errors, json.dumps(report, indent=2)
