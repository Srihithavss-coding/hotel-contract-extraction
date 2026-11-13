import json
import os
import random
from datetime import date, datetime, timezone
from time import perf_counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from extract import (
    AdvancePaymentCandidate,
    CandidateEvidenceAssessment,
    CandidateRecord,
    CancellationCandidate,
    CommercialTermCandidate,
    ContractFactCandidate,
    EvidenceSource,
    ExtractionError,
    ExtractionResult,
    ExtractionRun,
    RoomRateCandidate,
    SeasonCandidate,
    TokenUsage,
    _validate_sources,
    _create_client,
    extract_folder,
)
from ingest import PageContent, PageLine
from reconcile import (
    ReconciliationError,
    _normalise_meal_plan,
    reconcile_run,
    validate_output,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "expected"
_MISSING = object()


def _source(file="rate-letter.pdf", page=1, quote="evidence"):
    return EvidenceSource(file=file, page=page, quote=quote)


def _run(items, candidate_ids=None):
    records = []
    assessments = []
    for index, item in enumerate(items):
        candidate, status = item[:2]
        schedule_id = item[2] if len(item) > 2 else None
        issues = item[3] if len(item) > 3 else ()
        category = _category(candidate)
        candidate_id = (
            candidate_ids[index]
            if candidate_ids is not None
            else f"candidate-{index}"
        )
        records.append(
            CandidateRecord(
                candidate_id=candidate_id,
                category=category,
                candidate=candidate,
                schedule_id=schedule_id,
            )
        )
        assessments.append(
            CandidateEvidenceAssessment(
                candidate_id=candidate_id,
                category=category,
                status=status,
                issues=issues,
            )
        )
    return ExtractionRun(
        model="synthetic-test-model",
        candidate_records=tuple(records),
        evidence_assessments=tuple(assessments),
        usage=TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0,
        ),
    )


def _category(candidate):
    if isinstance(candidate, ContractFactCandidate):
        return "contract_fact"
    if isinstance(candidate, SeasonCandidate):
        return "season"
    if isinstance(candidate, RoomRateCandidate):
        return "room_rate"
    if isinstance(candidate, CancellationCandidate):
        return "cancellation"
    if isinstance(candidate, AdvancePaymentCandidate):
        return "advance_payment"
    if isinstance(candidate, CommercialTermCandidate):
        return "commercial_term"
    raise TypeError(type(candidate))


def _rate(
    *,
    room="Garden Room",
    plan="CP",
    occupancy="single",
    price=100,
    start=date(2026, 1, 1),
    end=date(2026, 3, 31),
    season="Winter",
    source=None,
    rate_type="unspecified",
    occupancy_applicability="unknown",
):
    quote = source or f"{room} {plan} {price}"
    return RoomRateCandidate(
        room=room,
        meal_plan=plan,
        occupancy=occupancy,
        occupancy_applicability=occupancy_applicability,
        price=price,
        season=season,
        start_date=start,
        end_date=end,
        rate_type=rate_type,
        source=_source(quote=quote),
    )


def _fact(field, value, quote, file="rate-letter.pdf", page=1):
    return ContractFactCandidate(
        field=field,
        value=value,
        source=_source(file=file, page=page, quote=quote),
    )


def _text_page(text):
    return PageContent(
        file="rate-letter.pdf",
        page=1,
        text=text,
        extraction_method="text",
        ocr_confidence=None,
        lines=[PageLine(text=line) for line in text.splitlines()],
    )


def test_uncertain_and_invalid_rates_never_enter_loaded_output():
    verified = _rate(source="Garden Room CP 100")
    uncertain = _rate(price=200, source="Garden Room CP 200")
    invalid = _rate(price=300, source="Garden Room CP 300")
    output = reconcile_run(
        _run(
            [
                (verified, "verified"),
                (uncertain, "uncertain"),
                (invalid, "invalid"),
            ]
        )
    )

    assert len(output["rates"]) == 1
    assert output["rates"][0]["price"] == 100


def test_missing_assessment_fails_before_reconciliation():
    candidate = _fact("hotel_name", "Hotel Cedar Bay", "Hotel Cedar Bay")
    record = CandidateRecord(
        candidate_id="missing-assessment",
        category="contract_fact",
        candidate=candidate,
    )

    with pytest.raises(ValidationError, match="Assessment references are incomplete"):
        ExtractionRun(
            model="test",
            candidate_records=(record,),
            evidence_assessments=(),
            usage=TokenUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                estimated_cost_usd=0,
            ),
        )


def test_conflicting_contract_facts_are_omitted_and_questioned():
    output = reconcile_run(
        _run(
            [
                (_fact("tax_included", True, "Rates include tax."), "verified"),
                (
                    _fact("tax_included", False, "GST extra.", file="email.pdf"),
                    "verified",
                ),
            ]
        )
    )

    assert "taxes_included" not in output["contract"]
    assert [question["about"] for question in output["questions"]] == [
        "taxes_included"
    ]
    assert len(output["questions"][0]["sources"]) == 2


def test_duplicate_conflict_evidence_does_not_duplicate_questions():
    items = [
        (_fact("hotel_name", "Hotel Cedar Bay", "Hotel Cedar Bay"), "verified"),
        (_fact("hotel_name", "Hotel Mango Cove", "Hotel Mango Cove", "email.pdf"), "verified"),
        (_fact("hotel_name", "Hotel Mango Cove", "Hotel Mango Cove, reiterated", "email.pdf", 2), "verified"),
    ]
    output = reconcile_run(_run(items))

    assert "hotel" not in output["contract"]
    assert sum(question["about"] == "hotel" for question in output["questions"]) == 1


def test_overlapping_seasons_omit_both_periods_and_keep_unaffected_rates():
    summer = SeasonCandidate(
        name="Summer",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 6, 30),
        source=_source(quote="Summer: 1 April 2026 to 30 June 2026"),
    )
    monsoon = SeasonCandidate(
        name="Monsoon",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 10, 15),
        source=_source(quote="Monsoon: 1 July 2026 to 15 October 2026"),
    )
    winter = SeasonCandidate(
        name="Winter",
        start_date=date(2026, 10, 10),
        end_date=date(2027, 3, 31),
        source=_source(quote="Winter: 10 October 2026 to 31 March 2027"),
    )
    output = reconcile_run(
        _run(
            [
                (_fact("validity_start", "2026-04-01", "Contract validity: 1 April 2026 to 31 March 2027"), "verified"),
                (_fact("validity_end", "2027-03-31", "Contract validity: 1 April 2026 to 31 March 2027"), "verified"),
                (summer, "verified"),
                (monsoon, "verified"),
                (winter, "verified"),
                (_rate(start=summer.start_date, end=summer.end_date, season="Summer", source="Garden Room CP 100"), "verified"),
                (_rate(start=monsoon.start_date, end=monsoon.end_date, season="Monsoon", source="Garden Room CP 200"), "verified"),
                (_rate(start=winter.start_date, end=winter.end_date, season="Winter", source="Garden Room CP 300"), "verified"),
            ]
        )
    )

    assert [rate["price"] for rate in output["rates"]] == [100]
    rate_questions = [item for item in output["questions"] if item["about"] == "rates"]
    assert len(rate_questions) == 1
    assert "overlaps" in rate_questions[0]["why"]
    assert len(rate_questions[0]["sources"]) == 2


def test_gap_is_reported_without_discarding_unaffected_season_rates():
    first = SeasonCandidate(
        name="Early",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        source=_source(quote="Early season: 1 January 2026 to 31 March 2026"),
    )
    second = SeasonCandidate(
        name="Late",
        start_date=date(2026, 5, 1),
        end_date=date(2026, 7, 31),
        source=_source(quote="Late season: 1 May 2026 to 31 July 2026"),
    )
    output = reconcile_run(
        _run(
            [
                (first, "verified"),
                (second, "verified"),
                (
                    _rate(
                        start=first.start_date,
                        end=first.end_date,
                        season="Early",
                        source="Garden Room CP 100",
                    ),
                    "verified",
                ),
                (
                    _rate(
                        start=second.start_date,
                        end=second.end_date,
                        season="Late",
                        price=200,
                        source="Garden Room CP 200",
                    ),
                    "verified",
                ),
            ]
        )
    )

    assert {rate["price"] for rate in output["rates"]} == {100, 200}, (
        output["questions"],
        output["rates"],
    )
    assert any("uncovered date gap" in item["why"] for item in output["questions"])


def test_unspecified_occupancy_expands_to_single_and_double_with_same_source():
    quote = "Heritage Room 3,200 3,600 4,400"
    candidate = RoomRateCandidate(
        room="Heritage Room",
        meal_plan="EP",
        occupancy="unspecified",
        occupancy_applicability="single_and_double",
        price=3200,
        season="Summer",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 9, 30),
        rate_type="unspecified",
        source=_source(quote=quote),
    )
    output = reconcile_run(_run([(candidate, "verified")]))

    assert {rate["occupancy"] for rate in output["rates"]} == {"single", "double"}
    assert all(rate["source"]["quote"] == quote for rate in output["rates"])


def test_unspecified_occupancy_requires_explicit_single_double_applicability():
    candidate = _rate(
        occupancy="unspecified",
        occupancy_applicability="unknown",
        source="Garden Room CP 100",
    )

    output = reconcile_run(_run([(candidate, "verified")]))

    assert output["rates"] == []
    assert len(output["not_loaded"]) == 1
    assert "both single and double" in output["not_loaded"][0]["reason"]


def test_explicit_per_room_applicability_allows_both_occupancies():
    candidate = _rate(
        occupancy="unspecified",
        occupancy_applicability="single_and_double",
        source="Garden Room CP 100 per room per night",
    )

    output = reconcile_run(_run([(candidate, "verified")]))

    assert {rate["occupancy"] for rate in output["rates"]} == {"single", "double"}


def test_conflicting_explicit_single_and_double_prices_block_unspecified_expansion():
    unspecified = _rate(
        occupancy="unspecified",
        occupancy_applicability="single_and_double",
        source="Garden Room CP 100 per room per night",
    )
    single = _rate(occupancy="single", price=100, source="Garden Room CP single 100")
    double = _rate(occupancy="double", price=150, source="Garden Room CP double 150")

    output = reconcile_run(
        _run(
            [
                (unspecified, "verified"),
                (single, "verified"),
                (double, "verified"),
            ]
        )
    )

    assert {(rate["occupancy"], rate["price"]) for rate in output["rates"]} == {
        ("single", 100),
        ("double", 150),
    }


def test_group_and_rack_rates_are_not_loaded_and_keep_citations():
    group = _rate(
        room="Garden Room",
        plan="MAP",
        occupancy="unspecified",
        price=6500,
        source="Special group rate of INR 6,500 per room per night on MAP.",
        rate_type="group",
    )
    rack = CommercialTermCandidate(
        kind="rack_rate",
        description="Published rack rates for reference only",
        source=_source(
            file="contract.pdf",
            page=4,
            quote="Published rack rates for reference only. Not applicable to this agreement.",
        ),
    )
    output = reconcile_run(_run([(group, "verified"), (rack, "verified")]))

    assert output["rates"] == []
    assert len(output["not_loaded"]) == 2
    assert {item["source"]["file"] for item in output["not_loaded"]} == {
        "rate-letter.pdf",
        "contract.pdf",
    }


def test_group_rate_duplicates_across_candidate_categories_are_deduplicated():
    quote = "Special group rate of INR 6,500 per room per night on MAP applies."
    rate = _rate(
        room="Garden Room",
        plan="MAP",
        occupancy="unspecified",
        price=6500,
        source=quote,
        rate_type="group",
    )
    term = CommercialTermCandidate(
        kind="group_rate",
        description="Special group rate of INR 6,500 on MAP",
        amount=6500,
        currency="INR",
        source=_source(quote=quote),
    )

    output = reconcile_run(_run([(rate, "verified"), (term, "verified")]))

    assert len(output["not_loaded"]) == 1
    assert output["not_loaded"][0]["source"]["quote"] == quote


def test_equivalent_group_terms_with_surrounding_quote_differences_are_deduplicated():
    full_quote = (
        "Garden Room special group rate of INR 6,500 per room per night on MAP applies."
    )
    short_quote = "Garden Room group rate INR 6,500 on MAP"
    rate = _rate(
        room="Garden Room",
        plan="MAP",
        occupancy="unspecified",
        price=6500,
        source=full_quote,
        rate_type="group",
    )
    term = CommercialTermCandidate(
        kind="group_rate",
        description="Garden Room group rate",
        amount=6500,
        currency="INR",
        room="Garden Room",
        meal_plan="MAP",
        occupancy="unspecified",
        source=_source(quote=short_quote),
    )

    output = reconcile_run(_run([(term, "verified"), (rate, "verified")]))

    assert len(output["not_loaded"]) == 1


def test_distinct_room_specific_group_rates_are_not_deduplicated():
    candidates = [
        _rate(
            room=room,
            plan="MAP",
            occupancy="unspecified",
            price=6500,
            source=f"{room} group rate INR 6,500 on MAP",
            rate_type="group",
        )
        for room in ("Garden Room", "Sea View Room")
    ]

    output = reconcile_run(_run([(candidate, "verified") for candidate in candidates]))

    assert len(output["not_loaded"]) == 2
    assert {item["text"].split(" group rate")[0] for item in output["not_loaded"]} == {
        "Garden Room",
        "Sea View Room",
    }


def test_not_loaded_order_is_independent_of_candidate_input_order():
    candidates = [
        _rate(
            room=room,
            plan="MAP",
            occupancy="unspecified",
            price=price,
            source=f"{room} group rate INR {price} on MAP",
            rate_type="group",
        )
        for room, price in (("Sea View Room", 7000), ("Garden Room", 6500))
    ]

    first = reconcile_run(_run([(candidate, "verified") for candidate in candidates]))
    second = reconcile_run(
        _run([(candidate, "verified") for candidate in reversed(candidates)])
    )

    assert first["not_loaded"] == second["not_loaded"]


def test_similar_group_rate_terms_with_distinct_evidence_are_not_merged():
    first_quote = "Group rate of INR 6,500 per room per night on MAP."
    second_quote = "Group rate of INR 7,000 per room per night on MAP."
    candidates = [
        _rate(
            room="Garden Room",
            plan="MAP",
            occupancy="unspecified",
            price=price,
            source=quote,
            rate_type="group",
        )
        for price, quote in ((6500, first_quote), (7000, second_quote))
    ]

    output = reconcile_run(
        _run([(candidate, "verified") for candidate in candidates])
    )

    assert len(output["not_loaded"]) == 2
    assert {item["source"]["quote"] for item in output["not_loaded"]} == {
        first_quote,
        second_quote,
    }


def test_generic_extra_bed_rate_expands_only_for_all_periods():
    rooms = ("Plantation View Room", "Ridge Cottage")
    periods = (
        (date(2026, 4, 1), date(2026, 6, 30)),
        (date(2026, 7, 1), date(2026, 9, 30)),
    )
    items = []
    for room in rooms:
        for start, end in periods:
            items.append(
                (
                    _rate(
                        room=room,
                        start=start,
                        end=end,
                        season=None,
                        source=f"{room} CP 5200",
                    ),
                    "verified",
                )
            )
    extra_quote = (
        "Extra bed for a third guest, all periods: INR 1,800 per night on CP."
    )
    extra_bed = RoomRateCandidate(
        room=None,
        meal_plan="CP",
        occupancy="extra_bed",
        price=1800,
        rate_type="unspecified",
        source=_source(file="email.pdf", quote=extra_quote),
    )
    output = reconcile_run(_run(items + [(extra_bed, "verified")]))
    expanded = [rate for rate in output["rates"] if rate["occupancy"] == "extra_bed"]

    assert len(expanded) == len(rooms) * len(periods)
    assert {rate["room"] for rate in expanded} == set(rooms)
    assert all(rate["source"]["file"] == "email.pdf" for rate in expanded)
    assert all(rate["source"]["quote"] == extra_quote for rate in expanded)


def _extra_bed_supporting_rates():
    rooms = ("Plantation View Room", "Ridge Cottage")
    periods = (
        (date(2026, 4, 1), date(2026, 6, 30)),
        (date(2026, 7, 1), date(2026, 9, 30)),
    )
    items = [
        (
            _rate(
                room=room,
                start=start,
                end=end,
                season=None,
                source=f"{room} CP 5200",
            ),
            "verified",
        )
        for room in rooms
        for start, end in periods
    ]
    return rooms, periods, items


def _extra_bed_candidate(quote, **overrides):
    values = {
        "room": None,
        "meal_plan": "CP",
        "occupancy": "extra_bed",
        "price": 1800,
        "rate_type": "unspecified",
        "source": _source(file="email.pdf", quote=quote),
    }
    values.update(overrides)
    return RoomRateCandidate(**values)


def test_assignment_rule_applies_unspecified_third_guest_extra_bed_rate_to_all_rooms():
    rooms, periods, rates = _extra_bed_supporting_rates()
    extra_bed = _extra_bed_candidate(
        "Extra bed for a third guest, all periods: INR 1,800 per night on CP."
    )

    output = reconcile_run(_run(rates + [(extra_bed, "verified")]))
    expanded = [rate for rate in output["rates"] if rate["occupancy"] == "extra_bed"]

    assert len(expanded) == len(rooms) * len(periods)
    assert {rate["room"] for rate in expanded} == set(rooms)


def test_assignment_room_scope_does_not_infer_missing_period_scope():
    _, _, rates = _extra_bed_supporting_rates()
    extra_bed = _extra_bed_candidate(
        "Extra bed for a third guest: INR 1,800 per night on CP."
    )

    output = reconcile_run(_run(rates + [(extra_bed, "verified")]))

    assert not any(rate["occupancy"] == "extra_bed" for rate in output["rates"])


def test_specific_room_scope_overrides_assignment_all_room_rule():
    rooms, periods, rates = _extra_bed_supporting_rates()
    extra_bed = _extra_bed_candidate(
        "Extra bed for a third guest at Ridge Cottage, all periods: "
        "INR 1,800 per night on CP."
    )

    output = reconcile_run(_run(rates + [(extra_bed, "verified")]))
    expanded = [rate for rate in output["rates"] if rate["occupancy"] == "extra_bed"]

    assert rooms == ("Plantation View Room", "Ridge Cottage")
    assert len(expanded) == len(periods)
    assert {rate["room"] for rate in expanded} == {"Ridge Cottage"}


def test_all_room_types_do_not_prove_extra_bed_period_coverage():
    _, _, rates = _extra_bed_supporting_rates()
    extra_bed = _extra_bed_candidate(
        "Extra bed for all room types: INR 1,800 per night on CP."
    )

    output = reconcile_run(_run(rates + [(extra_bed, "verified")]))

    assert not any(rate["occupancy"] == "extra_bed" for rate in output["rates"])


def test_extra_bed_expands_when_both_scopes_are_explicit():
    rooms, periods, rates = _extra_bed_supporting_rates()
    quote = (
        "Extra bed applies to all room types and all periods: "
        "INR 1,800 per night on CP."
    )
    extra_bed = _extra_bed_candidate(quote)

    output = reconcile_run(_run(rates + [(extra_bed, "verified")]))
    expanded = [rate for rate in output["rates"] if rate["occupancy"] == "extra_bed"]

    assert len(expanded) == len(rooms) * len(periods)
    assert all(rate["source"]["quote"] == quote for rate in expanded)


def test_extra_bed_with_one_room_and_period_does_not_expand_other_scopes():
    _, periods, rates = _extra_bed_supporting_rates()
    quote = "Ridge Cottage extra bed CP 1,800 for Summer."
    extra_bed = _extra_bed_candidate(
        quote,
        room="Ridge Cottage",
        start_date=periods[0][0],
        end_date=periods[0][1],
        season=None,
    )

    output = reconcile_run(_run(rates + [(extra_bed, "verified")]))
    expanded = [rate for rate in output["rates"] if rate["occupancy"] == "extra_bed"]

    assert len(expanded) == 1
    assert expanded[0]["room"] == "Ridge Cottage"
    assert expanded[0]["start"] == periods[0][0].isoformat()
    assert expanded[0]["source"]["quote"] == quote


def test_validator_accepts_complete_selectable_text_table_rate_without_occ_column():
    quote = "Heritage Room individual rates Room Only 3,200 3,600 4,400"
    page_text = (
        "Contracted rates\n"
        "All rates are quoted in INR per room per night.\n"
        "Summer season: 1 April 2026 to 30 September 2026\n"
        "Room Category\nRoom Only\nWith Breakfast\nBreakfast & Dinner\n"
        f"{quote}"
    )
    candidate = RoomRateCandidate(
        room="Heritage Room",
        meal_plan="Room Only",
        occupancy="unspecified",
        occupancy_applicability="single_and_double",
        price=3200,
        season="Summer",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 9, 30),
        rate_type="individual",
        source=_source(quote=quote),
    )
    page = _text_page(page_text)

    assessment = _validate_sources(ExtractionResult(room_rates=[candidate]), [page])[0]

    assert assessment.status == "verified", assessment.issues


def test_cancellation_conflict_is_omitted_and_questioned():
    output = reconcile_run(
        _run(
            [
                (
                    CancellationCandidate(
                        charge_percent=100,
                        days_max=21,
                        source=_source(
                            file="rate-letter.pdf",
                            page=2,
                            quote="Cancellation: 100% within 21 days of arrival.",
                        ),
                    ),
                    "verified",
                ),
                (
                    CancellationCandidate(
                        charge_percent=100,
                        days_max=30,
                        source=_source(
                            file="email.pdf",
                            page=1,
                            quote="Cancellation: 100% within 30 days of arrival.",
                        ),
                    ),
                    "verified",
                ),
            ]
        )
    )

    assert "cancellation" not in output["policies"]
    assert len(output["questions"]) == 1
    assert output["questions"][0]["about"] == "cancellation"
    assert len(output["questions"][0]["sources"]) == 2


def test_advance_balance_is_calculated_only_from_explicit_advance_and_balance():
    quote = (
        "25% advance at the time of confirmation. "
        "Balance payable 7 days before arrival."
    )
    candidate = AdvancePaymentCandidate(
        percentage=25,
        timing="at the time of confirmation",
        balance_wording="Balance payable 7 days before arrival.",
        source=_source(quote=quote),
    )
    page = _text_page(quote)
    assessment = _validate_sources(ExtractionResult(advance_payments=[candidate]), [page])[0]
    assert assessment.status == "verified"

    output = reconcile_run(_run([(candidate, "verified")]))

    assert output["policies"]["advance_payment"] == [
        {
            "percent": 25,
            "when": "at_booking",
            "days_before_arrival": None,
            "source": candidate.source.model_dump(mode="json"),
        },
        {
            "percent": 75,
            "when": "before_arrival",
            "days_before_arrival": 7,
            "source": candidate.source.model_dump(mode="json"),
        },
    ]


def _advance_candidate(percentage, timing, quote, balance_wording=None):
    return AdvancePaymentCandidate(
        percentage=percentage,
        timing=timing,
        balance_wording=balance_wording,
        source=_source(quote=quote),
    )


def test_advance_schedule_preserves_explicit_25_and_75_percent_installments():
    quote = "25% advance at booking. Balance 75% due 7 days before arrival."
    candidate = _advance_candidate(
        25,
        "at booking",
        quote,
        balance_wording="Balance 75% due 7 days before arrival.",
    )

    output = reconcile_run(_run([(candidate, "verified")]))

    assert [entry["percent"] for entry in output["policies"]["advance_payment"]] == [
        25,
        75,
    ]


def test_c01_style_separate_advance_and_balance_sentences_share_schedule_id():
    advance = _advance_candidate(
        50,
        "at the time of confirmation",
        "50% of the booking value is payable at the time of confirmation.",
    )
    balance = _advance_candidate(
        None,
        None,
        "The balance 50% is payable 21 days before arrival.",
        balance_wording="The balance 50% is payable 21 days before arrival.",
    )

    output = reconcile_run(
        _run(
            [
                (advance, "verified", "payment-section:10"),
                (balance, "verified", "payment-section:10"),
            ]
        )
    )

    assert [entry["percent"] for entry in output["policies"]["advance_payment"]] == [
        50,
        50,
    ]


def test_explicit_advance_schedule_over_100_percent_is_omitted_and_questioned():
    quote = "25% advance at booking. Balance 80% due 7 days before arrival."
    candidate = _advance_candidate(
        25,
        "at booking",
        quote,
        balance_wording="Balance 80% due 7 days before arrival.",
    )

    output = reconcile_run(_run([(candidate, "verified")]))

    assert "advance_payment" not in output["policies"]
    assert [question["about"] for question in output["questions"]] == [
        "advance_payment"
    ]


def test_multiple_installments_in_one_schedule_must_total_100_percent():
    quote = (
        "25% at booking; 50% 30 days before arrival; "
        "25% 7 days before arrival."
    )
    candidates = [
        (_advance_candidate(25, "at booking", quote), "verified", "schedule-a"),
        (
            _advance_candidate(50, "30 days before arrival", quote),
            "verified",
            "schedule-a",
        ),
        (
            _advance_candidate(25, "7 days before arrival", quote),
            "verified",
            "schedule-a",
        ),
    ]

    output = reconcile_run(_run(candidates))

    assert sorted(
        entry["percent"] for entry in output["policies"]["advance_payment"]
    ) == [25, 25, 50]
    assert all(
        entry["source"]["quote"] == quote
        for entry in output["policies"]["advance_payment"]
    )


def test_multiple_installments_over_100_percent_are_rejected():
    quote = "60% at booking; 50% 7 days before arrival."
    candidates = [
        (_advance_candidate(60, "at booking", quote), "verified", "schedule-a"),
        (
            _advance_candidate(50, "7 days before arrival", quote),
            "verified",
            "schedule-a",
        ),
    ]

    output = reconcile_run(_run(candidates))

    assert "advance_payment" not in output["policies"]
    assert any(question["about"] == "advance_payment" for question in output["questions"])


def test_overlapping_quotes_do_not_count_the_same_installment_twice():
    first = _advance_candidate(
        25,
        "at booking",
        "25% advance at booking. Balance 75% due 7 days before arrival.",
        balance_wording="Balance 75% due 7 days before arrival.",
    )
    overlap = _advance_candidate(
        75,
        "7 days before arrival",
        "Balance 75% due 7 days before arrival.",
    )

    output = reconcile_run(
        _run(
            [
                (first, "verified", "schedule-a"),
                (overlap, "verified", "schedule-a"),
            ]
        )
    )

    assert sorted(
        entry["percent"] for entry in output["policies"]["advance_payment"]
    ) == [25, 75]


def test_decimal_installments_are_compared_exactly():
    quote = "33.333% at booking; 33.333% later; 33.334% before arrival."
    candidates = [
        (_advance_candidate(33.333, "at booking", quote), "verified", "decimal-schedule"),
        (
            _advance_candidate(33.333, "30 days before arrival", quote),
            "verified",
            "decimal-schedule",
        ),
        (
            _advance_candidate(33.334, "7 days before arrival", quote),
            "verified",
            "decimal-schedule",
        ),
    ]

    output = reconcile_run(_run(candidates))

    assert len(output["policies"]["advance_payment"]) == 3


def test_installments_from_separate_clauses_are_not_combined_by_filename():
    first = _advance_candidate(
        60,
        "at booking",
        "Schedule A: 60% at booking.",
    )
    second = _advance_candidate(
        40,
        "30 days before arrival",
        "Schedule B: 40% 30 days before arrival.",
    )

    output = reconcile_run(
        _run([(first, "verified"), (second, "verified")])
    )

    assert "advance_payment" not in output["policies"]
    assert any(question["about"] == "advance_payment" for question in output["questions"])


def test_distinct_complete_payment_schedules_are_asked_about_not_combined():
    first = _advance_candidate(
        100,
        "30 days before arrival",
        "Schedule A: 100% advance, 30 days before arrival.",
    )
    second = _advance_candidate(
        100,
        "at booking",
        "Schedule B: 100% advance at booking.",
    )

    output = reconcile_run(
        _run([(first, "verified"), (second, "verified")])
    )

    assert "advance_payment" not in output["policies"]
    assert any(question["about"] == "advance_payment" for question in output["questions"])


def test_incomplete_advance_schedule_does_not_infer_the_remaining_balance():
    candidate = _advance_candidate(
        25,
        "at booking",
        "25% advance payable at booking.",
    )

    output = reconcile_run(_run([(candidate, "verified")]))

    assert "advance_payment" not in output["policies"]
    assert any(question["about"] == "advance_payment" for question in output["questions"])


def test_meal_plan_aliases_normalize_to_company_codes():
    assert _normalise_meal_plan("Room Only") == "EP"
    assert _normalise_meal_plan("Breakfast") == "CP"
    assert _normalise_meal_plan("BB") == "CP"
    assert _normalise_meal_plan("Breakfast and Dinner") == "MAP"
    assert _normalise_meal_plan("Half Board") == "MAP"
    assert _normalise_meal_plan("Full Board") == "AP"


def test_duplicate_rate_candidate_is_deduplicated_and_citation_is_preserved():
    candidate = _rate(source="Garden Room CP 100")
    output = reconcile_run(
        _run([(candidate, "verified"), (candidate, "verified")])
    )

    assert len(output["rates"]) == 1
    assert output["rates"][0]["source"] == candidate.source.model_dump(mode="json")


def test_schema_validation_rejects_unsupported_output_properties():
    output = {
        "contract": {},
        "rates": [],
        "policies": {},
        "questions": [],
        "not_loaded": [],
        "evidence_status": "verified",
    }

    with pytest.raises(ReconciliationError, match="schema validation"):
        validate_output(output)


def test_uncertain_tax_candidate_can_question_without_being_loaded():
    quote = "Rates are subject to applicable GST treatment."
    candidate = _fact("tax_included", True, quote)

    output = reconcile_run(
        _run(
            [
                (
                    candidate,
                    "uncertain",
                    None,
                    ("Quote does not clearly state whether taxes are included.",),
                )
            ]
        )
    )

    assert "taxes_included" not in output["contract"]
    assert len(output["questions"]) == 1
    assert output["questions"][0]["about"] == "taxes_included"
    assert output["questions"][0]["sources"][0]["quote"] == quote


def test_uncertain_rate_can_question_occupancy_but_never_load_rate():
    candidate = _rate(
        occupancy="unspecified",
        source="Garden Room CP 100",
    )

    output = reconcile_run(
        _run(
            [
                (
                    candidate,
                    "uncertain",
                    None,
                    ("Occupancy is not explicitly supported by the source text.",),
                )
            ]
        )
    )

    assert output["rates"] == []
    assert len(output["questions"]) == 1
    assert output["questions"][0]["about"] == "rates"
    assert output["questions"][0]["sources"][0]["quote"] == "Garden Room CP 100"


def test_invalid_candidate_does_not_generate_clarification_question():
    candidate = _fact(
        "tax_included",
        True,
        "Rates are subject to applicable GST treatment.",
    )

    output = reconcile_run(
        _run(
            [
                (
                    candidate,
                    "invalid",
                    None,
                    ("Quote does not clearly state whether taxes are included.",),
                )
            ]
        )
    )

    assert output["contract"] == {}
    assert output["questions"] == []


def test_non_material_uncertainty_does_not_generate_question():
    candidate = _fact(
        "hotel_name",
        "Hotel Cedar Bay",
        "Children may use the pool from 10 AM.",
    )

    output = reconcile_run(
        _run([(candidate, "uncertain", None, ("Name is unclear.",))])
    )

    assert output["questions"] == []


def test_relevant_child_policy_is_retained_in_not_loaded_with_citation():
    quote = "Children under 6 stay free when sharing existing bedding."
    candidate = CommercialTermCandidate(
        kind="child_policy",
        description="Children under six stay free.",
        source=_source(quote=quote),
    )

    output = reconcile_run(_run([(candidate, "verified")]))

    assert len(output["not_loaded"]) == 1
    assert output["not_loaded"][0]["text"] == quote
    assert output["not_loaded"][0]["source"]["quote"] == quote
    assert "not represented in the company output schema" in output["not_loaded"][0]["reason"]


def test_invalid_unsupported_child_price_is_not_emitted():
    quote = "Children under 6 stay free."
    candidate = CommercialTermCandidate(
        kind="child_policy",
        description="Children under 6 pay INR 900.",
        amount=900,
        currency="INR",
        source=_source(quote=quote),
    )

    output = reconcile_run(_run([(candidate, "invalid")]))

    assert output["not_loaded"] == []


def test_duplicate_child_policy_terms_are_deduplicated_but_irrelevant_text_is_omitted():
    quote = "Children under 6 stay free when sharing existing bedding."
    first = CommercialTermCandidate(
        kind="child_policy",
        description="Free stay for children below six.",
        source=_source(quote=quote),
    )
    duplicate = CommercialTermCandidate(
        kind="child_policy",
        description="Children stay free if sharing bedding.",
        source=_source(quote=quote),
    )
    cross_category_duplicate = CommercialTermCandidate(
        kind="exclusion",
        description="Children under six stay free.",
        source=_source(quote=quote),
    )
    irrelevant = CommercialTermCandidate(
        kind="child_policy",
        description="Children may use the pool.",
        source=_source(quote="Children may use the pool from 10 AM."),
    )

    output = reconcile_run(
        _run(
            [
                (first, "verified"),
                (duplicate, "verified"),
                (cross_category_duplicate, "verified"),
                (irrelevant, "verified"),
            ]
        )
    )

    assert len(output["not_loaded"]) == 1
    assert output["not_loaded"][0]["source"]["quote"] == quote


def _ordering_candidates():
    return [
        (_fact("tax_included", True, "Rates include GST.", "terms-a.pdf"), "verified"),
        (_fact("tax_included", False, "GST extra.", "terms-b.pdf"), "verified"),
        (_fact("currency", "INR", "All prices in INR.", "terms-a.pdf"), "verified"),
        (_fact("currency", "USD", "All prices in USD.", "terms-b.pdf"), "verified"),
        (
            _rate(
                price=100,
                source="Garden Room CP single 100",
            ),
            "verified",
        ),
        (
            _rate(
                price=110,
                source="Garden Room CP single 110",
            ),
            "verified",
        ),
        (
            CancellationCandidate(
                charge_percent=50,
                days_max=7,
                source=_source(file="terms-a.pdf", quote="50% cancellation within 7 days."),
            ),
            "verified",
        ),
        (
            CancellationCandidate(
                charge_percent=100,
                days_max=7,
                source=_source(file="terms-b.pdf", quote="100% cancellation within 7 days."),
            ),
            "verified",
        ),
        (
            _advance_candidate(
                100,
                "at booking",
                "Schedule A: 100% is payable at booking.",
            ),
            "verified",
            "schedule-a",
        ),
        (
            _advance_candidate(
                100,
                "30 days before arrival",
                "Schedule B: 100% is payable 30 days before arrival.",
            ),
            "verified",
            "schedule-b",
        ),
        (
            CommercialTermCandidate(
                kind="group_rate",
                description="Group rate INR 6500.",
                amount=6500,
                currency="INR",
                room="Garden Room",
                source=_source(file="terms-a.pdf", quote="Group rate INR 6500 Garden Room."),
            ),
            "verified",
        ),
    ]


def test_reconciliation_output_is_stable_across_reversed_and_shuffled_candidates():
    items = _ordering_candidates()
    ids = [f"first-run-{index}" for index in range(len(items))]
    baseline = reconcile_run(_run(items, ids))
    expected = json.dumps(baseline, ensure_ascii=False)

    permutations = [list(reversed(items))]
    for seed in range(5):
        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        permutations.append(shuffled)

    for run_index, shuffled in enumerate(permutations):
        shuffled_ids = [
            f"permuted-run-{run_index}-{index}" for index in range(len(shuffled))
        ]
        actual = reconcile_run(_run(shuffled, shuffled_ids))
        assert json.dumps(actual, ensure_ascii=False) == expected


def test_reconciliation_is_identical_for_three_runs_with_identical_candidates():
    items = _ordering_candidates()
    run = _run(items, [f"same-input-{index}" for index in range(len(items))])

    outputs = [reconcile_run(run) for _ in range(3)]

    assert outputs[0] == outputs[1] == outputs[2]


def _canonical_output(output):
    canonical = dict(output)
    for key in ("rates", "questions", "not_loaded"):
        canonical[key] = sorted(
            canonical[key], key=lambda value: json.dumps(value, sort_keys=True)
        )
    canonical["policies"] = {
        key: sorted(values, key=lambda value: json.dumps(value, sort_keys=True))
        if isinstance(values, list)
        else values
        for key, values in canonical["policies"].items()
    }
    return canonical


def _json_differences(expected, actual, path=""):
    differences = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() - actual.keys()):
            differences.append(
                {
                    "kind": "missing property",
                    "path": f"{path}.{key}" if path else key,
                    "expected": expected[key],
                    "actual": _MISSING,
                }
            )
        for key in sorted(actual.keys() - expected.keys()):
            differences.append(
                {
                    "kind": "additional property",
                    "path": f"{path}.{key}" if path else key,
                    "expected": _MISSING,
                    "actual": actual[key],
                }
            )
        for key in sorted(expected.keys() & actual.keys()):
            differences.extend(
                _json_differences(
                    expected[key],
                    actual[key],
                    f"{path}.{key}" if path else key,
                )
            )
    elif isinstance(expected, list) and isinstance(actual, list):
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            differences.extend(
                _json_differences(expected_item, actual_item, f"{path}[{index}]")
            )
        for index in range(len(actual), len(expected)):
            differences.append(
                {
                    "kind": "missing array item",
                    "path": f"{path}[{index}]",
                    "expected": expected[index],
                    "actual": _MISSING,
                }
            )
        for index in range(len(expected), len(actual)):
            differences.append(
                {
                    "kind": "additional array item",
                    "path": f"{path}[{index}]",
                    "expected": _MISSING,
                    "actual": actual[index],
                }
            )
    elif expected != actual or type(expected) is not type(actual):
        differences.append(
            {
                "kind": "unequal value",
                "path": path or "$",
                "expected": expected,
                "actual": actual,
            }
        )
    return differences


def _format_json_value(value):
    if value is _MISSING:
        return "<missing>"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _format_json_differences(differences):
    return "\n".join(
        f"- {difference['path']}: {difference['kind']} "
        f"(expected: {_format_json_value(difference['expected'])}; "
        f"actual: {_format_json_value(difference['actual'])})"
        for difference in differences
    )


def _live_failure_output_dir():
    if os.environ.get("SAVE_LIVE_FAILURE_OUTPUT") != "1":
        return None
    configured = os.environ.get("LIVE_FAILURE_OUTPUT_DIR")
    output_dir = Path(configured) if configured else ROOT / "artifacts" / "live-failures"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    expected_dir = EXPECTED.resolve()
    if output_dir == expected_dir or expected_dir in output_dir.parents:
        raise ValueError("Live failure outputs must be outside the expected fixtures directory.")
    return output_dir


def _write_live_failure_output(output, contract_id, output_dir):
    if not contract_id.isalnum():
        raise ValueError("Contract ID must contain only alphanumeric characters.")
    output_dir = Path(output_dir).resolve()
    expected_dir = EXPECTED.resolve()
    if output_dir == expected_dir or expected_dir in output_dir.parents:
        raise ValueError("Live failure outputs must be outside the expected fixtures directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    serialized = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    suffix = 0
    while True:
        suffix_text = f"-{suffix}" if suffix else ""
        output_path = output_dir / f"{contract_id}-{timestamp}{suffix_text}.json"
        try:
            with output_path.open("x", encoding="utf-8") as diagnostic_file:
                diagnostic_file.write(serialized)
            return output_path
        except FileExistsError:
            suffix += 1


def test_json_difference_report_identifies_nested_missing_extra_and_unequal_values():
    expected = {
        "policies": {
            "cancellation": [
                {
                    "charge_percent": 50,
                    "source": {
                        "file": "rate-letter.pdf",
                        "page": 2,
                        "quote": "Cancelled within 14 days: 50%.",
                    },
                    "within_days": 14,
                }
            ]
        }
    }
    actual = {
        "policies": {
            "cancellation": [
                {
                    "charge_percent": 100,
                    "source": {
                        "file": "other-letter.pdf",
                        "page": 2,
                        "document_id": "contract-a",
                    },
                    "within_days": 14,
                }
            ]
        }
    }

    differences = _json_differences(expected, actual)
    formatted = _format_json_differences(differences)

    assert {
        (item["path"], item["kind"]) for item in differences
    } == {
        ("policies.cancellation[0].charge_percent", "unequal value"),
        ("policies.cancellation[0].source.file", "unequal value"),
        ("policies.cancellation[0].source.quote", "missing property"),
        ("policies.cancellation[0].source.document_id", "additional property"),
    }
    assert "expected: 50; actual: 100" in formatted
    assert 'expected: "rate-letter.pdf"; actual: "other-letter.pdf"' in formatted
    assert (
        'source.quote: missing property (expected: "Cancelled within 14 days: 50%."; '
        "actual: <missing>)"
    ) in formatted
    assert (
        "source.document_id: additional property "
        '(expected: <missing>; actual: "contract-a")'
    ) in formatted


def test_live_failure_diagnostics_are_disabled_unless_opted_in(monkeypatch):
    monkeypatch.delenv("SAVE_LIVE_FAILURE_OUTPUT", raising=False)
    monkeypatch.setenv("LIVE_FAILURE_OUTPUT_DIR", "diagnostics")

    assert _live_failure_output_dir() is None


def test_live_failure_output_writer_saves_reconciled_json(tmp_path):
    output = {"policies": {"cancellation": [{"within_days": 14, "charge_percent": 50}]}}
    output_dir = tmp_path / "live-failures"

    saved_path = _write_live_failure_output(output, "c01", output_dir)

    assert saved_path.parent == output_dir
    assert saved_path.name.startswith("c01-")
    assert saved_path.suffix == ".json"
    assert json.loads(saved_path.read_text(encoding="utf-8")) == output


def _insufficient_quota_reason(error):
    if error.classification == "quota_exceeded":
        return "Gemini reports exhausted API quota."
    return None


def _report_live_reconciliation_error(error, elapsed):
    quota_reason = _insufficient_quota_reason(error)
    if quota_reason:
        message = (
            "Live reconciliation comparison blocked before extraction completed. "
            f"{quota_reason} No golden comparison was made."
        )
        print(message)
        pytest.skip(message)
    pytest.fail(
        "Live extraction failed before the golden comparison; this is an API or "
        f"extraction failure, not a reconciliation accuracy result. "
        f"Elapsed: {elapsed}s. Error: {error}",
        pytrace=False,
    )


def test_live_quota_exhaustion_is_blocked_not_an_accuracy_failure():
    error = ExtractionError(
        "Gemini structured extraction failed",
        status_code=429,
        classification="quota_exceeded",
    )

    with pytest.raises(pytest.skip.Exception, match="No golden comparison was made"):
        _report_live_reconciliation_error(error, elapsed=0.5)


def test_live_rate_limit_is_not_mislabeled_as_insufficient_credits():
    error = ExtractionError(
        "Gemini structured extraction failed",
        status_code=429,
        classification="rate_limit",
    )

    assert _insufficient_quota_reason(error) is None


CONTRACTS = ROOT / "contracts"


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_RECONCILIATION") != "1",
    reason="Set RUN_LIVE_RECONCILIATION=1 to run paid LLM fixture evaluations",
)
@pytest.mark.parametrize("contract_id", ["c01", "c02", "c03", "c04", "c05"])
def test_live_reconciliation_matches_golden_fixture(contract_id):
    started = perf_counter()
    try:
        run = extract_folder(CONTRACTS / contract_id, client=_create_client())
    except ExtractionError as exc:
        elapsed = round(perf_counter() - started, 3)
        _report_live_reconciliation_error(exc, elapsed)
    try:
        output = reconcile_run(run)
    except ReconciliationError as exc:
        pytest.fail(
            "Extraction completed, but reconciliation failed before the golden "
            f"comparison: {exc}",
            pytrace=False,
        )
    fixture = json.loads(
        (EXPECTED / f"{contract_id}.json").read_text(encoding="utf-8")
    )
    actual_canonical = _canonical_output(output)
    expected_canonical = _canonical_output(fixture)
    if actual_canonical != expected_canonical:
        differences = _json_differences(expected_canonical, actual_canonical)
        message = (
            f"Golden comparison failed for {contract_id}; "
            f"{len(differences)} differing JSON path(s):\n"
            f"{_format_json_differences(differences)}"
        )
        output_dir = _live_failure_output_dir()
        if output_dir is not None:
            saved_path = _write_live_failure_output(output, contract_id, output_dir)
            message += f"\nActual reconciled JSON saved to: {saved_path}"
        pytest.fail(message, pytrace=False)

