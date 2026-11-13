import re
from datetime import date
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
    ExtractionRun,
    EvidenceSource,
    ExtractionResult,
    RoomRateCandidate,
    SeasonCandidate,
    TokenUsage,
    _ocr_occupancy_check,
    _rate_type_check,
    _season_context_check,
    _text_unspecified_occupancy_check,
    _validate_sources,
)
from ingest import OCRWord, PageContent, PageLine, ingest_folder, tesseract_available


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"


def _page(text, lines=None, extraction_method="text"):
    return PageContent(
        file="evidence.pdf",
        page=1,
        text=text,
        extraction_method=extraction_method,
        ocr_confidence=95.0 if extraction_method == "ocr" else None,
        lines=lines or [PageLine(text=text)],
    )


def _assessment(candidate, page):
    result = ExtractionResult(
        contract_facts=[candidate] if isinstance(candidate, ContractFactCandidate) else [],
        seasons=[candidate] if isinstance(candidate, SeasonCandidate) else [],
        room_rates=[candidate] if isinstance(candidate, RoomRateCandidate) else [],
        cancellations=[candidate] if isinstance(candidate, CancellationCandidate) else [],
        advance_payments=(
            [candidate] if isinstance(candidate, AdvancePaymentCandidate) else []
        ),
        commercial_terms=(
            [candidate] if isinstance(candidate, CommercialTermCandidate) else []
        ),
    )
    return _validate_sources(result, [page])[0]


def test_hotel_name_with_supporting_quote_is_verified():
    quote = "Welcome to Hotel Cedar Bay."
    candidate = ContractFactCandidate(
        field="hotel_name",
        value="Hotel Cedar Bay",
        source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
    )

    assert _assessment(candidate, _page(quote)).status == "verified"


def test_valid_quote_does_not_verify_an_unsupported_hotel_name():
    quote = "Welcome to Hotel Cedar Bay."
    candidate = ContractFactCandidate(
        field="hotel_name",
        value="Hotel Mango Cove",
        source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
    )

    assert _assessment(candidate, _page(quote)).status == "invalid"


def test_season_with_unsupported_date_is_invalid():
    quote = "Peak season: 1 October 2026 to 15 January 2027"
    candidate = SeasonCandidate(
        name="Peak season",
        start_date=date(2026, 10, 2),
        end_date=date(2027, 1, 15),
        source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
    )

    assert _assessment(candidate, _page(quote)).status == "invalid"


def test_correct_price_does_not_verify_a_wrong_room_association():
    quote = "Garden Villa CP 7,000"
    candidate = RoomRateCandidate(
        room="Sea View Villa",
        meal_plan="CP",
        occupancy="unspecified",
        price=7000,
        rate_type="unspecified",
        source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
    )

    assert _assessment(candidate, _page(quote)).status == "invalid"


@pytest.mark.skipif(
    not tesseract_available(), reason="Tesseract is required for the scanned c02 page"
)
def test_c02_ocr_row_maps_price_to_the_single_occupancy_column():
    page = next(
        page for page in ingest_folder(CONTRACTS / "c02")
        if page.page == 2
    )
    row = next(
        line for line in page.lines
        if "garden" in line.text.lower()
        and "cp" in line.text.lower()
        and "7,000" in line.text
    )
    candidate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="single",
        price=7000,
        season="Peak season",
        start_date=date(2026, 10, 1),
        end_date=date(2027, 1, 15),
        rate_type="unspecified",
        source=EvidenceSource(file=page.file, page=page.page, quote=row.text),
    )

    assessment = _validate_sources(ExtractionResult(room_rates=[candidate]), [page])[0]

    assert assessment.status == "uncertain"
    assert not any("occupancy" in issue for issue in assessment.issues)
    assert any("Rate type is not explicit" in issue for issue in assessment.issues)


@pytest.mark.skipif(
    not tesseract_available(), reason="Tesseract is required for the scanned c02 page"
)
def test_c02_map_prices_are_independently_verified_from_targeted_ocr():
    pages = ingest_folder(CONTRACTS / "c02")
    page = next(item for item in pages if item.page == 2)
    row = next(
        line
        for line in page.lines
        if "garden" in line.text.casefold()
        and "map" in line.text.casefold()
        and "8,200" in line.text
        and "10,200" in line.text
    )
    double_price_word = next(word for word in row.words if word.text == "10,200")
    assert double_price_word.confidence is not None
    assert double_price_word.confidence >= 50
    assert double_price_word.width > 0
    assert double_price_word.height > 0

    candidates = [
        RoomRateCandidate(
            room="Garden Villa",
            meal_plan="MAP",
            occupancy=occupancy,
            price=price,
            season="Peak season",
            start_date=date(2026, 10, 1),
            end_date=date(2027, 1, 15),
            rate_type="individual",
            source=EvidenceSource(file=page.file, page=page.page, quote=row.text),
        )
        for occupancy, price in (("single", 8200), ("double", 10200))
    ]

    assessments = _validate_sources(ExtractionResult(room_rates=candidates), pages)

    assert [assessment.status for assessment in assessments] == [
        "verified",
        "verified",
    ]
    assert all(not assessment.issues for assessment in assessments)


@pytest.mark.parametrize(
    ("scope", "header", "expected"),
    [
        ("This schedule is priced per room per night.", "Room Only", True),
        ("Rates apply to both single and double occupancy.", "Room Only", True),
        ("", "Room Only", None),
        ("", "Single Double", False),
        ("Heritage Room per unit 3,200", "Room Only", None),
    ],
)
def test_unspecified_occupancy_requires_affirmative_scope(scope, header, expected):
    page_text = "\n".join(
        line
        for line in (
            "Contracted rates",
            scope,
            "Room Category",
            header,
            "Heritage Room CP 3,200",
        )
        if line
    )
    lines = [PageLine(text=line) for line in page_text.splitlines()]
    page = _page(page_text, lines=lines)
    row_index = len(lines) - 1

    assert _text_unspecified_occupancy_check(
        page, row_index, lines[row_index]
    ) is expected


def test_positive_contracted_schedule_context_verifies_unspecified_rate_type():
    quote = "Garden Villa CP 7,000 7,800"
    context_page = _page(
        "Rates in this letter are for individual bookings. "
        "Rate tables are on the next page."
    )
    table_page = PageContent(
        file=context_page.file,
        page=2,
        text=quote,
        extraction_method="text",
        ocr_confidence=None,
        lines=[PageLine(text=quote)],
    )
    candidate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="single",
        price=7000,
        rate_type="unspecified",
        source=EvidenceSource(
            file=table_page.file, page=table_page.page, quote=quote
        ),
    )

    assert _rate_type_check(
        candidate, table_page, 0, table_page.lines[0], [context_page, table_page]
    ) is True


def test_actual_c02_page_one_context_applies_to_its_page_two_table():
    pages = ingest_folder(CONTRACTS / "c02")
    context_page = next(page for page in pages if page.page == 1)
    table_page = next(page for page in pages if page.page == 2)
    row_index, row = next(
        (index, line)
        for index, line in enumerate(table_page.lines)
        if "gardenvilla" in re.sub(r"[^a-z0-9]", "", line.text.casefold())
        and "cp" in re.sub(r"[^a-z0-9]", "", line.text.casefold())
    )
    prices = re.findall(r"(?<!\d)\d[\d,]*", row.text)
    candidate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="single",
        price=float(prices[0].replace(",", "")),
        rate_type="individual",
        source=EvidenceSource(
            file=table_page.file,
            page=table_page.page,
            quote=row.text,
        ),
    )

    assert _rate_type_check(candidate, table_page, row_index, row, pages) is True


def test_contracted_rates_heading_supports_standard_schedule_type():
    table = _page(
        "Contracted rates 2026-27\nRoom Category\nGarden Villa CP 7,000",
        lines=[
            PageLine(text="Contracted rates 2026-27"),
            PageLine(text="Room Category"),
            PageLine(text="Garden Villa CP 7,000"),
        ],
    )
    candidate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="single",
        price=7000,
        rate_type="unspecified",
        source=EvidenceSource(
            file=table.file, page=table.page, quote=table.text
        ),
    )

    assert _rate_type_check(candidate, table, 2, table.lines[2], [table]) is True


def test_rate_type_context_keeps_group_heading_local_to_its_table():
    lines = [
        PageLine(text="Rates for individual bookings are listed below."),
        PageLine(text="Room Category"),
        PageLine(text="Garden Villa CP 7,000"),
        PageLine(text=""),
        PageLine(text="Special group rates"),
        PageLine(text="Room Category"),
        PageLine(text="Garden Villa CP 6,500"),
    ]
    page = _page("\n".join(line.text for line in lines), lines=lines)
    candidate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="double",
        price=6500,
        rate_type="group",
        source=EvidenceSource(file=page.file, page=1, quote=lines[6].text),
    )

    assert _rate_type_check(candidate, page, 6, lines[6], [page]) is True
    assert _rate_type_check(
        candidate.model_copy(update={"rate_type": "individual"}),
        page,
        6,
        lines[6],
        [page],
    ) is False


def test_rack_schedule_on_same_page_does_not_inherit_contracted_scope():
    lines = [
        PageLine(text="Contracted rates"),
        PageLine(text="Room Category"),
        PageLine(text="Garden Villa CP 7,000"),
        PageLine(text="Rack rates"),
        PageLine(text="Room Category"),
        PageLine(text="Garden Villa CP 9,000"),
    ]
    page = _page("\n".join(line.text for line in lines), lines=lines)
    candidate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="double",
        price=9000,
        rate_type="rack",
        source=EvidenceSource(file=page.file, page=1, quote=lines[5].text),
    )

    assert _rate_type_check(candidate, page, 5, lines[5], [page]) is True
    assert _rate_type_check(
        candidate.model_copy(update={"rate_type": "individual"}),
        page,
        5,
        lines[5],
        [page],
    ) is False


def test_unrelated_individual_statement_does_not_classify_a_later_ambiguous_table():
    lines = [
        PageLine(text="Rates for individual bookings are available."),
        PageLine(text="Seasonal schedule"),
        PageLine(text="Room Category"),
        PageLine(text="Garden Villa CP 7,000"),
    ]
    page = _page("\n".join(line.text for line in lines), lines=lines)
    candidate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="double",
        price=7000,
        rate_type="individual",
        source=EvidenceSource(file=page.file, page=1, quote=lines[3].text),
    )

    assert _rate_type_check(candidate, page, 3, lines[3], [page]) is None


def test_multiple_ambiguous_rate_tables_remain_unclassified():
    lines = [
        PageLine(text="Rate schedule A"),
        PageLine(text="Room Category"),
        PageLine(text="Garden Villa CP 7,000"),
        PageLine(text="Rate schedule B"),
        PageLine(text="Room Category"),
        PageLine(text="Garden Villa CP 8,000"),
    ]
    page = _page("\n".join(line.text for line in lines), lines=lines)
    candidate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="double",
        price=8000,
        rate_type="individual",
        source=EvidenceSource(file=page.file, page=1, quote=lines[5].text),
    )

    assert _rate_type_check(candidate, page, 5, lines[5], [page]) is None


def test_identically_named_files_do_not_share_cross_page_rate_context():
    context = PageContent(
        file="rate-letter.pdf",
        page=1,
        text=(
            "Rates in this letter are for individual bookings. "
            "The rate tables are on the next page."
        ),
        extraction_method="text",
        ocr_confidence=None,
        lines=[PageLine(text="Rates in this letter are for individual bookings. "
                             "The rate tables are on the next page.")],
        document_id="contract-a/rate-letter.pdf",
    )
    table = PageContent(
        file="rate-letter.pdf",
        page=2,
        text="Garden Villa CP 7,000",
        extraction_method="text",
        ocr_confidence=None,
        lines=[PageLine(text="Garden Villa CP 7,000")],
        document_id="contract-b/rate-letter.pdf",
    )
    candidate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="double",
        price=7000,
        rate_type="individual",
        source=EvidenceSource(file=table.file, page=2, quote=table.text),
    )

    assert _rate_type_check(candidate, table, 0, table.lines[0], [context, table]) is None


def test_unrelated_per_room_wording_does_not_apply_to_a_later_rate_table():
    lines = [
        PageLine(text="Contracted rates"),
        PageLine(text="The conference room is priced per room per night."),
        PageLine(text="Seasonal rates"),
        PageLine(text="Room Category"),
        PageLine(text="Room Only"),
        PageLine(text="Garden Villa CP 7,000"),
    ]
    page = _page("\n".join(line.text for line in lines), lines=lines)

    assert _text_unspecified_occupancy_check(page, 5, lines[5]) is None


def test_per_room_wording_does_not_leak_into_an_unlabelled_second_table():
    lines = [
        PageLine(text="Contracted rates"),
        PageLine(text="This schedule is priced per room per night."),
        PageLine(text="Room Category"),
        PageLine(text="Room Only"),
        PageLine(text="Garden Villa CP 7,000"),
        PageLine(text="Winter schedule"),
        PageLine(text="Room Category"),
        PageLine(text="Room Only"),
        PageLine(text="Garden Villa CP 8,000"),
    ]
    page = _page("\n".join(line.text for line in lines), lines=lines)

    assert _text_unspecified_occupancy_check(page, 4, lines[4]) is True
    assert _text_unspecified_occupancy_check(page, 8, lines[8]) is None


def test_model_applicability_marker_is_not_a_substitute_for_source_support():
    quote = "Garden Villa CP 7,000"
    page = _page(
        "Contracted rates\nRoom Category\nRoom Only\n" + quote,
        lines=[
            PageLine(text="Contracted rates"),
            PageLine(text="Room Category"),
            PageLine(text="Room Only"),
            PageLine(text=quote),
        ],
    )
    candidate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="unspecified",
        occupancy_applicability="single_and_double",
        price=7000,
        rate_type="individual",
        source=EvidenceSource(file=page.file, page=page.page, quote=quote),
    )

    assessment = _validate_sources(ExtractionResult(room_rates=[candidate]), [page])[0]

    assert assessment.status == "uncertain"


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("Special group rate of INR 6,500 on MAP", False),
        ("Published rack rates on CP", False),
        ("Garden Villa CP 7,000", None),
    ],
)
def test_rate_type_needs_positive_individual_context_or_explicit_exclusion(
    quote, expected
):
    page = _page(quote)
    candidate = RoomRateCandidate(
        room=None if quote.startswith("Special") or quote.startswith("Published") else "Garden Villa",
        meal_plan="MAP" if "MAP" in quote else "CP",
        occupancy="single",
        price=6500 if "6,500" in quote else 7000,
        rate_type="unspecified",
        source=EvidenceSource(file=page.file, page=page.page, quote=quote),
    )

    assert _rate_type_check(candidate, page, 0, page.lines[0], [page]) is expected


def test_occupancy_is_uncertain_when_price_lies_on_a_column_boundary():
    quote = "Garden Villa CP 7,000"
    header = PageLine(
        text="Single Double",
        words=[
            OCRWord("Single", 90, 10, 20, 10, 90),
            OCRWord("Double", 190, 10, 20, 10, 90),
        ],
    )
    row = PageLine(
        text=quote,
        words=[
            OCRWord("Garden", 10, 30, 30, 10, 90),
            OCRWord("Villa", 42, 30, 25, 10, 90),
            OCRWord("CP", 70, 30, 10, 10, 90),
            OCRWord("7,000", 145, 30, 10, 10, 90),
        ],
    )
    page = _page(
        f"{header.text}\n{row.text}",
        lines=[header, row],
        extraction_method="ocr",
    )
    candidate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="single",
        price=7000,
        rate_type="unspecified",
        source=EvidenceSource(file=page.file, page=page.page, quote=quote),
    )

    assessment = _assessment(candidate, page)

    assert assessment.status == "uncertain"
    assert any("boundary" in issue for issue in assessment.issues)


def test_cancellation_percentage_must_be_supported_by_its_quote():
    quote = "Cancellation charge: 100% within 14 days of arrival."
    candidate = CancellationCandidate(
        charge_percent=50,
        days_max=14,
        source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
    )

    assert _assessment(candidate, _page(quote)).status == "invalid"


def test_conflicting_cancellation_candidates_are_both_preserved():
    first_quote = "Cancellation charge: 100% within 21 days of arrival."
    second_quote = "Cancellation charge: 50% within 30 days of arrival."
    page = _page(f"{first_quote}\n{second_quote}")
    candidates = [
        CancellationCandidate(
            charge_percent=100,
            days_max=21,
            source=EvidenceSource(file=page.file, page=1, quote=first_quote),
        ),
        CancellationCandidate(
            charge_percent=50,
            days_max=30,
            source=EvidenceSource(file=page.file, page=1, quote=second_quote),
        ),
    ]

    assessments = _validate_sources(ExtractionResult(cancellations=candidates), [page])

    assert len(candidates) == 2
    assert [assessment.status for assessment in assessments] == ["verified", "verified"]


def test_ocr_capitalization_and_punctuation_variations_are_accepted():
    quote = "Contracting hotel: HOTEL Saffron-Courtyard!"
    candidate = ContractFactCandidate(
        field="hotel_name",
        value="Hotel Saffron Courtyard",
        source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
    )

    assert _assessment(candidate, _page(quote)).status == "verified"


def test_currency_and_tax_status_are_verified_from_explicit_wording():
    quote = "Rates are inclusive of all applicable taxes and quoted in INR."
    candidates = [
        ContractFactCandidate(
            field="tax_included",
            value=True,
            source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
        ),
        ContractFactCandidate(
            field="currency",
            value="INR",
            source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
        ),
    ]

    assessments = _validate_sources(
        ExtractionResult(contract_facts=candidates), [_page(quote)]
    )

    assert [assessment.status for assessment in assessments] == ["verified", "verified"]


def test_advance_payment_and_commercial_term_values_are_supported():
    advance_quote = (
        "Advance payment of 30% is due within 7 days; "
        "balance is payable before arrival."
    )
    commercial_quote = "Festive surcharge: INR 2,500 per room on New Year."
    candidates = ExtractionResult(
        advance_payments=[
            AdvancePaymentCandidate(
                percentage=30,
                timing="within 7 days",
                balance_wording="balance is payable before arrival",
                source=EvidenceSource(
                    file="evidence.pdf", page=1, quote=advance_quote
                ),
            )
        ],
        commercial_terms=[
            CommercialTermCandidate(
                kind="festive_surcharge",
                description="Festive surcharge per room on New Year",
                amount=2500,
                currency="INR",
                source=EvidenceSource(
                    file="evidence.pdf", page=1, quote=commercial_quote
                ),
            )
        ],
    )
    page = _page(f"{advance_quote}\n{commercial_quote}")

    assessments = _validate_sources(candidates, [page])

    assert [assessment.status for assessment in assessments] == [
        "verified",
        "verified",
    ]


def test_missing_source_evidence_is_retained_and_invalid():
    candidate = ContractFactCandidate(
        field="hotel_name",
        value="Hotel Cedar Bay",
        source=EvidenceSource(),
    )

    assessment = _assessment(candidate, _page("Hotel Cedar Bay"))

    assert assessment.status == "invalid"
    assert "Supporting quote is missing." in assessment.issues


def test_negated_hotel_name_is_not_verified_as_the_contracting_hotel():
    quote = "Our agency does not represent Hotel Cedar Bay."
    candidate = ContractFactCandidate(
        field="hotel_name",
        value="Hotel Cedar Bay",
        source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
    )

    assert _assessment(candidate, _page(quote)).status == "invalid"


def test_negated_tax_wording_does_not_verify_tax_included():
    quote = "Taxes aren't included in the quoted rates."
    candidate = ContractFactCandidate(
        field="tax_included",
        value=True,
        source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
    )

    assert _assessment(candidate, _page(quote)).status == "invalid"


def test_swapped_validity_start_and_end_dates_are_invalid():
    quote = "The agreement is valid from 1 October 2026 to 30 September 2027."
    candidates = [
        ContractFactCandidate(
            field="validity_start",
            value="2027-09-30",
            source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
        ),
        ContractFactCandidate(
            field="validity_end",
            value="2026-10-01",
            source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
        ),
    ]

    assessments = _validate_sources(
        ExtractionResult(contract_facts=candidates), [_page(quote)]
    )

    assert [assessment.status for assessment in assessments] == ["invalid", "invalid"]


def test_cancellation_percentage_must_be_in_a_cancellation_clause():
    quote = "Cancellation penalty is 100%; advance payment is 20%."
    candidate = CancellationCandidate(
        charge_percent=20,
        days_max=None,
        source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
    )

    assert _assessment(candidate, _page(quote)).status == "invalid"


def test_negated_cancellation_percentage_is_not_verified():
    quote = "No cancellation penalty of 20% is charged."
    candidate = CancellationCandidate(
        charge_percent=20,
        source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
    )

    assert _assessment(candidate, _page(quote)).status == "invalid"


def test_yen_symbol_alone_does_not_verify_jpy():
    quote = "Rates are quoted in ¥."
    candidate = ContractFactCandidate(
        field="currency",
        value="JPY",
        source=EvidenceSource(file="evidence.pdf", page=1, quote=quote),
    )

    assert _assessment(candidate, _page(quote)).status == "uncertain"


def test_side_by_side_season_headings_use_price_column_geometry():
    peak_text = "Peak season 1 October 2026 to 15 January 2027"
    lean_text = "Lean season 16 January 2027 to 30 September 2027"
    peak = PageLine(
        text=peak_text,
        words=[
            OCRWord("Peak", 10, 10, 35, 10, 95),
            OCRWord("season", 48, 10, 45, 10, 95),
            OCRWord("1", 98, 10, 5, 10, 95),
            OCRWord("October", 105, 10, 45, 10, 95),
            OCRWord("2026", 153, 10, 25, 10, 95),
            OCRWord("to", 182, 10, 10, 10, 95),
            OCRWord("15", 195, 10, 12, 10, 95),
            OCRWord("January", 210, 10, 45, 10, 95),
            OCRWord("2027", 258, 10, 25, 10, 95),
        ],
    )
    lean = PageLine(
        text=lean_text,
        words=[
            OCRWord("Lean", 320, 10, 30, 10, 95),
            OCRWord("season", 353, 10, 45, 10, 95),
            OCRWord("16", 402, 10, 12, 10, 95),
            OCRWord("January", 418, 10, 45, 10, 95),
            OCRWord("2027", 466, 10, 25, 10, 95),
            OCRWord("to", 495, 10, 10, 10, 95),
            OCRWord("30", 508, 10, 12, 10, 95),
            OCRWord("September", 524, 10, 60, 10, 95),
            OCRWord("2027", 587, 10, 25, 10, 95),
        ],
    )
    row_text = "Garden Villa CP 7,000"
    row = PageLine(
        text=row_text,
        words=[
            OCRWord("Garden", 20, 35, 40, 10, 95),
            OCRWord("Villa", 63, 35, 25, 10, 95),
            OCRWord("CP", 92, 35, 15, 10, 95),
            OCRWord("7,000", 120, 35, 35, 10, 95),
        ],
    )
    page = _page(
        f"{peak_text}\n{lean_text}\n{row_text}",
        lines=[peak, lean, row],
        extraction_method="ocr",
    )

    assert _season_context_check(
        page, 2, "Peak season", date(2026, 10, 1), date(2027, 1, 15), 7000
    ) is True
    assert _season_context_check(
        page, 2, "Lean season", date(2027, 1, 16), date(2027, 9, 30), 7000
    ) is False


def test_occupancy_price_in_the_wrong_column_is_not_verified():
    header = PageLine(
        text="Single Double",
        words=[
            OCRWord("Single", 10, 10, 40, 10, 95),
            OCRWord("Double", 200, 10, 45, 10, 95),
        ],
    )
    row = PageLine(
        text="Garden Villa CP 7,000",
        words=[
            OCRWord("Garden", 10, 30, 35, 10, 95),
            OCRWord("Villa", 48, 30, 25, 10, 95),
            OCRWord("CP", 76, 30, 12, 10, 95),
            OCRWord("7,000", 190, 30, 35, 10, 95),
        ],
    )
    page = _page(
        "Single Double\nGarden Villa CP 7,000",
        lines=[header, row],
        extraction_method="ocr",
    )
    candidate = RoomRateCandidate(
        room="Garden Villa",
        meal_plan="CP",
        occupancy="single",
        price=7000,
        rate_type="unspecified",
        source=EvidenceSource(
            file="evidence.pdf", page=1, quote=row.text
        ),
    )

    assessment = _assessment(candidate, page)

    assert assessment.status == "invalid"
    assert any("different occupancy" in issue for issue in assessment.issues)


@pytest.mark.parametrize(
    "candidate",
    [
        ContractFactCandidate(
            field="hotel_name",
            value="Hotel Cedar Bay",
            source=EvidenceSource(file="evidence.pdf", page=1, quote="Hotel Cedar Bay."),
        ),
        CancellationCandidate(
            charge_percent=100,
            days_max=14,
            source=EvidenceSource(
                file="evidence.pdf",
                page=1,
                quote="Cancellation charge is 100% within 14 days of arrival.",
            ),
        ),
        CommercialTermCandidate(
            kind="festive_surcharge",
            description="Festive surcharge per room on New Year",
            amount=2500,
            currency="INR",
            source=EvidenceSource(
                file="evidence.pdf",
                page=1,
                quote="Festive surcharge: INR 2,500 per room on New Year.",
            ),
        ),
    ],
)
def test_low_confidence_non_rate_ocr_evidence_is_uncertain(candidate):
    line = PageLine(
        text=candidate.source.quote,
        words=[
            OCRWord(word, index * 20, 10, 15, 10, 20)
            for index, word in enumerate(candidate.source.quote.split())
        ],
    )
    page = _page(
        candidate.source.quote,
        lines=[line],
        extraction_method="ocr",
    )

    assert _assessment(candidate, page).status == "uncertain"


def _make_run(records, assessments):
    return ExtractionRun(
        model="test-model",
        candidate_records=tuple(records),
        evidence_assessments=tuple(assessments),
        usage=TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0,
        ),
    )


def test_missing_assessment_is_rejected():
    record = CandidateRecord(
        candidate_id="candidate-a",
        category="contract_fact",
        candidate=ContractFactCandidate(
            field="hotel_name",
            value="Hotel Cedar Bay",
            source=EvidenceSource(file="evidence.pdf", page=1, quote="Hotel Cedar Bay"),
        ),
    )

    with pytest.raises(ValidationError, match="Assessment references are incomplete"):
        _make_run([record], [])


def test_duplicate_assessment_is_rejected():
    record = CandidateRecord(
        candidate_id="candidate-a",
        category="contract_fact",
        candidate=ContractFactCandidate(
            field="hotel_name",
            value="Hotel Cedar Bay",
            source=EvidenceSource(file="evidence.pdf", page=1, quote="Hotel Cedar Bay"),
        ),
    )
    assessment = CandidateEvidenceAssessment(
        candidate_id="candidate-a",
        category="contract_fact",
        status="verified",
    )

    with pytest.raises(ValidationError, match="Duplicate evidence assessments"):
        _make_run([record], [assessment, assessment])


def test_mismatched_assessment_category_is_rejected():
    record = CandidateRecord(
        candidate_id="candidate-a",
        category="contract_fact",
        candidate=ContractFactCandidate(
            field="hotel_name",
            value="Hotel Cedar Bay",
            source=EvidenceSource(file="evidence.pdf", page=1, quote="Hotel Cedar Bay"),
        ),
    )
    assessment = CandidateEvidenceAssessment(
        candidate_id="candidate-a",
        category="room_rate",
        status="verified",
    )

    with pytest.raises(ValidationError, match="category does not match"):
        _make_run([record], [assessment])


def test_duplicate_candidate_ids_are_rejected():
    record = CandidateRecord(
        candidate_id="same-id",
        category="contract_fact",
        candidate=ContractFactCandidate(
            field="hotel_name",
            value="Hotel Cedar Bay",
            source=EvidenceSource(file="evidence.pdf", page=1, quote="Hotel Cedar Bay"),
        ),
    )
    assessment = CandidateEvidenceAssessment(
        candidate_id="same-id",
        category="contract_fact",
        status="verified",
    )

    with pytest.raises(ValidationError, match="Duplicate candidate IDs"):
        _make_run([record, record], [assessment])


def test_reordering_candidates_does_not_change_assessment_identity():
    verified_record = CandidateRecord(
        candidate_id="verified-id",
        category="contract_fact",
        candidate=ContractFactCandidate(
            field="hotel_name",
            value="Hotel Cedar Bay",
            source=EvidenceSource(file="evidence.pdf", page=1, quote="Hotel Cedar Bay"),
        ),
    )
    uncertain_record = CandidateRecord(
        candidate_id="uncertain-id",
        category="contract_fact",
        candidate=ContractFactCandidate(
            field="hotel_name",
            value="Hotel Cedar Bay",
            source=EvidenceSource(file="evidence.pdf", page=1, quote="Hotel Cedar Bay"),
        ),
    )
    run = _make_run(
        [uncertain_record, verified_record],
        [
            CandidateEvidenceAssessment(
                candidate_id="verified-id",
                category="contract_fact",
                status="verified",
            ),
            CandidateEvidenceAssessment(
                candidate_id="uncertain-id",
                category="contract_fact",
                status="uncertain",
            ),
        ],
    )

    eligible = run.verified_candidates()

    assert [record.candidate_id for record in eligible] == ["verified-id"]
    assert eligible[0].candidate.value == "Hotel Cedar Bay"
