"""Deterministic conversion of verified evidence candidates to output JSON."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from jsonschema import (
    Draft202012Validator,
    FormatChecker,
    SchemaError as JsonSchemaDefinitionError,
    ValidationError as JsonSchemaValidationError,
)

from extract import (
    AdvancePaymentCandidate,
    CancellationCandidate,
    CandidateRecord,
    CommercialTermCandidate,
    ContractFactCandidate,
    ExtractionRun,
    RoomRateCandidate,
    SeasonCandidate,
    _quote_contains_price,
)


_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "output.schema.json"
_MEAL_PLAN_ALIASES = {
    "ep": "EP",
    "room only": "EP",
    "cp": "CP",
    "breakfast": "CP",
    "bb": "CP",
    "with breakfast": "CP",
    "bed and breakfast": "CP",
    "map": "MAP",
    "breakfast and dinner": "MAP",
    "breakfast dinner": "MAP",
    "half board": "MAP",
    "ap": "AP",
    "full board": "AP",
}


class ReconciliationError(ValueError):
    """Verified evidence could not be reconciled into schema-valid output."""


@dataclass(frozen=True)
class _Period:
    name: Optional[str]
    start: date
    end: date
    source: dict[str, Any]

    @property
    def key(self) -> tuple[str, date, date]:
        return (_normalise(self.name or ""), self.start, self.end)


@dataclass
class _PeriodAnalysis:
    periods: dict[tuple[str, date, date], _Period] = field(default_factory=dict)
    affected: set[tuple[str, date, date]] = field(default_factory=set)
    issues: list[str] = field(default_factory=list)


def reconcile_run(run: ExtractionRun) -> dict[str, Any]:
    """Reconcile explicitly verified candidates and validate the final schema."""
    records = run.verified_candidates()
    result: dict[str, Any] = {
        "contract": {},
        "rates": [],
        "policies": {},
        "questions": [],
        "not_loaded": [],
    }

    questions: list[dict[str, Any]] = result["questions"]
    facts = [
        record.candidate
        for record in records
        if isinstance(record.candidate, ContractFactCandidate)
    ]
    _reconcile_contract_facts(facts, result["contract"], questions)

    seasons = [
        record.candidate
        for record in records
        if isinstance(record.candidate, SeasonCandidate)
    ]
    rate_records = [
        record
        for record in records
        if isinstance(record.candidate, RoomRateCandidate)
    ]
    period_analysis = _analyse_periods(
        seasons,
        [record.candidate for record in rate_records],
        result["contract"],
    )
    if period_analysis.issues:
        sources = [
            period.source
            for key, period in period_analysis.periods.items()
            if key in period_analysis.affected
        ]
        if not sources:
            sources = [period.source for period in period_analysis.periods.values()]
        options = _period_options(period_analysis)
        _append_question(
            questions,
            _question(
                "rates",
                "Which rate-period dates should apply?",
                " ".join(dict.fromkeys(period_analysis.issues)),
                options,
                sources,
            ),
        )

    _reconcile_rates(
        rate_records,
        period_analysis,
        result["contract"],
        result["rates"],
        questions,
        result["not_loaded"],
    )

    commercial_terms = [
        record.candidate
        for record in records
        if isinstance(record.candidate, CommercialTermCandidate)
    ]
    _collect_not_loaded_terms(commercial_terms, result["not_loaded"])

    cancellations = [
        record.candidate
        for record in records
        if isinstance(record.candidate, CancellationCandidate)
    ]
    cancellation_policy = _reconcile_cancellations(cancellations, questions)
    if cancellation_policy:
        result["policies"]["cancellation"] = cancellation_policy

    advance_payments = [
        record
        for record in records
        if isinstance(record.candidate, AdvancePaymentCandidate)
    ]
    advance_policy = _reconcile_advance_payments(advance_payments, questions)
    if advance_policy:
        result["policies"]["advance_payment"] = advance_policy

    _collect_uncertain_questions(run, result["contract"], questions)
    result["rates"].sort(
        key=lambda item: (
            item["start"],
            item["end"],
            _normalise(item["room"]),
            item["meal_plan"],
            item["occupancy"],
            item["price"],
            _source_sort_key(item["source"]),
        )
    )
    _deduplicate_not_loaded(result["not_loaded"])
    result["questions"].sort(
        key=lambda item: (
            item["about"],
            _normalise(item["question"]),
            _normalise(item["why"]),
            json.dumps(item, sort_keys=True, ensure_ascii=False),
        )
    )
    result["contract"] = dict(sorted(result["contract"].items()))
    validate_output(result)
    return result


def _collect_uncertain_questions(
    run: ExtractionRun,
    contract: dict[str, Any],
    questions: list[dict[str, Any]],
) -> None:
    tax_issue = "Quote does not clearly state whether taxes are included."
    occupancy_issues = {
        "Unspecified OCR occupancy is not supported by this rate schedule.",
        "Occupancy is not explicitly supported by the source text.",
    }
    for record, assessment in run.assessed_candidates("uncertain"):
        candidate = record.candidate
        source = candidate.source.model_dump(mode="json")
        if isinstance(candidate, ContractFactCandidate):
            quote_has_tax = bool(
                re.search(r"\b(?:tax(?:es)?|gst|vat)\b", candidate.source.quote, re.I)
            )
            if (
                candidate.field == "tax_included"
                and tax_issue in assessment.issues
                and quote_has_tax
            ):
                contract.pop("taxes_included", None)
                existing_tax_questions = [
                    question
                    for question in questions
                    if question["about"] == "taxes_included"
                ]
                if existing_tax_questions:
                    for question in existing_tax_questions:
                        question["sources"] = _unique_sources(
                            question["sources"] + [source]
                        )
                else:
                    _append_question(
                        questions,
                        _question(
                            "taxes_included",
                            "Do the rates include tax?",
                            "The cited tax wording does not establish whether tax is included or extra.",
                            [
                                {
                                    "label": "Tax included",
                                    "loads": "Set taxes_included to true only after confirmation.",
                                },
                                {
                                    "label": "GST extra",
                                    "loads": "Set taxes_included to false only after confirmation.",
                                },
                            ],
                            [source],
                        ),
                    )
        elif isinstance(candidate, RoomRateCandidate):
            if not any(issue in occupancy_issues for issue in assessment.issues):
                continue
            quote = candidate.source.quote
            normalized_quote = _normalise(quote)
            normalized_room = _normalise(candidate.room or "")
            if (
                not candidate.room
                or not normalized_room
                or normalized_room not in normalized_quote
                or _normalise_meal_plan(candidate.meal_plan)
                != _meal_plan_in_text(quote)
                or not _quote_contains_price(quote, candidate.price)
                or candidate.rate_type in {"group", "rack"}
            ):
                continue
            occupancy_labels = {
                label.casefold()
                for label in re.findall(r"\b(?:single|double)\b", quote, re.I)
            }
            if len(occupancy_labels) == 1:
                continue
            _append_question(
                questions,
                _question(
                    "rates",
                    "Does this room rate apply to single or double occupancy?",
                    "The cited room, meal-plan, and price row does not establish its occupancy scope.",
                    [
                        {
                            "label": "Single occupancy",
                            "loads": "Load the cited rate for single occupancy only after confirmation.",
                        },
                        {
                            "label": "Double occupancy",
                            "loads": "Load the cited rate for double occupancy only after confirmation.",
                        },
                    ],
                    [source],
                ),
            )


def validate_output(output: dict[str, Any]) -> None:
    """Validate an output object against the unchanged company JSON schema."""
    try:
        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator(
            schema, format_checker=FormatChecker()
        ).validate(output)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconciliationError(f"Unable to load the output schema: {exc}") from exc
    except JsonSchemaValidationError as exc:
        raise ReconciliationError(
            f"Reconciled output failed schema validation: {exc.message}"
        ) from exc
    except JsonSchemaDefinitionError as exc:
        raise ReconciliationError(f"The output schema is invalid: {exc.message}") from exc


def _reconcile_contract_facts(
    facts: Sequence[ContractFactCandidate],
    contract: dict[str, Any],
    questions: list[dict[str, Any]],
) -> None:
    output_names = {
        "hotel_name": "hotel",
        "validity_start": "valid_from",
        "validity_end": "valid_to",
        "currency": "currency",
        "tax_included": "taxes_included",
    }
    by_field: dict[str, list[ContractFactCandidate]] = defaultdict(list)
    for candidate in facts:
        by_field[candidate.field].append(candidate)

    for field_name in sorted(by_field):
        candidates = sorted(
            by_field[field_name],
            key=lambda candidate: (
                _fact_value_key(candidate),
                _source_sort_key(candidate.source.model_dump(mode="json")),
                _candidate_payload_sort_key(candidate),
            ),
        )
        distinct: dict[str, ContractFactCandidate] = {}
        for candidate in candidates:
            normalized = _fact_value_key(candidate)
            distinct.setdefault(normalized, candidate)
        if len(distinct) > 1:
            sources = [item.source.model_dump(mode="json") for item in candidates]
            options = [
                _fact_option(field_name, candidate.value)
                for _, candidate in sorted(distinct.items())[:4]
            ]
            _append_question(
                questions,
                _question(
                    output_names[field_name],
                    _fact_question(field_name),
                    f"Verified source statements disagree about {field_name.replace('_', ' ')}.",
                    options,
                    sources,
                ),
            )
            continue

        chosen = min(candidates, key=_candidate_sort_key)
        value = _contract_value(field_name, chosen.value)
        if value is None:
            continue
        key = output_names[field_name]
        source = chosen.source.model_dump(mode="json")
        if field_name in {"validity_start", "validity_end"}:
            contract[key] = {"value": value, "source": source}
        elif field_name == "tax_included":
            contract[key] = {"value": value, "source": source}
        else:
            contract[key] = {"value": value, "source": source}

    start = _contract_date(contract.get("valid_from"))
    end = _contract_date(contract.get("valid_to"))
    if start and end and start > end:
        sources = [
            item["source"]
            for name in ("validity_start", "validity_end")
            for item in [contract.get(output_names[name])]
            if item is not None
        ]
        contract.pop("valid_from", None)
        contract.pop("valid_to", None)
        for field_name in ("validity_start", "validity_end"):
            _append_question(
                questions,
                _question(
                    output_names[field_name],
                    "Which contract validity dates are correct?",
                    "The verified start date is later than the verified end date.",
                    [
                        {
                            "label": "Use the start date as written",
                            "loads": "Use the cited start date after confirming the corresponding end date.",
                        },
                        {
                            "label": "Use the end date as written",
                            "loads": "Use the cited end date after confirming the corresponding start date.",
                        },
                    ],
                    sources,
                ),
            )


def _fact_value_key(candidate: ContractFactCandidate) -> str:
    if candidate.field in {"validity_start", "validity_end"}:
        try:
            return date.fromisoformat(str(candidate.value)).isoformat()
        except ValueError:
            return str(candidate.value)
    if candidate.field == "currency":
        return str(candidate.value).strip().upper()
    return str(candidate.value).strip().casefold()


def _contract_value(field_name: str, value: Any) -> Optional[Any]:
    if field_name == "tax_included":
        return value if isinstance(value, bool) else None
    if field_name in {"validity_start", "validity_end"}:
        try:
            return date.fromisoformat(str(value)).isoformat()
        except ValueError:
            return None
    text = str(value).strip()
    return text.upper() if field_name == "currency" else (text or None)


def _fact_question(field_name: str) -> str:
    if field_name == "tax_included":
        return "Do the rates include tax?"
    return f"Which {field_name.replace('_', ' ')} should be used?"


def _fact_option(field_name: str, value: Any) -> dict[str, str]:
    if field_name == "tax_included":
        included = bool(value)
        return {
            "label": "Tax included" if included else "GST extra",
            "loads": f"taxes_included = {str(included).lower()}",
        }
    return {"label": str(value), "loads": str(value)}


def _analyse_periods(
    seasons: Sequence[SeasonCandidate],
    rates: Sequence[RoomRateCandidate],
    contract: dict[str, Any],
) -> _PeriodAnalysis:
    analysis = _PeriodAnalysis()

    def add_period(name: Optional[str], start: Optional[date], end: Optional[date], source: Any) -> None:
        if start is None or end is None or start > end:
            return
        period = _Period(name, start, end, source.model_dump(mode="json"))
        existing = analysis.periods.get(period.key)
        if existing is None or (
            _source_sort_key(period.source),
            (period.name or "").casefold(),
            period.name or "",
        ) < (
            _source_sort_key(existing.source),
            (existing.name or "").casefold(),
            existing.name or "",
        ):
            analysis.periods[period.key] = period

    for candidate in seasons:
        add_period(candidate.name, candidate.start_date, candidate.end_date, candidate.source)
    for candidate in rates:
        add_period(
            candidate.season,
            candidate.start_date,
            candidate.end_date,
            candidate.source,
        )

    periods = sorted(
        analysis.periods.values(),
        key=lambda item: (item.start, item.end, _normalise(item.name or "")),
    )
    for index, first in enumerate(periods):
        for second in periods[index + 1 :]:
            if second.start > first.end:
                break
            if first.key != second.key:
                analysis.affected.update((first.key, second.key))
                analysis.issues.append(
                    f"{_period_label(first)} overlaps {_period_label(second)}."
                )

    for previous, following in zip(periods, periods[1:]):
        if previous.end < following.start - timedelta(days=1):
            analysis.issues.append(
                f"There is an uncovered date gap between {_period_label(previous)} "
                f"and {_period_label(following)}."
            )

    valid_from = _contract_date(contract.get("valid_from"))
    valid_to = _contract_date(contract.get("valid_to"))
    if periods and valid_from and valid_to:
        if periods[0].start > valid_from:
            analysis.affected.add(periods[0].key)
            analysis.issues.append(
                f"Rate periods begin after contract validity starts on {valid_from.isoformat()}."
            )
        if periods[-1].end < valid_to:
            analysis.affected.add(periods[-1].key)
            analysis.issues.append(
                f"Rate periods end before contract validity ends on {valid_to.isoformat()}."
            )
        for period in periods:
            if period.start < valid_from or period.end > valid_to:
                analysis.affected.add(period.key)
                analysis.issues.append(
                    f"{_period_label(period)} falls outside the contract validity dates."
                )
    return analysis


def _period_options(analysis: _PeriodAnalysis) -> list[dict[str, str]]:
    affected = [
        period
        for key, period in sorted(
            analysis.periods.items(),
            key=lambda item: (item[1].start, item[1].end, item[0][0]),
        )
        if key in analysis.affected
    ]
    options = [
        {
            "label": _period_label(period),
            "loads": f"Use the stated dates {_date_range(period.start, period.end)}.",
        }
        for period in affected[:4]
    ]
    if len(options) < 2:
        options.append(
            {
                "label": (
                    "Confirm the uncovered period"
                    if not affected
                    else "Confirm corrected dates"
                ),
                "loads": (
                    "Confirm whether the uncovered date gap is intentional; do not extend or repair the stated periods."
                    if not affected
                    else "Load affected rates only after the contracting team confirms the dates."
                ),
            }
        )
    return options[:4]


def _reconcile_rates(
    rate_records: Sequence[CandidateRecord],
    periods: _PeriodAnalysis,
    contract: dict[str, Any],
    output_rates: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    not_loaded: list[dict[str, Any]],
) -> None:
    candidates = sorted(
        (record.candidate for record in rate_records),
        key=lambda candidate: (
            _source_sort_key(candidate.source.model_dump(mode="json")),
            _candidate_payload_sort_key(candidate),
        ),
    )
    loadable: list[dict[str, Any]] = []
    standard_rates = [
        candidate
        for candidate in candidates
        if candidate.occupancy != "extra_bed"
        and candidate.rate_type not in {"group", "rack"}
        and candidate.room
    ]
    rooms = sorted(
        {candidate.room for candidate in standard_rates if candidate.room},
        key=lambda room: (_normalise(room), room.casefold(), room),
    )
    period_values = sorted(
        periods.periods.values(),
        key=lambda period: (
            period.start,
            period.end,
            _normalise(period.name or ""),
            _source_sort_key(period.source),
        ),
    )
    valid_from = _contract_date(contract.get("valid_from"))
    valid_to = _contract_date(contract.get("valid_to"))

    for candidate in candidates:
        if candidate.rate_type in {"group", "rack"}:
            _append_not_loaded(
                not_loaded,
                candidate.source.model_dump(mode="json"),
                candidate.source.quote,
                "Group and rack rates are outside the individual contracted rates to load.",
                identity=_rate_not_loaded_identity(candidate),
            )
            continue
        meal_plan = _normalise_meal_plan(candidate.meal_plan)
        if meal_plan is None or candidate.price <= 0:
            continue

        if candidate.occupancy == "extra_bed":
            expansions = _extra_bed_expansions(candidate, rooms, period_values)
            if not expansions:
                _append_not_loaded(
                    not_loaded,
                    candidate.source.model_dump(mode="json"),
                    candidate.source.quote,
                    "The extra-bed rate does not clearly establish its room and period applicability.",
                    identity=_rate_not_loaded_identity(candidate, "extra_bed"),
                )
                continue
            for room, start, end, season_key in expansions:
                if (
                    season_key in periods.affected
                    or valid_from and start < valid_from
                    or valid_to and end > valid_to
                ):
                    continue
                loadable.append(
                    _rate_output(candidate, room, meal_plan, "extra_bed", start, end)
                )
            continue

        resolved_period = _resolve_rate_period(candidate, period_values)
        if resolved_period is None:
            continue
        if resolved_period.key in periods.affected:
            continue
        if (
            valid_from
            and resolved_period.start < valid_from
            or valid_to
            and resolved_period.end > valid_to
        ):
            continue
        if not candidate.room:
            continue
        if candidate.occupancy == "unspecified":
            if candidate.occupancy_applicability != "single_and_double":
                _append_not_loaded(
                    not_loaded,
                    candidate.source.model_dump(mode="json"),
                    candidate.source.quote,
                    "The source does not establish that this price applies to both single and double occupancy.",
                    identity=_rate_not_loaded_identity(candidate, "occupancy"),
                )
                continue
            conflicting_prices = [
                other
                for other in candidates
                if other is not candidate
                and other.room
                and _normalise(other.room) == _normalise(candidate.room)
                and _normalise_meal_plan(other.meal_plan) == meal_plan
                and other.occupancy in {"single", "double"}
                and float(other.price) != float(candidate.price)
                and _resolve_rate_period(other, period_values) == resolved_period
            ]
            if conflicting_prices:
                _append_not_loaded(
                    not_loaded,
                    candidate.source.model_dump(mode="json"),
                    candidate.source.quote,
                    "Explicit occupancy rates conflict with expanding this one price to both occupancies.",
                    identity=_rate_not_loaded_identity(candidate, "occupancy"),
                )
                continue
            occupancies = ("single", "double")
        else:
            occupancies = (candidate.occupancy,)
        for occupancy in occupancies:
            if occupancy not in {"single", "double"}:
                continue
            loadable.append(
                _rate_output(
                    candidate,
                    candidate.room,
                    meal_plan,
                    occupancy,
                    resolved_period.start,
                    resolved_period.end,
                )
            )

    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for rate in loadable:
        key = (
            rate["start"],
            rate["end"],
            _normalise(rate["room"]),
            rate["meal_plan"],
            rate["occupancy"],
        )
        by_key[key].append(rate)

    for key in sorted(by_key):
        competing = by_key[key]
        prices = {float(item["price"]) for item in competing}
        if len(prices) > 1:
            sources = [item["source"] for item in competing]
            options = [
                {
                    "label": f"{item['room']} {item['meal_plan']} {item['occupancy']}: {item['price']}",
                    "loads": f"Load price {item['price']} for {_date_range(date.fromisoformat(item['start']), date.fromisoformat(item['end']))}.",
                }
                for item in sorted(
                    competing,
                    key=lambda value: (
                        value["price"],
                        _source_sort_key(value["source"]),
                        json.dumps(value, sort_keys=True, ensure_ascii=False),
                    ),
                )[:4]
            ]
            _append_question(
                questions,
                _question(
                    "rates",
                    "Which conflicting room rate is correct?",
                    "Verified sources give different prices for the same room, period, meal plan, and occupancy.",
                    options,
                    sources,
                ),
            )
            continue
        chosen = min(competing, key=lambda item: _source_sort_key(item["source"]))
        output_rates.append(chosen)


def _extra_bed_expansions(
    candidate: RoomRateCandidate,
    rooms: Sequence[str],
    periods: Sequence[_Period],
) -> list[tuple[str, date, date, Optional[tuple[str, date, date]]]]:
    if candidate.room and candidate.start_date and candidate.end_date:
        matching = [
            period for period in periods
            if period.start == candidate.start_date and period.end == candidate.end_date
        ]
        if candidate.season:
            matching = [
                period for period in matching
                if _normalise(period.name or "") == _normalise(candidate.season)
            ]
        if len(matching) > 1:
            return []
        key = matching[0].key if matching else None
        return [(candidate.room, candidate.start_date, candidate.end_date, key)]

    quote = candidate.source.quote
    scope_text = _normalise(quote)
    # The assignment explicitly maps its room-unspecified third-guest extra-bed
    # rate to all room types. Period coverage still needs source evidence.
    assignment_applies_to_all_rooms = (
        candidate.room is None
        and candidate.occupancy == "extra_bed"
        and bool(
            re.search(
                r"\bextra[\s-]+bed\s+for\s+(?:a\s+)?third\s+guest\b",
                quote,
                re.IGNORECASE,
            )
        )
        and not re.search(
            r"\b(?:no|not|never)\s+(?:an?\s+)?extra[\s-]+bed\b",
            quote,
            re.IGNORECASE,
        )
    )
    applies_all_periods = (
        "all periods" in scope_text
        or "every period" in scope_text
    )
    applies_all_rooms = bool(
        re.search(
            r"\b(?:all|every|any)\s+(?:relevant\s+)?(?:room|room type|room category)s?\b",
            quote,
            flags=re.IGNORECASE,
        )
        or assignment_applies_to_all_rooms
    )
    named_rooms = [
        room
        for room in rooms
        if re.search(rf"\b{re.escape(room)}\b", quote, re.IGNORECASE)
    ]
    has_exact_period = candidate.start_date is not None and candidate.end_date is not None
    has_named_period = candidate.season is not None
    if not (applies_all_periods or has_exact_period or has_named_period):
        return []
    if candidate.room is None and not applies_all_rooms and not named_rooms:
        return []

    selected_periods = [
        period
        for period in periods
        if (candidate.season is None or _normalise(candidate.season) == _normalise(period.name or ""))
        and (
            candidate.start_date is None
            or candidate.start_date == period.start
        )
        and (
            candidate.end_date is None
            or candidate.end_date == period.end
        )
    ]
    selected_rooms = (
        [candidate.room]
        if candidate.room
        else (named_rooms if named_rooms else (list(rooms) if applies_all_rooms else []))
    )
    if not selected_periods or not selected_rooms:
        return []
    return [
        (room, period.start, period.end, period.key)
        for room in selected_rooms
        for period in selected_periods
    ]


def _resolve_rate_period(
    candidate: RoomRateCandidate, periods: Sequence[_Period]
) -> Optional[_Period]:
    if candidate.start_date and candidate.end_date:
        matching = [
            period
            for period in periods
            if period.start == candidate.start_date
            and period.end == candidate.end_date
            and (
                candidate.season is None
                or not period.name
                or _normalise(period.name) == _normalise(candidate.season)
            )
        ]
        if len(matching) > 1:
            return None
        if matching:
            return min(matching, key=lambda item: _source_sort_key(item.source))
        return _Period(
            candidate.season,
            candidate.start_date,
            candidate.end_date,
            candidate.source.model_dump(mode="json"),
        )
    if candidate.season:
        matching = [
            period for period in periods
            if _normalise(period.name or "") == _normalise(candidate.season)
        ]
        if len(matching) == 1:
            return matching[0]
    return None


def _rate_output(
    candidate: RoomRateCandidate,
    room: str,
    meal_plan: str,
    occupancy: str,
    start: date,
    end: date,
) -> dict[str, Any]:
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "room": room,
        "meal_plan": meal_plan,
        "occupancy": occupancy,
        "price": candidate.price,
        "source": candidate.source.model_dump(mode="json"),
    }


def _normalise_meal_plan(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = re.sub(r"[\s/&-]+", " ", value.strip().casefold())
    normalized = " ".join(normalized.split())
    return _MEAL_PLAN_ALIASES.get(normalized)


def _collect_not_loaded_terms(
    terms: Sequence[CommercialTermCandidate],
    output: list[dict[str, Any]],
) -> None:
    reasons = {
        "group_rate": "Group rates are outside the individual contracted rates to load.",
        "rack_rate": "Rack rates are public reference prices, not contracted rates to load.",
        "exclusion": "The cited exclusion is not represented as a loadable rate or policy.",
        "festive_surcharge": "The surcharge cannot be loaded without complete applicable dates and amount.",
        "child_policy": (
            "Child-policy terms are not represented in the company output schema; "
            "retain the cited wording for review."
        ),
    }
    for candidate in terms:
        reason = reasons.get(candidate.kind)
        if reason is None:
            continue
        if candidate.kind == "child_policy" and not _is_relevant_child_policy(
            candidate.source.quote
        ):
            continue
        _append_not_loaded(
            output,
            candidate.source.model_dump(mode="json"),
            candidate.source.quote,
            reason,
            identity=_commercial_not_loaded_identity(candidate),
        )


def _is_relevant_child_policy(quote: str) -> bool:
    if not re.search(r"\b(?:child(?:ren)?|kids?|infants?)\b", quote, re.I):
        return False
    return bool(
        re.search(
            r"\b(?:age|aged|under|years?\s+old|free|complimentary|charge|charged|"
            r"pay|price|rate|extra\s+bed|bedding|share|breakfast|meal|discount|"
            r"permitted|allowed|prohibited|not\s+allowed|accompanied|supervised|"
            r"stay|welcome)\b",
            quote,
            re.I,
        )
    )


def _reconcile_cancellations(
    candidates: Sequence[CancellationCandidate],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tiers = []
    for candidate in candidates:
        if candidate.charge_percent <= 0:
            continue
        deadline = candidate.days_max
        if deadline is None:
            deadline = candidate.days_min
        if deadline is None:
            continue
        tiers.append(
            {
                "within_days": deadline,
                "charge_percent": candidate.charge_percent,
                "source": candidate.source.model_dump(mode="json"),
                "days_min": candidate.days_min,
            }
        )

    unique: dict[tuple[int, float], dict[str, Any]] = {}
    for tier in tiers:
        key = (tier["within_days"], float(tier["charge_percent"]))
        existing = unique.get(key)
        if existing is None or _source_sort_key(tier["source"]) < _source_sort_key(existing["source"]):
            unique[key] = tier
    tiers = list(unique.values())

    conflict = any(
        _different_sources(first["source"], second["source"])
        and (
            first["within_days"] == second["within_days"]
            and first["charge_percent"] != second["charge_percent"]
            or first["charge_percent"] == second["charge_percent"]
            and first["within_days"] != second["within_days"]
        )
        for index, first in enumerate(tiers)
        for second in tiers[index + 1 :]
    )
    if conflict:
        options = [
            {
                "label": f"{tier['charge_percent']:g}% within {tier['within_days']} days",
                "loads": f"Charge {tier['charge_percent']:g}% if cancelled within {tier['within_days']} days of arrival.",
            }
            for tier in sorted(
                tiers,
                key=lambda item: (item["within_days"], item["charge_percent"]),
            )[:4]
        ]
        _append_question(
            questions,
            _question(
                "cancellation",
                "Which cancellation penalty terms apply?",
                "Verified source statements disagree about cancellation deadlines or penalties; no policy was selected.",
                options,
                [tier["source"] for tier in tiers],
            ),
        )
        return []

    return [
        {
            "within_days": tier["within_days"],
            "charge_percent": tier["charge_percent"],
            "source": tier["source"],
        }
        for tier in sorted(tiers, key=lambda item: (-item["within_days"], item["charge_percent"]))
    ]


def _reconcile_advance_payments(
    records: Sequence[CandidateRecord],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    schedules: dict[str, dict[str, Any]] = {}
    for record in records:
        candidate = record.candidate
        if not isinstance(candidate, AdvancePaymentCandidate):
            continue
        source = candidate.source.model_dump(mode="json")
        quote = candidate.source.quote
        key = record.schedule_id or (
            "unassociated:"
            + _candidate_payload_sort_key(candidate)
        )
        schedule = schedules.setdefault(
            key, {"entries": [], "balances": [], "sources": [], "issue": None}
        )
        schedule["sources"].append(source)

        if candidate.percentage is not None and candidate.percentage > 0:
            timing_text = candidate.timing or quote
            timing = _payment_timing(timing_text) or _payment_timing(quote)
            if timing is None and candidate.balance_wording:
                timing = _payment_timing(candidate.balance_wording)
            if timing is None:
                schedule["issue"] = "An installment has no supported payment timing."
            else:
                schedule["entries"].append(
                    {
                        "percent": Decimal(str(candidate.percentage)),
                        "when": timing[0],
                        "days_before_arrival": timing[1],
                        "source": source,
                    }
                )

        balance_text = candidate.balance_wording
        if balance_text and _is_balance_text(balance_text):
            timing = _payment_timing(balance_text)
            if timing is None:
                timing = _payment_timing(quote, prefer_balance=True)
            if timing is None:
                schedule["issue"] = "The balance has no supported payment timing."
            else:
                explicit_percent = _percentage_in_balance_text(balance_text)
                schedule["balances"].append(
                    {
                        "percent": (
                            Decimal(str(explicit_percent))
                            if explicit_percent is not None
                            else None
                        ),
                        "when": timing[0],
                        "days_before_arrival": timing[1],
                        "source": source,
                    }
                )

    reconciled_schedules = []
    incomplete_sources = []
    for key in sorted(schedules):
        schedule = schedules[key]
        entries_by_key = {}
        for entry in sorted(
            schedule["entries"],
            key=lambda item: (
                item["when"],
                item["days_before_arrival"] or 0,
                item["percent"],
                _source_sort_key(item["source"]),
            ),
        ):
            entries_by_key.setdefault(
                (entry["when"], entry["days_before_arrival"], entry["percent"]),
                entry,
            )
        entries = list(entries_by_key.values())
        balances_by_key = {}
        for balance in sorted(
            schedule["balances"],
            key=lambda item: (
                item["when"],
                item["days_before_arrival"] or 0,
                item["percent"] or Decimal("0"),
                _source_sort_key(item["source"]),
            ),
        ):
            balances_by_key.setdefault(
                (balance["when"], balance["days_before_arrival"], balance["percent"]),
                balance,
            )
        balances = list(balances_by_key.values())
        if len(balances) > 1:
            schedule["issue"] = "The schedule contains multiple balance installments."
        if balances:
            balance = balances[0]
            if balance["percent"] is None:
                advance_total = sum(
                    (entry["percent"] for entry in entries), Decimal("0")
                )
                remaining = Decimal("100") - advance_total
                if entries and Decimal("0") < remaining <= Decimal("100"):
                    balance["percent"] = remaining
                else:
                    schedule["issue"] = (
                        "The balance percentage cannot be calculated from this schedule."
                    )
            if balance["percent"] is not None:
                entries.append(balance)

        unique_entries = {}
        for entry in sorted(
            entries,
            key=lambda item: (
                item["when"],
                item["days_before_arrival"] or 0,
                item["percent"],
                _source_sort_key(item["source"]),
            ),
        ):
            unique_entries.setdefault(
                (entry["when"], entry["days_before_arrival"], entry["percent"]),
                entry,
            )
        entries = list(unique_entries.values())
        total = sum((entry["percent"] for entry in entries), Decimal("0"))
        if any(
            entry["percent"] is None
            or not Decimal("0") < entry["percent"] <= Decimal("100")
            for entry in entries
        ):
            schedule["issue"] = "An installment percentage is outside the valid range."
        elif total > Decimal("100"):
            schedule["issue"] = "The complete payment schedule exceeds 100%."
        elif balances and total != Decimal("100"):
            schedule["issue"] = "The complete payment schedule does not total 100%."
        elif total < Decimal("100"):
            schedule["issue"] = "The payment schedule is incomplete."
        elif total == Decimal("100") and schedule["issue"] is None:
            reconciled_schedules.append(entries)

        if schedule["issue"] is not None:
            incomplete_sources.extend(schedule["sources"])

    if incomplete_sources:
        _append_question(
            questions,
            _question(
                "advance_payment",
                "What is the complete advance-payment schedule?",
                " ".join(
                    sorted(
                        {
                            schedule["issue"]
                            for schedule in schedules.values()
                            if schedule["issue"] is not None
                        }
                    )
                ),
                [],
                incomplete_sources,
            ),
        )
        return []

    if not reconciled_schedules:
        return []
    signatures = {
        tuple(
            sorted(
                [
                    (
                        entry["when"],
                        entry["days_before_arrival"],
                        entry["percent"],
                    )
                    for entry in entries
                ],
                key=lambda item: (
                    item[0],
                    item[1] if item[1] is not None else -1,
                    item[2],
                ),
            )
        )
        for entries in reconciled_schedules
    }
    if len(signatures) > 1:
        all_entries = [entry for entries in reconciled_schedules for entry in entries]
        options = [
            {
                "label": f"{entry['percent']:g}% {entry['when'].replace('_', ' ')}",
                "loads": _advance_load_text(entry),
            }
            for entry in sorted(
                all_entries,
                key=lambda item: (
                    item["when"],
                    item["days_before_arrival"] or 0,
                    item["percent"],
                    _source_sort_key(item["source"]),
                ),
            )[:4]
        ]
        _append_question(
            questions,
            _question(
                "advance_payment",
                "Which advance-payment schedule applies?",
                "Separate verified payment clauses establish different complete schedules.",
                options,
                [
                    entry["source"]
                    for entries in reconciled_schedules
                    for entry in entries
                ],
            ),
        )
        return []

    chosen = min(
        reconciled_schedules,
        key=lambda entries: min(_source_sort_key(entry["source"]) for entry in entries),
    )
    unique_entries = {
        (
            entry["when"],
            entry["days_before_arrival"],
            entry["percent"],
        ): entry
        for entry in chosen
    }
    return [
        {
            "percent": float(entry["percent"]),
            "when": entry["when"],
            "days_before_arrival": entry["days_before_arrival"],
            "source": entry["source"],
        }
        for entry in sorted(
            unique_entries.values(),
            key=lambda item: (
                item["when"],
                item["days_before_arrival"] or 0,
                item["percent"],
            ),
        )
    ]


def _payment_timing(
    text: str, prefer_balance: bool = False
) -> Optional[tuple[str, Optional[int]]]:
    clauses = re.split(r"[;\n.!?]+", text)
    if prefer_balance:
        clauses = [clause for clause in clauses if _is_balance_text(clause)] or clauses
    for clause in clauses:
        if re.search(
            r"\b(?:at|upon|on)\s+(?:the\s+)?(?:time\s+of\s+)?"
            r"(?:booking|confirmation|reservation)\b",
            clause,
            flags=re.IGNORECASE,
        ) or re.search(r"\bat\s+booking\b", clause, flags=re.IGNORECASE):
            return "at_booking", None
        day_match = re.search(
            r"\b(\d+)\s+days?\s+(?:before|prior\s+to)\s+arrival\b",
            clause,
            flags=re.IGNORECASE,
        )
        if day_match:
            return "before_arrival", int(day_match.group(1))
        if re.search(r"\bbefore\s+arrival\b|\bprior\s+to\s+arrival\b", clause, re.I):
            return "before_arrival", None
    return None


def _is_balance_text(text: str) -> bool:
    return bool(re.search(r"\b(?:balance|remaining\s+balance|remainder)\b", text, re.I))


def _percentage_in_balance_text(text: str) -> Optional[float]:
    if not _is_balance_text(text):
        return None
    match = re.search(
        r"\b(?:balance|remaining\s+balance|remainder)\b[^%]{0,40}?"
        r"(\d+(?:\.\d+)?)\s*(?:%|percent)",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return float(match.group(1))


def _advance_load_text(entry: dict[str, Any]) -> str:
    if entry["when"] == "at_booking":
        return f"Pay {entry['percent']:g}% at booking."
    days = entry["days_before_arrival"]
    if days is None:
        return f"Pay {entry['percent']:g}% before arrival."
    return f"Pay {entry['percent']:g}% {days} days before arrival."


def _question(
    about: str,
    question: str,
    why: str,
    options: Sequence[dict[str, str]],
    sources: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    unique_options = _canonical_options(options)
    if len(unique_options) < 2:
        unique_options.append(
            {
                "label": "Confirm with the contracting team",
                "loads": "Do not load the disputed value until it is confirmed.",
            }
        )
    if len(unique_options) < 2:
        unique_options.append(
            {
                "label": "Leave the value unresolved",
                "loads": "Do not load the disputed value.",
            }
        )
    unique_options = _canonical_options(unique_options)
    return {
        "about": about,
        "question": question,
        "why": why,
        "options": unique_options[:4],
        "sources": _unique_sources(sources),
    }


def _canonical_options(
    options: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    unique = {
        (option["label"], option["loads"]): option
        for option in options
    }
    return sorted(
        unique.values(),
        key=lambda option: (
            _normalise(option["label"]),
            _normalise(option["loads"]),
            option["label"],
            option["loads"],
        ),
    )


def _append_question(
    questions: list[dict[str, Any]], question: dict[str, Any]
) -> None:
    key = (
        question["about"],
        _normalise(question["question"]),
        _normalise(question["why"]),
    )
    for existing in questions:
        existing_key = (
            existing["about"],
            _normalise(existing["question"]),
            _normalise(existing["why"]),
        )
        if existing_key != key:
            continue
        existing["question"] = min(
            (existing["question"], question["question"]),
            key=lambda value: (value.casefold(), value),
        )
        existing["why"] = min(
            (existing["why"], question["why"]),
            key=lambda value: (value.casefold(), value),
        )
        existing["options"] = _canonical_options(
            existing["options"] + question["options"]
        )[:4]
        existing["sources"] = _unique_sources(
            existing["sources"] + question["sources"]
        )
        return
    questions.append(question)


def _append_not_loaded(
    output: list[dict[str, Any]],
    source: dict[str, Any],
    text: str,
    reason: str,
    identity: Optional[tuple[Any, ...]] = None,
) -> None:
    if not text.strip():
        return
    item = {"text": text, "reason": reason, "source": source}
    if identity is not None:
        item["_identity"] = identity
    output.append(item)


def _rate_not_loaded_identity(
    candidate: RoomRateCandidate, kind: Optional[str] = None
) -> tuple[Any, ...]:
    return (
        "rate",
        kind or candidate.rate_type,
        _normalise(candidate.room or "") or None,
        _normalise_meal_plan(candidate.meal_plan),
        candidate.occupancy,
        _normalise(candidate.season or "") or None,
        candidate.start_date.isoformat() if candidate.start_date else None,
        candidate.end_date.isoformat() if candidate.end_date else None,
        Decimal(str(candidate.price)),
        _currency_in_text(candidate.source.quote),
    )


def _commercial_not_loaded_identity(
    candidate: CommercialTermCandidate,
) -> tuple[Any, ...]:
    rate_kind = (
        "group" if candidate.kind == "group_rate"
        else "rack" if candidate.kind == "rack_rate"
        else candidate.kind
    )
    identity = (
        "rate" if candidate.kind in {"group_rate", "rack_rate"} else "term",
        rate_kind,
        _normalise(candidate.room or "") or None,
        _normalise_meal_plan(candidate.meal_plan)
        or _meal_plan_in_text(f"{candidate.description} {candidate.source.quote}"),
        candidate.occupancy,
        _normalise(candidate.season or "") or None,
        candidate.start_date.isoformat() if candidate.start_date else None,
        candidate.end_date.isoformat() if candidate.end_date else None,
        Decimal(str(candidate.amount)) if candidate.amount is not None else None,
        (candidate.currency or _currency_in_text(candidate.source.quote) or "").upper()
        or None,
    )
    return (
        identity
        if candidate.kind in {"group_rate", "rack_rate"}
        else identity + (_normalise(candidate.description),)
    )


def _currency_in_text(text: str) -> Optional[str]:
    match = re.search(r"\b(INR|USD|EUR|GBP|AED|AUD|CAD)\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _not_loaded_identities_match(
    first: tuple[Any, ...], second: tuple[Any, ...]
) -> bool:
    if not first or not second or first[0] != second[0]:
        return False
    if (
        first[0] == "term"
        and "child_policy" in {first[1], second[1]}
    ):
        return first[2:-1] == second[2:-1]
    if first[1] != second[1]:
        return False
    if first[0] == "term" and first[-1] != second[-1]:
        return False
    first_fields = first[2:-1] if first[0] == "term" else first[2:]
    second_fields = second[2:-1] if second[0] == "term" else second[2:]
    if len(first_fields) != len(second_fields):
        return False
    for first_value, second_value in zip(first_fields, second_fields):
        if first_value is not None and second_value is not None:
            if first_value != second_value:
                return False
    return True


def _quotes_describe_same_term(first: str, second: str) -> bool:
    first_words = _normalise(first)
    second_words = _normalise(second)
    first_tokens = set(first_words.split())
    second_tokens = set(second_words.split())
    return bool(
        first_words
        and second_words
        and (
            first_words in second_words
            or second_words in first_words
            or first_tokens <= second_tokens
            or second_tokens <= first_tokens
        )
    )


def _deduplicate_not_loaded(items: list[dict[str, Any]]) -> None:
    ordered = sorted(
        items,
        key=lambda item: (
            _source_sort_key(item["source"]),
            repr(item.get("_identity")),
            _normalise(item["text"]),
            _normalise(item["reason"]),
        ),
    )
    unique: list[dict[str, Any]] = []
    identities: list[tuple[Any, ...]] = []
    for item in ordered:
        identity = item.get("_identity")
        source = item["source"]
        duplicate_index = None
        if identity is not None:
            for index, existing in enumerate(unique):
                existing_source = existing["source"]
                if (
                    source.get("file") == existing_source.get("file")
                    and source.get("page") == existing_source.get("page")
                    and _not_loaded_identities_match(identity, identities[index])
                    and (
                        _normalise(source.get("quote", ""))
                        == _normalise(existing_source.get("quote", ""))
                        if identity[0] == "term"
                        and "child_policy"
                        in {identity[1], identities[index][1]}
                        else _quotes_describe_same_term(
                            source.get("quote", ""), existing_source.get("quote", "")
                        )
                    )
                ):
                    duplicate_index = index
                    break
        if duplicate_index is not None:
            existing = unique[duplicate_index]
            existing_identity = identities[duplicate_index]
            reason_kind = (
                "child_policy"
                if "child_policy" in {identity[1], existing_identity[1]}
                else identity[1]
            )
            canonical_reason = {
                "group": "Group rates are outside the individual contracted rates to load.",
                "rack": "Rack rates are public reference prices, not contracted rates to load.",
                "child_policy": (
                    "Child-policy terms are not represented in the company output "
                    "schema; retain the cited wording for review."
                ),
            }.get(reason_kind)
            if canonical_reason:
                existing["reason"] = canonical_reason
            continue
        unique.append(
            {
                "text": item["text"],
                "reason": item["reason"],
                "source": source,
            }
        )
        identities.append(identity or ())
    items[:] = sorted(
        unique,
        key=lambda item: (
            _source_sort_key(item["source"]),
            _normalise(item["text"]),
            _normalise(item["reason"]),
            json.dumps(item, sort_keys=True, ensure_ascii=False),
        ),
    )


def _period_label(period: _Period) -> str:
    name = f"{period.name} " if period.name else ""
    return f"{name}{_date_range(period.start, period.end)}".strip()


def _date_range(start: date, end: date) -> str:
    return f"{start.isoformat()} to {end.isoformat()}"


def _contract_date(value: Optional[dict[str, Any]]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value["value"])
    except (KeyError, TypeError, ValueError):
        return None


def _normalise(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _meal_plan_in_text(text: str) -> Optional[str]:
    normalized = _normalise(text)
    aliases = sorted(_MEAL_PLAN_ALIASES, key=len, reverse=True)
    for alias in aliases:
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized):
            return _MEAL_PLAN_ALIASES[alias]
    return None


def _unique_sources(sources: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for source in sources:
        if not source.get("file") or source.get("page", 0) < 1 or not source.get("quote"):
            continue
        key = (source["file"], source["page"], source["quote"])
        if key not in seen:
            seen.add(key)
            result.append(source)
    return sorted(result, key=_source_sort_key)


def _candidate_sort_key(candidate: ContractFactCandidate) -> tuple[Any, ...]:
    return (
        _source_sort_key(candidate.source.model_dump(mode="json")),
        _candidate_payload_sort_key(candidate),
    )


def _candidate_payload_sort_key(candidate: Any) -> str:
    return json.dumps(
        candidate.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
    )


def _source_sort_key(source: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(source.get("file", "")).casefold(),
        int(source.get("page", 0)),
        str(source.get("quote", "")),
        str(source.get("file", "")),
    )


def _different_sources(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return (
        first.get("file") != second.get("file")
        or first.get("page") != second.get("page")
        or first.get("quote") != second.get("quote")
    )
