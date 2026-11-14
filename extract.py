"""LLM extraction of evidence candidates from ingested contract pages."""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Optional, Sequence, Union
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ingest import IngestionTimings, PageContent, ingest_folder

load_dotenv()

MealPlan = Literal[
    "EP",
    "CP",
    "MAP",
    "AP",
    "Room Only",
    "Breakfast",
    "BB",
    "With Breakfast",
    "Breakfast and Dinner",
    "Breakfast & Dinner",
    "Half Board",
    "Full Board",
]
Occupancy = Literal["single", "double", "extra_bed", "unspecified"]
RateType = Literal["individual", "group", "rack", "unspecified"]


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceSource(EvidenceModel):
    file: str = ""
    page: int = 0
    quote: str = ""


class ContractFactCandidate(EvidenceModel):
    field: Literal[
        "hotel_name",
        "validity_start",
        "validity_end",
        "currency",
        "tax_included",
    ]
    value: Union[bool, str]
    source: EvidenceSource


class SeasonCandidate(EvidenceModel):
    name: str = Field(min_length=1)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    source: EvidenceSource


class RoomRateCandidate(EvidenceModel):
    room: Optional[str] = None
    meal_plan: Optional[MealPlan] = None
    occupancy: Occupancy
    occupancy_applicability: Literal["single_and_double", "unknown"] = "unknown"
    price: float = Field(gt=0)
    season: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    rate_type: RateType
    source: EvidenceSource


class CancellationCandidate(EvidenceModel):
    charge_percent: float = Field(ge=0, le=100)
    days_min: Optional[int] = Field(default=None, ge=0)
    days_max: Optional[int] = Field(default=None, ge=0)
    source: EvidenceSource


class AdvancePaymentCandidate(EvidenceModel):
    percentage: Optional[float] = Field(default=None, ge=0, le=100)
    timing: Optional[str] = None
    balance_wording: Optional[str] = None
    source: EvidenceSource


class CommercialTermCandidate(EvidenceModel):
    kind: Literal[
        "festive_surcharge",
        "child_policy",
        "group_rate",
        "rack_rate",
        "exclusion",
        "other",
    ]
    description: str = Field(min_length=1)
    amount: Optional[float] = Field(default=None, gt=0)
    currency: Optional[str] = None
    room: Optional[str] = None
    meal_plan: Optional[MealPlan] = None
    occupancy: Optional[Occupancy] = None
    season: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    source: EvidenceSource


class ExtractionResult(EvidenceModel):
    """Unreconciled candidate facts; this is not the final contract schema."""

    contract_facts: list[ContractFactCandidate] = Field(default_factory=list)
    seasons: list[SeasonCandidate] = Field(default_factory=list)
    room_rates: list[RoomRateCandidate] = Field(default_factory=list)
    cancellations: list[CancellationCandidate] = Field(default_factory=list)
    advance_payments: list[AdvancePaymentCandidate] = Field(default_factory=list)
    commercial_terms: list[CommercialTermCandidate] = Field(default_factory=list)


class TokenUsage(EvidenceModel):
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    estimated_cost_usd: Optional[float] = Field(default=None, ge=0)
    preflight_input_token_bound: Optional[int] = Field(default=None, ge=0)
    preflight_estimated_cost_usd: Optional[float] = Field(default=None, ge=0)


class ExtractionTimings(EvidenceModel):
    pdf_discovery_ms: float = Field(default=0, ge=0)
    pdf_ingestion_ms: float = Field(default=0, ge=0)
    rendering_ms: float = Field(default=0, ge=0)
    ocr_ms: float = Field(default=0, ge=0)
    prompt_construction_preflight_ms: float = Field(default=0, ge=0)
    model_api_request_response_ms: float = Field(default=0, ge=0)
    pydantic_response_parsing_ms: float = Field(default=0, ge=0)
    evidence_validation_ms: float = Field(default=0, ge=0)
    total_extraction_ms: float = Field(default=0, ge=0)


CandidateCategory = Literal[
    "contract_fact",
    "season",
    "room_rate",
    "cancellation",
    "advance_payment",
    "commercial_term",
]

CandidatePayload = Union[
    ContractFactCandidate,
    SeasonCandidate,
    RoomRateCandidate,
    CancellationCandidate,
    AdvancePaymentCandidate,
    CommercialTermCandidate,
]


class CandidateRecord(EvidenceModel):
    """A stable audit identity wrapped around one unchanged extraction candidate."""

    candidate_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    category: CandidateCategory
    candidate: CandidatePayload
    schedule_id: Optional[str] = None

    @model_validator(mode="after")
    def category_matches_candidate(self) -> "CandidateRecord":
        expected_types = {
            "contract_fact": ContractFactCandidate,
            "season": SeasonCandidate,
            "room_rate": RoomRateCandidate,
            "cancellation": CancellationCandidate,
            "advance_payment": AdvancePaymentCandidate,
            "commercial_term": CommercialTermCandidate,
        }
        if not isinstance(self.candidate, expected_types[self.category]):
            raise ValueError("Candidate category does not match its candidate model")
        if self.schedule_id is not None and self.category != "advance_payment":
            raise ValueError("Only advance-payment candidates may have a schedule ID")
        return self


class CandidateEvidenceAssessment(EvidenceModel):
    """Deterministic evidence status linked to a stable candidate ID."""

    candidate_id: str = Field(min_length=1)
    category: CandidateCategory
    status: Literal["verified", "uncertain", "invalid"]
    issues: tuple[str, ...] = ()


class AssessmentIntegrityError(ValueError):
    """A run contains missing, duplicate, or mismatched assessment references."""


class ExtractionRun(EvidenceModel):
    model: str
    candidate_records: tuple[CandidateRecord, ...]
    evidence_assessments: tuple[CandidateEvidenceAssessment, ...]
    usage: TokenUsage
    timings: ExtractionTimings = Field(default_factory=ExtractionTimings)

    @model_validator(mode="after")
    def assessment_links_are_complete(self) -> "ExtractionRun":
        self._assessment_map()
        return self

    def _assessment_map(self) -> dict[str, CandidateEvidenceAssessment]:
        record_ids = [record.candidate_id for record in self.candidate_records]
        assessment_ids = [item.candidate_id for item in self.evidence_assessments]
        if len(record_ids) != len(set(record_ids)):
            raise AssessmentIntegrityError("Duplicate candidate IDs in extraction run")
        if len(assessment_ids) != len(set(assessment_ids)):
            raise AssessmentIntegrityError("Duplicate evidence assessments for a candidate")
        if set(record_ids) != set(assessment_ids):
            missing = set(record_ids) - set(assessment_ids)
            orphaned = set(assessment_ids) - set(record_ids)
            raise AssessmentIntegrityError(
                f"Assessment references are incomplete (missing={len(missing)}, "
                f"orphaned={len(orphaned)})"
            )
        records_by_id = {record.candidate_id: record for record in self.candidate_records}
        assessments_by_id = {
            item.candidate_id: item for item in self.evidence_assessments
        }
        for candidate_id, assessment in assessments_by_id.items():
            if records_by_id[candidate_id].category != assessment.category:
                raise AssessmentIntegrityError(
                    f"Assessment category does not match candidate {candidate_id}"
                )
        return assessments_by_id

    def verified_candidates(
        self, category: Optional[CandidateCategory] = None
    ) -> tuple[CandidateRecord, ...]:
        """Return only explicitly verified records; fail if links are inconsistent."""
        return self.candidates_with_status("verified", category)

    def candidates_with_status(
        self,
        status: Literal["verified", "uncertain", "invalid"],
        category: Optional[CandidateCategory] = None,
    ) -> tuple[CandidateRecord, ...]:
        """Return records with one explicit assessment status after link validation."""
        return tuple(
            record
            for record, _ in self.assessed_candidates(status, category)
        )

    def assessed_candidates(
        self,
        status: Literal["verified", "uncertain", "invalid"],
        category: Optional[CandidateCategory] = None,
    ) -> tuple[tuple[CandidateRecord, CandidateEvidenceAssessment], ...]:
        """Return candidates paired with validated assessments of one status."""
        assessments = self._assessment_map()
        return tuple(
            (record, assessments[record.candidate_id])
            for record in self.candidate_records
            if (category is None or record.category == category)
            and assessments[record.candidate_id].status == status
        )


class ExtractionError(RuntimeError):
    """The structured extraction request failed or returned invalid evidence."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        classification: Optional[str] = None,
        provider_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.classification = classification
        self.provider_code = provider_code
        self.details = {
            "status_code": self.status_code,
            "classification": self.classification,
            "provider_code": self.provider_code,
        }


class ModelPricingConfigurationError(ExtractionError):
    """The selected model does not have an explicit, matching price configuration."""


@dataclass(frozen=True)
class ExtractionConfig:
    model: str = "gemini-3.8-flash"
    pricing_model: Optional[str] = None
    input_cost_per_million: Optional[float] = None
    output_cost_per_million: Optional[float] = None

    @classmethod
    def from_env(cls) -> "ExtractionConfig":
        config = cls(
            model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip(),
            pricing_model=(
                os.environ.get("GEMINI_PRICING_MODEL", "").strip() or None
            ),
            input_cost_per_million=_optional_env_price(
                "GEMINI_INPUT_COST_PER_MILLION"
            ),
            output_cost_per_million=_optional_env_price(
                "GEMINI_OUTPUT_COST_PER_MILLION"
            ),
        )
        if not config.model:
            raise ModelPricingConfigurationError("GEMINI_MODEL must not be empty")
        return config

    def require_model_prices(self) -> tuple[float, float]:
        if not self.pricing_model:
            raise ModelPricingConfigurationError(
                "Set GEMINI_PRICING_MODEL to the exact selected model and provide "
                "both per-million token prices before making an API request."
            )
        if self.pricing_model != self.model:
            raise ModelPricingConfigurationError(
                f"Pricing is configured for {self.pricing_model!r}, but "
                f"GEMINI_MODEL selects {self.model!r}."
            )
        if (
            self.input_cost_per_million is None
            or self.output_cost_per_million is None
        ):
            raise ModelPricingConfigurationError(
                "Both GEMINI_INPUT_COST_PER_MILLION and "
                "GEMINI_OUTPUT_COST_PER_MILLION must be explicitly configured."
            )
        rates = (self.input_cost_per_million, self.output_cost_per_million)
        if any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise ModelPricingConfigurationError(
                "Model token prices must be finite and zero or greater."
            )
        return rates


def _optional_env_price(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ModelPricingConfigurationError(
            f"{name} must be a numeric USD price per million tokens."
        ) from exc


class ExtractionRequestTooLarge(ExtractionError):
    """The complete evidence request exceeds a deterministic safety limit."""


class EstimatedCostLimitError(ExtractionError):
    """A conservative or usage-based estimate exceeds the per-contract budget."""


MAX_EXTRACTION_PAGES = 30
MAX_EXTRACTION_REQUEST_CHARS = 120_000
MAX_COMPLETION_TOKENS = 8192
MAX_ESTIMATED_API_COST_USD = 0.15
REQUEST_WRAPPER_TOKEN_BOUND = 32


def extract_folder(
    folder: str | Path,
    config: Optional[ExtractionConfig] = None,
    client: Optional[Any] = None,
    *,
    pages: Optional[Sequence[PageContent]] = None,
) -> ExtractionRun:
    """Extract evidence candidates from every PDF page in a contract folder."""
    total_started = perf_counter()
    settings = config or ExtractionConfig.from_env()
    ingestion_timings = IngestionTimings()
    page_evidence = (
        list(pages)
        if pages is not None
        else ingest_folder(folder, timings=ingestion_timings)
    )
    if not page_evidence:
        return ExtractionRun(
            model=settings.model,
            candidate_records=(),
            evidence_assessments=(),
            usage=TokenUsage(),
            timings=_extraction_timings(
                ingestion_timings,
                total_started,
            ),
        )

    prompt_started = perf_counter()
    prompt = _load_prompt()
    page_payload = [_page_for_model(page) for page in page_evidence]
    message = json.dumps({"pages": page_payload}, ensure_ascii=False)
    request_chars = len(prompt) + len(message)
    if len(page_evidence) > MAX_EXTRACTION_PAGES:
        raise ExtractionRequestTooLarge(
            f"Contract has {len(page_evidence)} pages; the extraction limit is "
            f"{MAX_EXTRACTION_PAGES}. No evidence was truncated or sent."
        )
    if request_chars > MAX_EXTRACTION_REQUEST_CHARS:
        raise ExtractionRequestTooLarge(
            f"Complete extraction request is {request_chars:,} characters; the "
            f"limit is {MAX_EXTRACTION_REQUEST_CHARS:,}. No evidence was truncated "
            "or sent."
        )
    input_price, output_price = settings.require_model_prices()
    response_schema = json.dumps(
        ExtractionResult.model_json_schema(), ensure_ascii=False
    )
    input_token_bound = (
        len(prompt.encode("utf-8"))
        + len(message.encode("utf-8"))
        + len(response_schema.encode("utf-8"))
        + REQUEST_WRAPPER_TOKEN_BOUND
    )
    preflight_cost = (
        input_token_bound * input_price + MAX_COMPLETION_TOKENS * output_price
    ) / 1_000_000
    if preflight_cost > MAX_ESTIMATED_API_COST_USD:
        raise EstimatedCostLimitError(
            "The conservative preflight estimate "
            f"${preflight_cost:.8f} exceeds the "
            f"${MAX_ESTIMATED_API_COST_USD:.2f} per-contract limit; no request was sent."
        )
    prompt_preflight_ms = (perf_counter() - prompt_started) * 1000

    request_started = perf_counter()
    active_client = client if client is not None else _create_client(timeout_seconds=180)
    response = _request_extraction(active_client, settings.model, prompt, message)
    request_response_ms = (perf_counter() - request_started) * 1000

    parsing_started = perf_counter()
    candidates, input_tokens, output_tokens, total_tokens = _read_response(response)
    response_parsing_ms = (perf_counter() - parsing_started) * 1000

    validation_started = perf_counter()
    candidate_records = _candidate_records(candidates, page_evidence)
    evidence_assessments = _validate_sources(candidate_records, page_evidence)
    evidence_validation_ms = (perf_counter() - validation_started) * 1000

    estimated_cost = None
    if input_tokens is not None and output_tokens is not None:
        estimated_cost = (
            input_tokens * input_price
            + output_tokens * output_price
        ) / 1_000_000
    return ExtractionRun(
        model=settings.model,
        candidate_records=candidate_records,
        evidence_assessments=evidence_assessments,
        usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
            preflight_input_token_bound=input_token_bound,
            preflight_estimated_cost_usd=preflight_cost,
        ),
        timings=_extraction_timings(
            ingestion_timings,
            total_started,
            prompt_preflight_ms=prompt_preflight_ms,
            request_response_ms=request_response_ms,
            response_parsing_ms=response_parsing_ms,
            evidence_validation_ms=evidence_validation_ms,
        ),
    )


def _extraction_timings(
    ingestion: IngestionTimings,
    total_started: float,
    *,
    prompt_preflight_ms: float = 0,
    request_response_ms: float = 0,
    response_parsing_ms: float = 0,
    evidence_validation_ms: float = 0,
) -> ExtractionTimings:
    return ExtractionTimings(
        pdf_discovery_ms=ingestion.pdf_discovery_seconds * 1000,
        pdf_ingestion_ms=ingestion.pdf_ingestion_seconds * 1000,
        rendering_ms=ingestion.rendering_seconds * 1000,
        ocr_ms=ingestion.ocr_seconds * 1000,
        prompt_construction_preflight_ms=prompt_preflight_ms,
        model_api_request_response_ms=request_response_ms,
        pydantic_response_parsing_ms=response_parsing_ms,
        evidence_validation_ms=evidence_validation_ms,
        total_extraction_ms=(perf_counter() - total_started) * 1000,
    )


def _create_client(timeout_seconds: Optional[float] = None) -> Any:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ExtractionError(
            "Set GEMINI_API_KEY in the environment to enable extraction"
        )
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ExtractionError(
            "The Google Gen AI SDK is missing. Install the project requirements first."
        ) from exc
    timeout_ms = None
    if timeout_seconds is not None:
        timeout_ms = max(1, int(timeout_seconds * 1000))
    http_options = types.HttpOptions(
        timeout=timeout_ms,
        retry_options=types.HttpRetryOptions(attempts=1),
    )
    return genai.Client(api_key=api_key, http_options=http_options)


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent / "prompts" / "extraction_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def _page_for_model(page: PageContent) -> dict:
    return {
        "file": page.file,
        "page": page.page,
        "extraction_method": page.extraction_method,
        "ocr_confidence": page.ocr_confidence,
        "text": page.text,
        "lines": [line for line in page.to_dict()["lines"]],
    }


def _request_extraction(
    client: Any,
    model: str,
    prompt: str,
    page_message: str,
) -> Any:
    try:
        from google.genai import errors, types
        from httpx2 import HTTPError
    except ImportError as exc:
        raise ExtractionError(
            "The Google Gen AI SDK or HTTP transport is missing. "
            "Install the project requirements first."
        ) from exc

    try:
        return client.models.generate_content(
            model=model,
            contents=page_message,
            config=types.GenerateContentConfig(
                system_instruction=prompt,
                response_mime_type="application/json",
                response_schema=_gemini_compatible_response_schema(),
                max_output_tokens=MAX_COMPLETION_TOKENS,
            ),
        )
    except errors.APIError as exc:
        status_code = getattr(exc, "code", None)
        if not isinstance(status_code, int):
            status_code = None
        provider_code, provider_details = _gemini_error_details(exc)
        raise ExtractionError(
            "Gemini structured extraction failed",
            status_code=status_code,
            classification=_classify_gemini_api_error(
                status_code, provider_code, provider_details
            ),
            provider_code=provider_code or None,
        ) from exc
    except HTTPError as exc:
        raise ExtractionError(
            "Gemini network request failed",
            classification="network_error",
        ) from exc
    except ValidationError as exc:
        raise ExtractionError(
            "Gemini request configuration did not match the evidence schema",
            classification="configuration_error",
        ) from exc


def _gemini_compatible_response_schema() -> dict[str, Any]:
    schema = ExtractionResult.model_json_schema()
    schema["$defs"]["RoomRateCandidate"]["properties"]["price"].pop(
        "exclusiveMinimum", None
    )
    amount_schema = schema["$defs"]["CommercialTermCandidate"]["properties"][
        "amount"
    ]
    for variant in amount_schema.get("anyOf", []):
        variant.pop("exclusiveMinimum", None)
    _remove_additional_properties(schema)
    return schema


def _remove_additional_properties(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("additionalProperties", None)
        for value in node.values():
            _remove_additional_properties(value)
    elif isinstance(node, list):
        for value in node:
            _remove_additional_properties(value)


def _gemini_error_details(error: BaseException) -> tuple[str, str]:
    response_json = getattr(error, "details", None)
    if response_json is None:
        response_json = getattr(error, "response_json", None)
    body_error = (
        response_json.get("error", response_json)
        if isinstance(response_json, dict)
        else {}
    )
    if not isinstance(body_error, dict):
        body_error = {}
    code = body_error.get("status") or body_error.get("code") or ""
    message = body_error.get("message") or getattr(error, "message", "") or ""
    return str(code), str(message)


def _classify_gemini_api_error(
    status_code: Optional[int],
    provider_code: str,
    details: str,
) -> str:
    normalized = f"{provider_code} {details}".casefold()
    if status_code in {401, 403} or any(
        phrase in normalized
        for phrase in (
            "api_key_invalid",
            "api key not valid",
            "invalid api key",
            "unauthenticated",
            "permission denied",
        )
    ):
        return "authentication_error"

    token_limit = bool(
        re.search(r"\btokens?\s+(?:per|/)\s*min(?:ute)?\b|\btpm\b", normalized)
    )
    request_limit = bool(
        re.search(r"\brequests?\s+(?:per|/)\s*min(?:ute)?\b|\brpm\b", normalized)
    )
    if token_limit and not request_limit:
        return "tokens_per_minute_limit"
    if request_limit and not token_limit:
        return "requests_per_minute_limit"
    if status_code == 429:
        if any(phrase in normalized for phrase in ("quota", "billing")):
            return "quota_exceeded"
        return "rate_limit"
    if status_code in {400, 413} and any(
        phrase in normalized
        for phrase in (
            "context_length",
            "maximum context",
            "context window",
            "request too large",
            "too many tokens",
        )
    ):
        return "request_size_limit"
    return "api_error"


def _read_response(
    response: Any,
) -> tuple[ExtractionResult, Optional[int], Optional[int], Optional[int]]:
    try:
        text = getattr(response, "text", None)
    except (AttributeError, ValueError) as exc:
        raise ExtractionError(
            "Gemini response did not contain readable JSON",
            classification="invalid_response",
        ) from exc
    if not isinstance(text, str) or not text.strip():
        candidates = getattr(response, "candidates", None) or []
        finish_reason = (
            getattr(candidates[0], "finish_reason", None) if candidates else None
        )
        if str(getattr(finish_reason, "name", finish_reason)).casefold() == "max_tokens":
            message = "Gemini response exceeded the configured output-token limit"
        else:
            message = "Gemini response did not contain structured candidate JSON"
        raise ExtractionError(message, classification="invalid_response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            "Gemini returned malformed JSON",
            classification="invalid_response",
        ) from exc
    if not isinstance(payload, dict):
        raise ExtractionError(
            "Gemini response JSON must be an object",
            classification="invalid_response",
        )
    missing_fields = sorted(set(ExtractionResult.model_fields) - payload.keys())
    if missing_fields:
        raise ExtractionError(
            "Gemini response omitted required candidate categories: "
            + ", ".join(missing_fields),
            classification="invalid_response",
        )
    try:
        candidates = ExtractionResult.model_validate(payload)
    except ValidationError as exc:
        raise ExtractionError(
            "Gemini response did not match the evidence schema",
            classification="invalid_response",
        ) from exc

    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return candidates, None, None, None
    try:
        input_tokens = _optional_usage_count(usage, "prompt_token_count")
        output_tokens = _optional_usage_count(usage, "candidates_token_count")
        total_tokens = _optional_usage_count(usage, "total_token_count")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ExtractionError(
            "Gemini response contained malformed token usage",
            classification="invalid_response",
        ) from exc
    return candidates, input_tokens, output_tokens, total_tokens


def _optional_usage_count(usage: Any, name: str) -> Optional[int]:
    value = getattr(usage, name, None)
    if value is None:
        return None
    count = int(value)
    if count < 0:
        raise ValueError(f"{name} must not be negative")
    return count


def _validate_sources(
    records: Union[ExtractionResult, Sequence[CandidateRecord]],
    pages: Sequence[PageContent],
) -> tuple[CandidateEvidenceAssessment, ...]:
    """Assess each candidate independently; conflicts never remove candidates."""
    if isinstance(records, ExtractionResult):
        records = _candidate_records(records)
    pages_by_source: dict[tuple[str, int], list[PageContent]] = {}
    for page in pages:
        pages_by_source.setdefault((page.file, page.page), []).append(page)
    assessments = []
    for record in records:
        source = record.candidate.source
        matching_pages = pages_by_source.get(
            (source.file, source.page), []
        )
        quote_pages = [
            item for item in matching_pages if source.quote and source.quote in item.text
        ]
        if len(quote_pages) == 1:
            page = quote_pages[0]
        elif not quote_pages and len(matching_pages) == 1:
            page = matching_pages[0]
        else:
            page = None
        if page is None:
            document_pages = ()
        elif page.document_id:
            document_pages = tuple(
                item for item in pages if item.document_id == page.document_id
            )
        else:
            document_pages = tuple(
                item for item in pages if item.file == page.file
            )
        assessments.append(_assess_candidate(record, page, document_pages))
    return tuple(assessments)


def _candidate_records(
    result: ExtractionResult,
    pages: Sequence[PageContent] = (),
) -> tuple[CandidateRecord, ...]:
    records = tuple(
        CandidateRecord(category=category, candidate=candidate)
        for category, candidates in (
            ("contract_fact", result.contract_facts),
            ("season", result.seasons),
            ("room_rate", result.room_rates),
            ("cancellation", result.cancellations),
            ("advance_payment", result.advance_payments),
            ("commercial_term", result.commercial_terms),
        )
        for candidate in candidates
    )
    return _assign_payment_schedule_ids(records, pages) if pages else records


def _resolve_source_page(
    source: EvidenceSource, pages: Sequence[PageContent]
) -> Optional[PageContent]:
    matches = [
        page
        for page in pages
        if page.file == source.file and page.page == source.page
    ]
    quote_matches = [
        page for page in matches if source.quote and source.quote in page.text
    ]
    if len(quote_matches) == 1:
        return quote_matches[0]
    return matches[0] if not quote_matches and len(matches) == 1 else None


def _assign_payment_schedule_ids(
    records: Sequence[CandidateRecord],
    pages: Sequence[PageContent],
) -> tuple[CandidateRecord, ...]:
    locations: dict[str, tuple[PageContent, int, int, int, Optional[int], str]] = {}
    for record in records:
        if record.category != "advance_payment":
            continue
        page = _resolve_source_page(record.candidate.source, pages)
        if page is None:
            continue
        quote = record.candidate.source.quote
        start = page.text.find(quote)
        if start < 0 or page.text.find(quote, start + 1) >= 0:
            continue
        end = start + len(quote)
        start_line = page.text.count("\n", 0, start)
        end_line = page.text.count("\n", 0, end)
        lines = page.text.splitlines()
        section_index = None
        for index in range(min(start_line, len(lines))):
            if _is_payment_section_heading(lines[index]):
                section_index = index
            elif section_index is not None and _is_payment_section_boundary(lines[index]):
                section_index = None
        if section_index is not None:
            for index in range(section_index + 1, min(start_line + 1, len(lines))):
                if _is_payment_section_heading(lines[index]):
                    section_index = index
        schedule_label = _payment_schedule_label(lines, start_line)
        document_key = page.document_id or page.file
        document_hash = hashlib.sha256(
            document_key.encode("utf-8")
        ).hexdigest()[:16]
        section_key = f"{document_hash}:{page.page}:{section_index}"
        locations[record.candidate_id] = (
            page,
            start,
            end,
            start_line,
            section_index,
            f"{section_key}:{schedule_label}",
        )

    parent = {candidate_id: candidate_id for candidate_id in locations}

    def find(candidate_id: str) -> str:
        while parent[candidate_id] != candidate_id:
            parent[candidate_id] = parent[parent[candidate_id]]
            candidate_id = parent[candidate_id]
        return candidate_id

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    records_by_id = {record.candidate_id: record for record in records}
    location_ids = sorted(locations)
    for offset, first_id in enumerate(location_ids):
        first_page, first_start, first_end, _, _, first_section = locations[first_id]
        for second_id in location_ids[offset + 1 :]:
            second_page, second_start, second_end, _, _, second_section = locations[second_id]
            if (
                first_page.document_id != second_page.document_id
                or first_page.file != second_page.file
                or first_page.page != second_page.page
                or first_section != second_section
            ):
                continue
            if first_start < second_end and second_start < first_end:
                union(first_id, second_id)
            elif (
                first_section == second_section
                and not first_section.endswith(":default")
            ):
                union(first_id, second_id)

    for balance_id in location_ids:
        balance_record = records_by_id[balance_id]
        balance_candidate = balance_record.candidate
        balance_text = (
            balance_candidate.balance_wording or balance_candidate.source.quote
        )
        if not _is_balance_payment_text(balance_text):
            continue
        balance_location = locations[balance_id]
        prior_candidates = []
        for advance_id in location_ids:
            if advance_id == balance_id:
                continue
            advance_record = records_by_id[advance_id]
            advance_candidate = advance_record.candidate
            advance_location = locations[advance_id]
            if (
                advance_location[0].document_id != balance_location[0].document_id
                or advance_location[0].file != balance_location[0].file
                or advance_location[0].page != balance_location[0].page
                or advance_location[5] != balance_location[5]
                or advance_location[1] >= balance_location[1]
                or balance_location[1] - advance_location[2] > 300
                or balance_location[3] - advance_location[3] > 2
                or advance_candidate.percentage is None
                or _is_balance_payment_text(
                    advance_candidate.balance_wording
                    or advance_candidate.source.quote
                )
            ):
                continue
            prior_candidates.append(advance_id)
        if len(prior_candidates) == 1:
            union(prior_candidates[0], balance_id)

    grouped_starts: dict[str, list[int]] = {}
    for candidate_id in location_ids:
        grouped_starts.setdefault(find(candidate_id), []).append(
            locations[candidate_id][1]
        )
    schedule_ids = {}
    for candidate_id in location_ids:
        page, _, _, _, section_index, section_key = locations[candidate_id]
        root_id = find(candidate_id)
        earliest_start = min(grouped_starts[root_id])
        schedule_ids[candidate_id] = (
            f"{section_key}:{earliest_start}"
            if section_index is not None
            else f"{section_key}:clause:{earliest_start}"
        )

    return tuple(
        record.model_copy(update={"schedule_id": schedule_ids.get(record.candidate_id)})
        if record.category == "advance_payment"
        else record
        for record in records
    )


def _is_payment_section_heading(text: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:advance\s+)?payment(?:\s+(?:terms?|schedule|policy))?\s*:?\s*$",
            text,
            flags=re.IGNORECASE,
        )
    )


def _is_balance_payment_text(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:balance|remaining\s+balance|remainder)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _is_payment_section_boundary(text: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:cancellation|other|children|check[\s-]?in(?:\s+and\s+check[\s-]?out)?|"
            r"identification|pets\s+and\s+smoking|terms|policies)\s*:?\s*$",
            text,
            flags=re.IGNORECASE,
        )
    )


def _payment_schedule_label(lines: Sequence[str], line_index: int) -> str:
    labels = []
    for index, line in enumerate(lines[: line_index + 1]):
        match = re.search(
            r"\b(?:payment\s+)?(?:schedule|option|alternative)\s+([A-Z]|\d+)\b",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            labels.append(f"{match.group(1).casefold()}@{index}")
    return labels[-1] if labels else "default"


def _assess_candidate(
    record: CandidateRecord,
    page: Optional[PageContent],
    document_pages: Sequence[PageContent] = (),
) -> CandidateEvidenceAssessment:
    issues: list[str] = []
    checks: list[Optional[bool]] = []
    candidate = record.candidate
    source = candidate.source

    if not source.file or source.page < 1:
        issues.append("Source filename or 1-based page number is missing.")
        checks.append(False)
    if page is None:
        issues.append("Source does not identify an ingested page.")
        checks.append(False)
    if not source.quote:
        issues.append("Supporting quote is missing.")
        checks.append(False)
    elif page is not None and source.quote not in page.text:
        issues.append("Supporting quote is not an exact substring of the page text.")
        checks.append(False)
    if any(check is False for check in checks):
        return CandidateEvidenceAssessment(
            candidate_id=record.candidate_id,
            category=record.category,
            status="invalid",
            issues=tuple(issues),
        )

    quote = source.quote
    confidence_check = (
        True
        if isinstance(candidate, RoomRateCandidate)
        else _ocr_quote_confidence_check(page, quote)
    )
    if confidence_check is not True:
        _add_check(
            checks,
            issues,
            confidence_check,
            "Supporting OCR words are missing or below the confidence threshold.",
        )
    if isinstance(candidate, ContractFactCandidate):
        check, issue = _validate_contract_fact(candidate, quote)
        _add_check(checks, issues, check, issue)
    elif isinstance(candidate, SeasonCandidate):
        _add_check(
            checks,
            issues,
            _text_value_check(candidate.name, quote),
            "Season name is not supported by the quote.",
        )
        if candidate.start_date is not None or candidate.end_date is not None:
            _add_check(
                checks,
                issues,
                _season_dates_check(candidate.start_date, candidate.end_date, quote),
                "Season start/end dates are not supported in the stated order.",
            )
    elif isinstance(candidate, RoomRateCandidate):
        _validate_room_rate(
            candidate, page, checks, issues, document_pages=document_pages
        )
    elif isinstance(candidate, CancellationCandidate):
        _validate_cancellation(candidate, quote, checks, issues)
    elif isinstance(candidate, AdvancePaymentCandidate):
        _validate_advance_payment(candidate, quote, checks, issues)
    elif isinstance(candidate, CommercialTermCandidate):
        _validate_commercial_term(candidate, quote, checks, issues)

    if any(check is False for check in checks):
        status = "invalid"
    elif any(check is None for check in checks):
        status = "uncertain"
    else:
        status = "verified"
    return CandidateEvidenceAssessment(
        candidate_id=record.candidate_id,
        category=record.category,
        status=status,
        issues=tuple(issues),
    )


def _add_check(
    checks: list[Optional[bool]],
    issues: list[str],
    check: Optional[bool],
    issue: str,
) -> None:
    checks.append(check)
    if check is not True:
        issues.append(issue)


def _normalised_words(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return re.findall(r"[a-z0-9]+", ascii_text.lower())


def _contains_phrase(text: str, phrase: str) -> bool:
    text_words = _normalised_words(text)
    phrase_words = _normalised_words(phrase)
    if not phrase_words or len(phrase_words) > len(text_words):
        return False
    return any(
        text_words[index : index + len(phrase_words)] == phrase_words
        for index in range(len(text_words) - len(phrase_words) + 1)
    )


def _text_value_check(value: Optional[str], text: str) -> Optional[bool]:
    if value is None or not value.strip():
        return None
    return _contains_phrase(text, value)


def _ocr_quote_confidence_check(
    page: Optional[PageContent], quote: str
) -> Optional[bool]:
    if page is None or page.extraction_method != "ocr":
        return True
    quote_words = _normalised_words(quote)
    supporting_lines = []
    for line in page.lines:
        line_words = _normalised_words(line.text)
        if not line_words:
            continue
        if (
            _contains_phrase(" ".join(quote_words), " ".join(line_words))
            or _contains_phrase(" ".join(line_words), " ".join(quote_words))
        ):
            supporting_lines.append(line)
    if not supporting_lines or any(not line.words for line in supporting_lines):
        return None
    if any(
        word.confidence is None
        or word.confidence < 50
        or word.width <= 0
        or word.height <= 0
        for line in supporting_lines
        for word in line.words
    ):
        return None
    return True


def _phrase_match(text: str, phrase: str) -> Optional[re.Match]:
    words = _normalised_words(phrase)
    if not words:
        return None
    pattern = r"(?<!\w)" + r"[\W_]+".join(re.escape(word) for word in words) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE)


def _hotel_name_check(value: str, quote: str) -> Optional[bool]:
    match = _phrase_match(quote, value)
    if match is None:
        return False
    left = max(quote.rfind(mark, 0, match.start()) for mark in ".!?;\n") + 1
    right_candidates = [
        position
        for mark in ".!?;\n"
        if (position := quote.find(mark, match.end())) >= 0
    ]
    right = min(right_candidates) if right_candidates else len(quote)
    before = quote[left : match.start()]
    after = quote[match.end() : right]
    if re.search(
        r"\b(?:does\s+not|doesn't|did\s+not|never|not|is\s+not|isn't|"
        r"are\s+not|aren't|was\s+not|weren't|no\s+longer)\b[^.!?;\n]{0,70}$",
        before,
        flags=re.IGNORECASE,
    ):
        if re.search(r"\bnot\s+only\b", before, flags=re.IGNORECASE):
            return None
        return False
    if re.match(
        r"\s*(?:is|are|was|were)\s+(?:not|n't)\b|\s*not\s+(?:the|our|a|an)\b",
        after,
        flags=re.IGNORECASE,
    ):
        return False
    identity_cue_before = re.search(
        r"\b(?:welcome\s+to|hotel(?:\s+name)?|property(?:\s+name)?|resort|"
        r"contracting\s+hotel|contract(?:ing)?\s+party|agreement\s+with|"
        r"contract\s+with)\s*(?:is|:|,|the)?\s*$",
        before,
        flags=re.IGNORECASE,
    )
    identity_cue_after = re.match(
        r"\s*(?:is\s+(?:the\s+)?(?:contracting|contracted|covered)\s+"
        r"(?:hotel|property)|is\s+the\s+hotel\s+under\s+this\s+agreement)\b",
        after,
        flags=re.IGNORECASE,
    )
    if identity_cue_before or identity_cue_after:
        return True
    return None


def _date_occurrences(text: str) -> list[tuple[date, int, int]]:
    months = {
        name.lower(): number
        for number, name in enumerate(calendar.month_name)
        if name
    }
    months.update(
        {
            name.lower(): number
            for number, name in enumerate(calendar.month_abbr)
            if name
        }
    )
    month_pattern = "|".join(
        sorted((re.escape(name) for name in months), key=len, reverse=True)
    )
    mentions: list[tuple[date, int, int]] = []

    patterns = (
        (
            rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pattern})\.?,?\s+(\d{{4}})\b",
            "day_month_year",
        ),
        (
            rf"\b({month_pattern})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
            "month_day_year",
        ),
    )
    for pattern, order in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                if order == "day_month_year":
                    day_value, month_name, year_value = match.groups()
                else:
                    month_name, day_value, year_value = match.groups()
                parsed = date(
                        int(year_value),
                        months[month_name.lower()],
                        int(day_value),
                    )
                mentions.append((parsed, match.start(), match.end()))
            except ValueError:
                continue

    for match in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text):
        try:
            parsed = date(
                int(match.group(1)), int(match.group(2)), int(match.group(3))
            )
            mentions.append((parsed, match.start(), match.end()))
        except ValueError:
            continue
    return sorted(mentions, key=lambda mention: mention[1])


def _date_mentions(text: str) -> set[date]:
    return {value for value, _, _ in _date_occurrences(text)}


def _split_clauses(text: str) -> list[str]:
    return [
        clause.strip()
        for clause in re.split(r"[;\n.!?]+", text)
        if clause.strip()
    ]


def _validity_date_check(
    value: date, field: str, text: str
) -> Optional[bool]:
    checked_role = False
    for clause in _split_clauses(text):
        occurrences = _date_occurrences(clause)
        if not occurrences:
            continue
        has_validity_cue = bool(
            re.search(
                r"\b(?:valid|validity|effective|contract period|agreement period)\b",
                clause,
                flags=re.IGNORECASE,
            )
        )
        if not has_validity_cue:
            continue
        if re.search(
            r"\b(?:not|never|no\s+longer|isn't|aren't)\b[^.!?;]{0,45}"
            r"\b(?:valid|effective)\b",
            clause,
            flags=re.IGNORECASE,
        ):
            return False
        checked_role = True
        dates = [item[0] for item in occurrences]
        if field == "validity_start":
            explicit_single_start = bool(
                re.search(r"\b(?:from|starting|commencing)\b", clause, re.IGNORECASE)
            )
            if len(dates) >= 2 or explicit_single_start:
                if value == dates[0]:
                    return True
            else:
                return None
        else:
            explicit_single_end = bool(
                re.search(r"\b(?:until|through|expires|ending)\b", clause, re.IGNORECASE)
            )
            if len(dates) >= 2 or explicit_single_end:
                if value == dates[-1]:
                    return True
            else:
                return None
    if checked_role:
        return False
    return None


def _season_dates_check(
    start_date: Optional[date],
    end_date: Optional[date],
    text: str,
) -> Optional[bool]:
    if start_date is None or end_date is None:
        return None
    if start_date > end_date:
        return False
    occurrences = _date_occurrences(text)
    dates = [item[0] for item in occurrences]
    if len(dates) < 2:
        return None
    return dates[0] == start_date and dates[-1] == end_date


def _validate_contract_fact(
    candidate: ContractFactCandidate, quote: str
) -> tuple[Optional[bool], str]:
    field = candidate.field
    value = candidate.value
    if field == "hotel_name":
        if not isinstance(value, str):
            return False, "Hotel name candidate is not text."
        check = _hotel_name_check(value, quote)
        return check, "Hotel name is not supported by the quote."
    if field == "currency":
        if not isinstance(value, str):
            return False, "Currency candidate is not text."
        check = _contract_currency_check(value, quote)
        return check, "Currency is not supported by the quote."
    if field in {"validity_start", "validity_end"}:
        if not isinstance(value, str):
            return False, f"{field.replace('_', ' ').capitalize()} is not a date."
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError:
            return False, f"{field.replace('_', ' ').capitalize()} is not a date."
        return (
            _validity_date_check(parsed_date, field, quote),
            f"{field.replace('_', ' ').capitalize()} is not supported by the quote.",
        )
    if field == "tax_included":
        if not isinstance(value, bool):
            return False, "Tax inclusion value is missing or not boolean."
        polarities = set()
        tax_clauses = [
            clause.strip()
            for clause in re.split(
                r"[;.!?]+|\bbut\b|\bhowever\b",
                quote,
                flags=re.IGNORECASE,
            )
            if clause.strip()
        ]
        for clause in tax_clauses:
            positive = re.search(
                r"\b(?:tax(?:es)?|gst|vat)\b.{0,35}\b(?:included|inclusive)\b"
                r"|\b(?:included|inclusive)\b.{0,35}\b(?:tax(?:es)?|gst|vat)\b",
                clause,
                flags=re.IGNORECASE,
            )
            negative = re.search(
                r"\b(?:tax(?:es)?|gst|vat)\b.{0,35}\b(?:not|n't|never|excluded|"
                r"excluding|exclusive|aren't|isn't|are not|is not)\b"
                r"|\b(?:not|n't|never|excluded|excluding|exclusive|aren't|isn't|"
                r"are not|is not)\b.{0,35}\b(?:tax(?:es)?|gst|vat)\b",
                clause,
                flags=re.IGNORECASE,
            )
            if negative:
                polarities.add(False)
            elif positive:
                polarities.add(True)
        if len(polarities) != 1:
            return None, "Quote does not clearly state whether taxes are included."
        return value == next(iter(polarities)), "Tax inclusion status contradicts the quote."
    return None, "Contract fact could not be assessed."


def _currency_value_check(value: Optional[str], text: str) -> Optional[bool]:
    if not value:
        return None
    if _contains_phrase(text, value):
        return True
    normalized_text = text.lower()
    markers = {
        "INR": (r"₹", r"\brupees?\b", r"\bindian rupees?\b"),
        "USD": (r"\bus dollars?\b", r"\b美元\b"),
        "EUR": (r"€", r"\beuros?\b"),
        "GBP": (r"£", r"\bpounds?\b", r"\bsterling\b"),
        "AED": (r"\bdirhams?\b",),
        "AUD": (r"\baud\b", r"\baustralian dollars?\b"),
        "CAD": (r"\bcad\b", r"\bcanadian dollars?\b"),
        "JPY": (r"¥", r"\byen\b"),
    }
    patterns = markers.get(value.upper())
    if patterns and any(
        re.search(pattern, normalized_text, re.IGNORECASE) for pattern in patterns
    ):
        if value.upper() == "JPY" and re.search(r"¥", normalized_text):
            return None
        return True
    explicit_codes = re.findall(r"\b[A-Z]{3}\b", text)
    if explicit_codes:
        return False
    symbol_markers = {
        "USD": "$",
        "AUD": "$",
        "CAD": "$",
        "INR": "₹",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
    }
    symbol = symbol_markers.get(value.upper())
    if symbol and symbol in text:
        return None if symbol == "$" else True
    return None


def _contract_currency_check(value: str, text: str) -> Optional[bool]:
    currency_role = (
        r"\bcurrenc(?:y|ies)\b|\bquoted\s+in\b|\bdenominated\s+in\b|"
        r"\b(?:rates?|prices?|amounts?)\s+(?:are\s+)?(?:quoted\s+)?in\b"
    )
    clauses = _role_clauses(text, currency_role)
    if any(_currency_value_check(value, clause) is True for clause in clauses):
        return True
    if any(_currency_value_check(value, clause) is False for clause in clauses):
        return False
    return None


def _validate_room_rate(
    candidate: RoomRateCandidate,
    page: Optional[PageContent],
    checks: list[Optional[bool]],
    issues: list[str],
    document_pages: Sequence[PageContent] = (),
) -> None:
    quote = candidate.source.quote
    row_index, row = _matching_rate_row(page, candidate)
    row_text = row.text if row is not None else quote

    for label, value in (("room name", candidate.room), ("meal plan", candidate.meal_plan)):
        if label == "room name" and value is None and candidate.occupancy == "extra_bed":
            text_check = (
                True
                if _contains_phrase(candidate.source.quote, "extra bed")
                else None
            )
        elif label == "meal plan" and page is not None and page.extraction_method == "text":
            text_check = _text_meal_plan_check(
                page, row_index, row_text, value, candidate.price
            )
        else:
            text_check = (
                _ocr_row_text_value_check(row, value)
                if page is not None and page.extraction_method == "ocr" and row is not None
                else _text_value_check(value, row_text)
            )
        _add_check(
            checks,
            issues,
            text_check,
            f"Rate {label} is not supported by the same table row.",
        )
    price_supported = _quote_contains_price(row_text, candidate.price)
    _add_check(
        checks,
        issues,
        price_supported,
        "Rate price is not present in the supporting table row.",
    )
    if (
        page is not None
        and page.extraction_method == "ocr"
        and candidate.occupancy == "unspecified"
    ):
        _add_check(
            checks,
            issues,
            _ocr_price_confidence_check(row, candidate.price),
            "OCR rate price confidence or coordinates are insufficient.",
        )

    if page is None:
        occupancy_check = None
    elif page.extraction_method == "ocr":
        if candidate.occupancy == "unspecified":
            occupancy_check = _text_unspecified_occupancy_check(
                page, row_index, row, candidate.source.quote
            )
            occupancy_issue = (
                ""
                if occupancy_check is True
                else "Unspecified OCR occupancy is not supported by this rate schedule."
            )
        else:
            occupancy_check, occupancy_issue = _ocr_occupancy_check(
                page, row_index, row, candidate
            )
        _add_check(checks, issues, occupancy_check, occupancy_issue)
    else:
        if candidate.occupancy == "unspecified":
            occupancy_check = _text_unspecified_occupancy_check(
                page, row_index, row, candidate.source.quote
            )
        else:
            occupancy_check = _text_occupancy_check(quote, candidate.occupancy)
        _add_check(
            checks,
            issues,
            occupancy_check,
            "Occupancy is not explicitly supported by the source text.",
        )

    if candidate.season is None:
        season_check = None
    elif page is not None and row_index is not None:
        season_check = _season_context_check(
            page,
            row_index,
            candidate.season,
            candidate.start_date,
            candidate.end_date,
            candidate.price,
        )
    else:
        season_check = None
    _add_check(
        checks,
        issues,
        season_check,
        "Season or season dates cannot be associated with the rate row.",
    )
    if candidate.season is None and (
        candidate.start_date is not None or candidate.end_date is not None
    ):
        _add_check(
            checks,
            issues,
            None,
            "Rate dates are present without a supported season association.",
        )

    rate_type_check = _rate_type_check(
        candidate, page, row_index, row, document_pages
    )
    _add_check(
        checks,
        issues,
        rate_type_check,
        "Rate type is not explicit in the row or its nearby heading.",
    )


def _matching_rate_row(
    page: Optional[PageContent], candidate: RoomRateCandidate
) -> tuple[Optional[int], Optional[Any]]:
    if page is None or not page.lines:
        return None, None
    quote_words = _normalised_words(candidate.source.quote)
    matching = []
    for index, line in enumerate(page.lines):
        line_words = _normalised_words(line.text)
        if not line_words or not _quote_contains_price(line.text, candidate.price):
            continue
        if candidate.room and not _contains_phrase(line.text, candidate.room):
            continue
        if quote_words and (
            quote_words == line_words
            or _contains_phrase(" ".join(quote_words), " ".join(line_words))
            or _contains_phrase(" ".join(line_words), " ".join(quote_words))
        ):
            matching.append((index, line))
    if not matching:
        if page.extraction_method != "text" or not candidate.room:
            return None, None
        for start_index, line in enumerate(page.lines):
            if not _contains_phrase(line.text, candidate.room):
                continue
            for end_index in range(start_index, min(len(page.lines), start_index + 12)):
                text = "\n".join(
                    item.text for item in page.lines[start_index : end_index + 1]
                )
                if (
                    _quote_contains_price(text, candidate.price)
                    and quote_words
                    and _contains_phrase(" ".join(_normalised_words(text)), " ".join(quote_words))
                ):
                    return start_index, type("TextRateRow", (), {"text": text, "words": []})()
        return None, None
    return max(matching, key=lambda item: len(_normalised_words(item[1].text)))


def _text_meal_plan_check(
    page: PageContent,
    row_index: Optional[int],
    row_text: str,
    value: Optional[str],
    price: float,
) -> Optional[bool]:
    if not value:
        return None
    code = _canonical_meal_plan(value)
    if code is None:
        return False
    if _contains_phrase(row_text, value) or _contains_phrase(row_text, code):
        return True
    if row_index is None:
        return None

    price_tokens = [
        float(token.replace(",", ""))
        for token in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])", row_text)
    ]
    price_indexes = [
        index for index, observed in enumerate(price_tokens)
        if abs(observed - price) < 0.000001
    ]
    if len(price_indexes) != 1:
        return None

    aliases = {
        "EP": ("room only", "ep"),
        "CP": ("with breakfast", "breakfast", "bed and breakfast", "bb", "cp"),
        "MAP": ("breakfast and dinner", "breakfast & dinner", "half board", "map"),
        "AP": ("full board", "ap"),
    }
    found = []
    header_start = max(0, row_index - 30)
    for line in reversed(page.lines[header_start:row_index]):
        normalized = " ".join(_normalised_words(line.text))
        for meal_plan, labels in aliases.items():
            if any(_contains_phrase(normalized, label) for label in labels):
                found.append(meal_plan)
        if re.search(r"\b(?:room\s+category|room\s+type|occupancy)\b", line.text, re.I):
            break
    ordered = list(reversed(found))
    if not ordered or len(ordered) != len(price_tokens):
        return None
    return ordered[price_indexes[0]] == code


def _canonical_meal_plan(value: str) -> Optional[str]:
    normalized = " ".join(_normalised_words(value))
    aliases = {
        "ep": "EP",
        "room only": "EP",
        "cp": "CP",
        "breakfast": "CP",
        "bb": "CP",
        "with breakfast": "CP",
        "bed and breakfast": "CP",
        "map": "MAP",
        "breakfast and dinner": "MAP",
        "half board": "MAP",
        "ap": "AP",
        "full board": "AP",
    }
    return aliases.get(normalized)


def _text_occupancy_check(quote: str, occupancy: str) -> Optional[bool]:
    labels = {
        "single": ("single",),
        "double": ("double",),
        "extra_bed": ("extra bed", "extra-bed"),
    }
    if occupancy == "unspecified":
        return False if any(
            _contains_phrase(quote, label)
            for values in labels.values()
            for label in values
        ) else None
    if any(_contains_phrase(quote, label) for label in labels.get(occupancy, ())):
        return True
    if any(
        _contains_phrase(quote, label)
        for key, values in labels.items()
        if key != occupancy
        for label in values
    ):
        return False
    return None


def _text_unspecified_occupancy_check(
    page: PageContent,
    row_index: Optional[int],
    row: Optional[Any],
    quote: str = "",
) -> Optional[bool]:
    if row_index is None or row is None:
        return None
    section_start = _rate_section_start(page, row_index)
    if section_start is None:
        return None
    header_index = _rate_table_header_index(page, row_index)
    if header_index is None or header_index < section_start:
        return None
    header_text = " ".join(
        line.text for line in page.lines[header_index:row_index]
    )
    if re.search(r"\boccupancy\b|\b(?:single|double)\b", header_text, re.I):
        return False

    previous_headers = [
        index
        for index in range(section_start, header_index)
        if re.search(
            r"\broom\s+(?:type|category)\b|^\s*room\s*$",
            page.lines[index].text,
            re.IGNORECASE,
        )
    ]
    table_context_start = previous_headers[-1] + 1 if previous_headers else section_start
    table_context = "\n".join(
        line.text for line in page.lines[table_context_start:header_index]
    )
    explicit_scope = re.search(
        r"\b(?:apply|applies|applicable|priced|charged)\s+to\s+both\s+single\s+and\s+double\b|"
        r"\bsame\s+(?:price|rate)\s+for\s+single\s+and\s+double\b",
        table_context,
        flags=re.IGNORECASE,
    )
    per_room_scope = re.search(
        r"\bper\s+room(?:\s+per\s+night)?\b",
        table_context,
        flags=re.IGNORECASE,
    )
    global_per_room_scope = re.search(
        r"\ball\s+rates?\b.{0,100}\bper\s+room(?:\s+per\s+night)?\b",
        "\n".join(line.text for line in page.lines[section_start:header_index]),
        flags=re.IGNORECASE,
    )
    if explicit_scope:
        if page.extraction_method == "ocr" and explicit_scope.group(0) not in quote:
            return None
        return True
    if per_room_scope or global_per_room_scope:
        scope_match = per_room_scope or global_per_room_scope
        if page.extraction_method == "ocr" and scope_match.group(0) not in quote:
            return None
        return True
    return None


def _rate_section_start(page: PageContent, row_index: int) -> Optional[int]:
    start = None
    for index, line in enumerate(page.lines[:row_index]):
        marker = _rate_type_marker(line.text)
        if marker is not None or _is_unclassified_rate_heading(line.text):
            start = index
    return start


def _rate_table_header_index(
    page: PageContent, row_index: int
) -> Optional[int]:
    for index in range(min(row_index, len(page.lines)) - 1, -1, -1):
        if re.search(r"\broom\s+(?:type|category)\b|^\s*room\s*$", page.lines[index].text, re.I):
            return index
    return None


def _rate_type_marker(text: str) -> Optional[str]:
    patterns = (
        ("individual", r"\bindividual\s+(?:booking\s+)?rates?\b|\brates?\s+for\s+individual\s+bookings\b"),
        ("group", r"\b(?:special\s+)?group\s+rates?\b|\bfor\s+groups?\b"),
        ("rack", r"\brack\s+rates?\b"),
        ("contracted", r"\bcontract(?:ed)?\s+(?:net\s+)?rates?\b|\brate\s+contract\b|\bcontract\s+rates?\b"),
    )
    matches = [
        (match.end(), kind)
        for kind, pattern in patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]
    return max(matches)[1] if matches else None


def _is_unclassified_rate_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped or re.search(r"[.!?]", stripped):
        return False
    if re.fullmatch(r"(?:room\s+)?rates?", stripped, flags=re.IGNORECASE):
        return False
    if not re.search(r"\b(?:rates?|prices?|schedule|tables?)\b", stripped, re.I):
        return False
    return _rate_type_marker(stripped) is None


def _has_individual_next_page_context(
    page: Optional[PageContent],
    document_pages: Sequence[PageContent],
) -> bool:
    if page is None:
        return False
    for context_page in document_pages:
        same_document = (
            context_page.document_id == page.document_id
            if context_page.document_id and page.document_id
            else context_page.file == page.file
        )
        if not same_document:
            continue
        if context_page.page + 1 != page.page:
            continue
        text = context_page.text
        states_individual_scope = bool(
            re.search(
                r"\brates?\s+in\s+this\s+(?:letter|agreement|contract)\s+are\s+for\s+individual\s+bookings\b|"
                r"\brates?\s+for\s+individual\s+bookings\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        points_to_next_page = bool(
            re.search(
                r"\b(?:rate\s+)?tables?\b[^.!?]{0,100}\bnext\s+page\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        if states_individual_scope and points_to_next_page:
            return True
    return False


def _ocr_row_text_value_check(line: Any, value: Optional[str]) -> Optional[bool]:
    if value is None or not value.strip():
        return None
    expected = _normalised_words(value)
    observed = [
        (token, word)
        for word in line.words
        for token in _normalised_words(word.text)
    ]
    matches = []
    for start in range(len(observed) - len(expected) + 1):
        sample = observed[start : start + len(expected)]
        if [token for token, _ in sample] == expected:
            matches.append([word for _, word in sample])
    if not matches:
        return False
    for words in matches:
        if all(
            word.confidence is not None
            and word.confidence >= 50
            and word.width > 0
            and word.height > 0
            for word in words
        ):
            return True
    return None


def _ocr_price_confidence_check(line: Optional[Any], price: float) -> Optional[bool]:
    if line is None or not line.words:
        return None
    matching = [
        word for word in line.words if _word_matches_price(word.text, price)
    ]
    if len(matching) != 1:
        return None
    word = matching[0]
    if (
        word.confidence is None
        or word.confidence < 50
        or word.width <= 0
        or word.height <= 0
    ):
        return None
    return True


def _occupancy_words(line: Any) -> list[tuple[str, Any]]:
    found = []
    words = line.words
    for index, word in enumerate(words):
        token = "".join(_normalised_words(word.text))
        if token in {"single", "double"}:
            found.append((token, word))
        elif token == "extra" and index + 1 < len(words):
            if "".join(_normalised_words(words[index + 1].text)) == "bed":
                found.append(("extra_bed", word))
    return found


def _ocr_occupancy_check(
    page: PageContent,
    row_index: Optional[int],
    row: Optional[Any],
    candidate: RoomRateCandidate,
) -> tuple[Optional[bool], str]:
    if row_index is None or row is None or not row.words:
        return None, "OCR row coordinates do not identify the candidate's rate cell."
    row_y = sum(word.y + word.height / 2 for word in row.words) / len(row.words)
    typical_height = _median_word_height(page)
    if typical_height is None:
        return None, "OCR page does not provide usable word-height geometry."
    headers = []
    for index, line in enumerate(page.lines):
        if index >= row_index or not line.words:
            continue
        header_words = _occupancy_words(line)
        if not header_words:
            continue
        header_y = sum(word.y + word.height / 2 for _, word in header_words) / len(header_words)
        vertical_gap = row_y - header_y
        if 0 < vertical_gap <= 16 * typical_height:
            headers.append((vertical_gap, header_words))
    if not headers:
        return None, "No nearby OCR occupancy header has usable coordinates."
    nearest_gap = min(gap for gap, _ in headers)
    header_words = [
        item
        for gap, words in headers
        if gap - nearest_gap <= 3 * typical_height
        for item in words
    ]
    if len(header_words) < 2:
        return None, "OCR occupancy heading does not identify multiple columns."
    if any(
        word.confidence is None or word.confidence < 30
        or word.width <= 0 or word.height <= 0
        for _, word in header_words
    ):
        return None, "OCR occupancy heading confidence or coordinates are insufficient."

    price_words = [
        word
        for word in row.words
        if _word_matches_price(word.text, candidate.price)
    ]
    if not price_words:
        return None, "The candidate price has no matching OCR word in its row."
    mapped_occupancies = []
    header_centres = sorted(
        (word.x + word.width / 2, label) for label, word in header_words
    )
    for price_word in price_words:
        if (
            price_word.confidence is None
            or price_word.confidence < 50
            or price_word.width <= 0
            or price_word.height <= 0
        ):
            return None, "OCR price confidence or coordinates are insufficient."
        cell_x = price_word.x + price_word.width / 2
        mapped = _map_x_to_column(cell_x, header_centres)
        if mapped is None:
            return None, "OCR price cell is too close to an occupancy-column boundary."
        mapped_occupancies.append(mapped)

    matching_cells = [
        occupancy for occupancy in mapped_occupancies
        if occupancy == candidate.occupancy
    ]
    if len(matching_cells) == 1:
        return True, ""
    if not matching_cells and mapped_occupancies:
        return False, "Price coordinates place the rate under a different occupancy."
    return None, "Multiple OCR price cells make the occupancy association ambiguous."


def _median_word_height(page: PageContent) -> Optional[float]:
    heights = [
        float(word.height)
        for line in page.lines
        for word in line.words
        if word.height > 0
    ]
    if not heights:
        return None
    heights.sort()
    middle = len(heights) // 2
    if len(heights) % 2:
        return heights[middle]
    return (heights[middle - 1] + heights[middle]) / 2


def _map_x_to_column(
    x: float, columns: Sequence[tuple[float, str]]
) -> Optional[str]:
    ordered = sorted(columns)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0][1]
    first_gap = ordered[1][0] - ordered[0][0]
    last_gap = ordered[-1][0] - ordered[-2][0]
    if first_gap <= 0 or last_gap <= 0:
        return None
    if x < ordered[0][0] - first_gap / 2 or x > ordered[-1][0] + last_gap / 2:
        return None
    for (left_x, left_label), (right_x, right_label) in zip(
        ordered, ordered[1:]
    ):
        gap = right_x - left_x
        if gap <= 0:
            return None
        boundary = (left_x + right_x) / 2
        if abs(x - boundary) <= gap * 0.1:
            return None
        if x < boundary:
            return left_label
    return ordered[-1][1]


def _word_matches_price(text: str, price: float) -> bool:
    for token in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])", text):
        if abs(float(token.replace(",", "")) - price) < 0.000001:
            return True
    return False


def _season_context_check(
    page: PageContent,
    row_index: int,
    season: str,
    start_date: Optional[date],
    end_date: Optional[date],
    price: float,
) -> Optional[bool]:
    anchors = []
    for index, line in enumerate(page.lines[:row_index]):
        if "season" in _normalised_words(line.text) and _date_mentions(line.text):
            anchor_y = _line_center_y(line)
            anchors.append((index, anchor_y, line))
    if not anchors:
        return None

    row = page.lines[row_index]
    if page.extraction_method != "ocr":
        if len(anchors) != 1:
            return None
        return _season_heading_check(
            season, start_date, end_date, anchors[0][2].text
        )

    row_y = _line_center_y(row)
    if row_y is None:
        return None
    preceding = [
        anchor
        for anchor in anchors
        if anchor[1] is not None and anchor[1] < row_y
    ]
    if not preceding:
        return None
    nearest_gap = min(row_y - anchor[1] for anchor in preceding)
    typical_height = _median_word_height(page)
    if typical_height is None:
        return None
    prior_rate_rows = [
        index
        for index, line in enumerate(page.lines[:row_index])
        if _line_looks_like_rate_row(line.text)
    ]
    table_start = max(prior_rate_rows, default=-1)
    table_anchors = [anchor for anchor in preceding if anchor[0] > table_start]
    if len(table_anchors) > 1:
        nearby = table_anchors
    else:
        nearby = [
            anchor
            for anchor in preceding
            if (row_y - anchor[1]) - nearest_gap <= 3 * typical_height
        ]

    if any(
        not anchor[2].words
        or any(
            word.confidence is None
            or word.confidence < 50
            or word.width <= 0
            or word.height <= 0
            for word in anchor[2].words
        )
        for anchor in nearby
    ):
        return None
    if len(nearby) == 1:
        return _season_heading_check(
            season, start_date, end_date, nearby[0][2].text
        )

    price_words = [
        word for word in row.words if _word_matches_price(word.text, price)
    ]
    if len(price_words) != 1:
        return None
    columns = [
        (
            sum(word.x + word.width / 2 for word in line.words) / len(line.words),
            line.text,
        )
        for _, _, line in nearby
    ]
    selected_text = _map_x_to_column(
        price_words[0].x + price_words[0].width / 2, columns
    )
    if selected_text is None:
        return None
    return _season_heading_check(season, start_date, end_date, selected_text)


def _line_center_y(line: Any) -> Optional[float]:
    if not line.words:
        return None
    return sum(word.y + word.height / 2 for word in line.words) / len(line.words)


def _line_looks_like_rate_row(text: str) -> bool:
    has_meal_plan = bool(
        re.search(r"\b(?:EP|CP|MAP|AP)\b", text, flags=re.IGNORECASE)
    )
    has_number = bool(
        re.search(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])", text)
    )
    return has_meal_plan and has_number


def _season_heading_check(
    season: str,
    start_date: Optional[date],
    end_date: Optional[date],
    anchor: str,
) -> Optional[bool]:
    if not _contains_phrase(anchor, season):
        return False
    occurrences = [value for value, _, _ in _date_occurrences(anchor)]
    if start_date is None or end_date is None:
        return None
    if len(occurrences) != 2 or _normalised_words(anchor).count("season") != 1:
        return None
    return occurrences[0] == start_date and occurrences[-1] == end_date


def _rate_type_check(
    candidate: RoomRateCandidate,
    page: Optional[PageContent],
    row_index: Optional[int],
    row: Optional[Any],
    document_pages: Sequence[PageContent] = (),
) -> Optional[bool]:
    active_context: Optional[str] = None
    has_local_boundary = False
    if page is not None and row_index is not None:
        for line in page.lines[: row_index + 1]:
            marker = _rate_type_marker(line.text)
            if marker is not None:
                active_context = marker
                has_local_boundary = True
            elif _is_unclassified_rate_heading(line.text):
                active_context = None
                has_local_boundary = True
    else:
        active_context = _rate_type_marker(candidate.source.quote)

    if active_context in {"group", "rack"}:
        return candidate.rate_type == active_context
    if active_context in {"individual", "contracted"}:
        return candidate.rate_type in {"individual", "unspecified"}
    if has_local_boundary:
        return None

    if _has_individual_next_page_context(page, document_pages):
        return candidate.rate_type in {"individual", "unspecified"}
    return None


def _validate_cancellation(
    candidate: CancellationCandidate,
    quote: str,
    checks: list[Optional[bool]],
    issues: list[str],
) -> None:
    percent_check = _role_percentage_check(
        quote,
        candidate.charge_percent,
        r"\bcancell?ation\b|\bcancel(?:led|ing)?\b",
    )
    _add_check(
        checks,
        issues,
        percent_check,
        "Cancellation penalty percentage is not supported by the quote.",
    )
    cancellation_role = r"\bcancell?ation\b|\bcancel(?:led|ing)?\b"
    if candidate.days_min is None and candidate.days_max is None:
        _add_check(
            checks,
            issues,
            None,
            "Cancellation deadline is not identified in the candidate.",
        )
    expected_days = {
        value for value in (candidate.days_min, candidate.days_max) if value is not None
    }
    if expected_days:
        _add_check(
            checks,
            issues,
            _role_days_check(quote, expected_days, cancellation_role),
            "Cancellation day window is not supported by the quote.",
        )


def _day_mentions(text: str) -> set[int]:
    return {
        int(match.group(1))
        for match in re.finditer(
            r"\b(\d+)\s*(?:calendar\s+)?days?\b", text, flags=re.IGNORECASE
        )
    }


def _validate_advance_payment(
    candidate: AdvancePaymentCandidate,
    quote: str,
    checks: list[Optional[bool]],
    issues: list[str],
) -> None:
    percentage_check = (
        _role_percentage_check(
            quote,
            candidate.percentage,
            r"\badvance\b|\bbalance\b|\bremaining\b|\bdeposit\b|"
            r"\bpre[- ]?payment\b|\bpayment\b|"
            r"\binstalment\b|\binstallment\b",
        )
        if candidate.percentage is not None
        else None
    )
    _add_check(
        checks,
        issues,
        percentage_check,
        "Advance percentage is not supported by the quote.",
    )
    for label, value in (
        ("payment timing", candidate.timing),
        ("balance wording", candidate.balance_wording),
    ):
        role = (
            r"\bbalance\b|\bremaining\b"
            if label == "balance wording"
            else (
                r"\badvance\b|\bdeposit\b|\bpre[- ]?payment\b|\bpayment\b|"
                r"\binstalment\b|\binstallment\b|\bbalance\b|\bremaining\b"
            )
        )
        check = _role_text_check(value, quote, role)
        _add_check(
            checks,
            issues,
            check,
            f"{label.capitalize()} is not supported by the quote.",
        )


def _role_clauses(text: str, role_pattern: str) -> list[str]:
    return [
        clause
        for clause in _split_clauses(text)
        if re.search(role_pattern, clause, flags=re.IGNORECASE)
    ]


def _role_percentage_check(
    text: str, value: Optional[float], role_pattern: str
) -> Optional[bool]:
    if value is None:
        return None
    linked = [
        _percentage_mentions(clause)
        for clause in _role_clauses(text, role_pattern)
    ]
    linked_values = set().union(*linked) if linked else set()
    if value not in linked_values:
        return False if _percentage_mentions(text) else None
    if any(
        value in _percentage_mentions(clause)
        and _percentage_is_negated(clause, value)
        for clause in _role_clauses(text, role_pattern)
    ):
        return False
    if len(linked_values) != 1:
        return None
    return True


def _percentage_is_negated(text: str, value: float) -> bool:
    for match in re.finditer(
        r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|percent(?:age)?)(?!\w)",
        text,
        flags=re.IGNORECASE,
    ):
        if float(match.group(1)) != value:
            continue
        before = text[: match.start()]
        if re.search(
            r"\b(?:not|never)\s+(?:equal\s+to\s+)?$|"
            r"\bno\b.{0,45}\b(?:cancell?ation|cancel|penalt(?:y|ies)|"
            r"charge|advance|deposit|payment)\b.{0,35}$|"
            r"\bwithout\b.{0,45}\b(?:cancell?ation|cancel|penalt(?:y|ies)|"
            r"charge|advance|deposit|payment)\b.{0,35}$",
            before,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _role_day_check(
    text: str, value: int, role_pattern: str
) -> Optional[bool]:
    linked = [_day_mentions(clause) for clause in _role_clauses(text, role_pattern)]
    linked_values = set().union(*linked) if linked else set()
    if value not in linked_values:
        return False if _day_mentions(text) else None
    return True if len(linked_values) == 1 else None


def _role_days_check(
    text: str, values: set[int], role_pattern: str
) -> Optional[bool]:
    linked = [_day_mentions(clause) for clause in _role_clauses(text, role_pattern)]
    linked_values = set().union(*linked) if linked else set()
    if not linked_values:
        return None if not _day_mentions(text) else False
    if linked_values != values:
        return False
    return True


def _role_text_check(
    value: Optional[str], text: str, role_pattern: str
) -> Optional[bool]:
    if value is None or not value.strip():
        return None
    clauses = _role_clauses(text, role_pattern)
    if any(_contains_phrase(clause, value) for clause in clauses):
        return True
    if _contains_phrase(text, value):
        return False
    return None


def _validate_commercial_term(
    candidate: CommercialTermCandidate,
    quote: str,
    checks: list[Optional[bool]],
    issues: list[str],
) -> None:
    kind_markers = {
        "festive_surcharge": ("festive", "festival", "christmas", "new year"),
        "child_policy": ("child", "children", "infant"),
        "group_rate": ("group rate", "group rates"),
        "rack_rate": ("rack rate", "rack rates"),
        "exclusion": ("excluded", "excluding", "not included", "exclusion"),
        "other": (),
    }
    markers = kind_markers[candidate.kind]
    clauses = _split_clauses(quote)
    relevant_clauses = [
        clause
        for clause in clauses
        if any(_contains_phrase(clause, marker) for marker in markers)
    ] if markers else []
    kind_check = True if relevant_clauses else (False if markers else None)
    if candidate.kind in {"festive_surcharge", "group_rate", "rack_rate"} and any(
        re.search(
            r"\b(?:no\s+(?:festive\s+)?surcharges?|not\s+applicable|not\s+charged|"
            r"not\s+payable|waived|waive)\b",
            clause,
            flags=re.IGNORECASE,
        )
        for clause in relevant_clauses
    ):
        kind_check = False
    _add_check(
        checks,
        issues,
        kind_check,
        "Commercial term category is not explicit in the quote.",
    )
    description_words = [
        word for word in _normalised_words(candidate.description)
        if word not in {"a", "an", "the", "of", "for", "and", "to", "with", "is"}
    ]
    quote_words = _normalised_words(quote)
    overlap = sum(word in quote_words for word in description_words)
    description_check: Optional[bool]
    if description_words and overlap == len(description_words):
        description_check = True
    elif description_words and overlap == 0:
        description_check = False
    else:
        description_check = None
    _add_check(
        checks,
        issues,
        description_check,
        "Commercial term description is not fully supported by the quote.",
    )
    amount_check = None
    if candidate.amount is not None:
        amount_check = (
            True
            if any(_quote_contains_price(clause, candidate.amount) for clause in relevant_clauses)
            else (
                False
                if _quote_contains_price(quote, candidate.amount) and relevant_clauses
                else None
            )
        )
    _add_check(
        checks,
        issues,
        amount_check,
        "Commercial term amount is not supported by the quote.",
    )
    if candidate.currency:
        currency_text = " ".join(relevant_clauses) if relevant_clauses else quote
        _add_check(
            checks,
            issues,
            _currency_value_check(candidate.currency, currency_text),
            "Commercial term currency is not supported by the quote.",
        )
    if candidate.room:
        _add_check(
            checks,
            issues,
            _text_value_check(candidate.room, quote),
            "Commercial term room scope is not supported by the quote.",
        )
    if candidate.meal_plan:
        _add_check(
            checks,
            issues,
            _text_value_check(candidate.meal_plan, quote)
            or (
                True
                if _canonical_meal_plan(candidate.meal_plan)
                and _contains_phrase(
                    quote, _canonical_meal_plan(candidate.meal_plan) or ""
                )
                else None
            ),
            "Commercial term meal-plan scope is not supported by the quote.",
        )
    if candidate.occupancy:
        _add_check(
            checks,
            issues,
            _text_occupancy_check(quote, candidate.occupancy),
            "Commercial term occupancy scope is not supported by the quote.",
        )
    if candidate.season:
        _add_check(
            checks,
            issues,
            _text_value_check(candidate.season, quote),
            "Commercial term season is not supported by the quote.",
        )
    if candidate.start_date is not None or candidate.end_date is not None:
        _add_check(
            checks,
            issues,
            _season_dates_check(candidate.start_date, candidate.end_date, quote),
            "Commercial term dates are not supported by the quote.",
        )


def _percentage_mentions(text: str) -> set[float]:
    return {
        float(match.group(1))
        for match in re.finditer(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|percent(?:age)?)(?!\w)",
            text,
            flags=re.IGNORECASE,
        )
    }


def _quote_contains_price(quote: str, price: float) -> bool:
    for token in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])", quote):
        if abs(float(token.replace(",", "")) - price) < 0.000001:
            return True
    return False
