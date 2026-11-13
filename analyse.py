"""Command-line entry point for one hotel contract folder."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional, Sequence

from extract import (
    MAX_ESTIMATED_API_COST_USD,
    AssessmentIntegrityError,
    EstimatedCostLimitError,
    ExtractionConfig,
    ExtractionError,
    ExtractionRequestTooLarge,
    ExtractionRun,
    ModelPricingConfigurationError,
    _create_client,
    _capture_rate_limit_headers,
    _classify_api_error,
    _is_rate_limit_header,
    extract_folder,
)
from ingest import (
    IngestionTimeoutError,
    NoPDFDocumentsError,
    TesseractUnavailable,
    ingest_folder_bounded,
)
from reconcile import ReconciliationError, reconcile_run, validate_output


MAX_RUNTIME_SECONDS = 60.0


class CLIInputError(ValueError):
    """The supplied contract-folder argument is not usable."""


class PipelineError(RuntimeError):
    """The contract could not be processed into a complete final output."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyse.py",
        description="Extract and reconcile one hotel contract folder as JSON.",
    )
    parser.add_argument(
        "contract_folder",
        help="folder containing the PDF documents for one contract",
    )
    return parser


def _process_contract(
    folder_arg: str,
    started_at: float,
) -> dict:
    deadline = started_at + MAX_RUNTIME_SECONDS
    folder = Path(folder_arg).expanduser()
    if not folder.exists():
        raise CLIInputError(f"Contract folder does not exist: {folder}")
    if not folder.is_dir():
        raise CLIInputError(f"Contract path is not a directory: {folder}")

    print("Discovering and ingesting PDF documents.", file=sys.stderr)
    try:
        pages = ingest_folder_bounded(folder, deadline=deadline)
    except NoPDFDocumentsError as exc:
        raise CLIInputError(str(exc)) from exc
    if not pages:
        raise PipelineError("PDF ingestion returned no pages; no output was generated.")
    readable_pages = [page for page in pages if page.text.strip()]
    for page in pages:
        if page.warning or page.unreadable:
            detail = page.warning or "page contains no readable text"
            print(
                f"Warning: {page.file}, page {page.page}: {detail}",
                file=sys.stderr,
            )
    if not readable_pages:
        raise PipelineError(
            "No readable page text was produced; extraction was not attempted."
        )
    print(
        f"Ingested {len(pages)} page(s).",
        file=sys.stderr,
    )

    config = ExtractionConfig.from_env()
    config.require_model_prices()
    remaining = _ensure_runtime(deadline, "preparing model extraction")
    client = _create_client(timeout_seconds=remaining)
    run = extract_folder(
        folder,
        config=config,
        client=client,
        pages=pages,
    )
    _ensure_runtime(deadline, "extracting evidence")
    _report_usage(run)
    if (
        run.usage.estimated_cost_usd is not None
        and run.usage.estimated_cost_usd > MAX_ESTIMATED_API_COST_USD
    ):
        raise EstimatedCostLimitError(
            f"Estimated API cost ${run.usage.estimated_cost_usd:.6f} exceeds the "
            f"${MAX_ESTIMATED_API_COST_USD:.2f} per-contract limit; final JSON "
            "was not emitted."
        )
    if (
        run.usage.preflight_estimated_cost_usd is not None
        and run.usage.preflight_estimated_cost_usd > MAX_ESTIMATED_API_COST_USD
    ):
        raise EstimatedCostLimitError(
            "The preflight API cost estimate exceeds the per-contract limit; "
            "final JSON was not emitted."
        )

    _ensure_runtime(deadline, "starting reconciliation")
    output = reconcile_run(run)
    validate_output(output)
    _ensure_runtime(deadline, "validating the final output")
    return output


def _ensure_runtime(deadline: float, stage: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PipelineError(
            f"The {MAX_RUNTIME_SECONDS:.0f}-second runtime budget expired while "
            f"{stage}; no final JSON was emitted."
        )
    return remaining


def _report_usage(run: ExtractionRun) -> None:
    usage = run.usage
    print(f"Model: {run.model}", file=sys.stderr)
    print(
        "Provider-reported token usage: "
        f"input={_display_count(usage.input_tokens)}, "
        f"output={_display_count(usage.output_tokens)}, "
        f"total={_display_count(usage.total_tokens)}.",
        file=sys.stderr,
    )
    if usage.preflight_estimated_cost_usd is not None:
        print(
            "Conservative preflight cost estimate: "
            f"${usage.preflight_estimated_cost_usd:.8f} using explicitly configured "
            "model pricing; this is not provider billing.",
            file=sys.stderr,
        )
    if usage.estimated_cost_usd is None:
        print(
            "Estimated API cost unavailable because the provider returned no "
            "complete token usage.",
            file=sys.stderr,
        )
    else:
        print(
            f"Estimated API cost: ${usage.estimated_cost_usd:.8f} using configured "
            "per-token rates; this is not provider-reported billing.",
            file=sys.stderr,
        )


def _display_count(value: Optional[int]) -> str:
    return str(value) if value is not None else "unavailable"


def _exception_chain(error: BaseException) -> list[BaseException]:
    chain = []
    seen = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _provider_error_details(
    error: BaseException,
) -> tuple[Optional[int], str, str, str, dict[str, str]]:
    status = getattr(error, "status_code", None)
    if not isinstance(status, int):
        status = None
    code = str(getattr(error, "provider_code", "") or "")
    classification = str(getattr(error, "classification", "") or "")
    headers = dict(getattr(error, "rate_limit_headers", {}) or {})
    details = []
    for item in _exception_chain(error):
        candidate_status = getattr(item, "status_code", None)
        if status is None and isinstance(candidate_status, int):
            status = candidate_status
        candidate_code = getattr(item, "provider_code", None) or getattr(
            item, "code", None
        )
        if candidate_code:
            code = str(candidate_code)
        if not classification:
            classification = str(getattr(item, "classification", "") or "")
        if status == 429 and not headers:
            headers = _capture_rate_limit_headers(getattr(item, "response", None))
        body = getattr(item, "body", None)
        body_error = body.get("error", body) if isinstance(body, dict) else {}
        if isinstance(body_error, dict):
            code = str(body_error.get("code") or code)
            details.extend(
                str(value)
                for value in (body_error.get("type"), body_error.get("message"))
                if value
            )
        details.append(str(item))
    normalized_details = " ".join(details).casefold()
    if not classification and status is not None:
        classification = _classify_api_error(status, code, normalized_details, headers)
    return status, code.casefold(), normalized_details, classification, headers


def _format_rate_limit_headers(headers: dict[str, str]) -> str:
    safe_headers = {
        str(name).casefold(): str(value).strip()
        for name, value in headers.items()
        if _is_rate_limit_header(str(name).casefold())
    }
    if not safe_headers:
        return " Rate-limit response headers: none of the tracked headers were present."
    rendered = ", ".join(
        f"{name}={value}" for name, value in sorted(safe_headers.items())
    )
    return f" Rate-limit response headers: {rendered}."


def _format_error(error: BaseException) -> str:
    if isinstance(error, CLIInputError):
        return str(error)
    if isinstance(error, ModelPricingConfigurationError):
        return f"Model pricing configuration error: {error}"
    if isinstance(error, EstimatedCostLimitError):
        return f"Estimated API cost limit exceeded: {error}"
    if isinstance(error, TesseractUnavailable):
        return f"OCR/Tesseract unavailable: {error}"
    if isinstance(error, IngestionTimeoutError):
        return f"Contract runtime deadline exceeded during ingestion: {error}"
    if isinstance(error, AssessmentIntegrityError):
        return f"Evidence assessment integrity failure: {error}"
    if isinstance(error, ReconciliationError):
        return f"Reconciliation or final schema validation failed: {error}"
    if isinstance(error, ExtractionRequestTooLarge):
        return f"Request-size or context safety limit: {error}"
    if isinstance(error, ExtractionError):
        if "OPENAI_API_KEY" in str(error):
            return (
                "OpenAI authentication configuration error: set OPENAI_API_KEY "
                "in the environment."
            )
        if any(
            phrase in str(error).casefold()
            for phrase in (
                "did not match the evidence schema",
                "did not contain structured candidate data",
                "did not contain a completion choice",
                "exceeded the configured output-token limit",
            )
        ):
            return f"Malformed or incomplete extraction response: {error}"
        status, code, details, classification, headers = _provider_error_details(error)
        if status == 401:
            return "OpenAI authentication failed (HTTP 401). Check OPENAI_API_KEY."
        if status == 429 and classification == "insufficient_quota":
            return (
                "OpenAI API credits/quota are insufficient. No automatic retry was "
                f"performed (HTTP 429).{_format_rate_limit_headers(headers)}"
            )
        if status == 429:
            rate_limit_messages = {
                "requests_per_minute_limit": "OpenAI requests-per-minute limit reached",
                "tokens_per_minute_limit": "OpenAI tokens-per-minute limit reached",
                "rate_limit": "OpenAI rate limit reached; limit type was not identified",
            }
            message = rate_limit_messages.get(
                classification,
                "OpenAI rate limit reached; limit type was not identified",
            )
            return (
                f"{message} (HTTP 429); no automatic retry was performed."
                f"{_format_rate_limit_headers(headers)}"
            )
        context_phrases = (
            "context_length",
            "maximum context",
            "context window",
            "request too large",
            "too many tokens",
            "max_tokens",
        )
        context_details = f"{details} {code}"
        if status in {400, 413} and any(
            phrase in context_details for phrase in context_phrases
        ):
            return f"Model context/request-size limit reached (HTTP {status})."
        if status is not None:
            return (
                f"OpenAI API request failed (HTTP {status}; "
                f"classification: {classification})."
            )
        if classification == "other_api_error":
            return (
                "OpenAI API request failed (HTTP status unavailable; "
                "classification: other_api_error)."
            )
        return f"Extraction failed: {error}"
    if isinstance(error, OSError):
        return f"Could not read the contract input: {error}"
    if isinstance(error, PipelineError):
        return str(error)
    return f"Unexpected processing failure ({type(error).__name__}): {error}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    started_at = time.monotonic()
    try:
        with redirect_stdout(sys.stderr):
            output = _process_contract(args.contract_folder, started_at)
            rendered = json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        elapsed = time.monotonic() - started_at
        if elapsed > MAX_RUNTIME_SECONDS:
            raise PipelineError(
                f"Processing exceeded the {MAX_RUNTIME_SECONDS:.0f}-second runtime "
                "limit; final JSON was not emitted."
            )
        print(f"Elapsed time: {elapsed:.3f}s", file=sys.stderr)
        sys.stdout.write(rendered + "\n")
        sys.stdout.flush()
        return 0
    except Exception as exc:
        print(f"analyse.py: error: {_format_error(exc)}", file=sys.stderr)
        print(
            f"Elapsed time: {time.monotonic() - started_at:.3f}s",
            file=sys.stderr,
        )
        return 2 if isinstance(exc, CLIInputError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
