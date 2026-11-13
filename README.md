# Hotel Contract Analyser

This project reads one hotel-contract folder, extracts source-backed evidence
from its PDFs, validates each candidate against the cited page, reconciles only
verified candidates, and emits JSON matching the unchanged company schema in
`schema/output.schema.json`.

The five sample folders are c01 through c05. `ASSIGNMENT.pdf` is the source of
the assignment requirements. `expected/` contains comparison fixtures; it is
not used as input to production extraction or reconciliation.

## Requirements and installation

- Python 3.9 or newer.
- Tesseract OCR installed and available on `PATH` for scanned PDFs. If it is
  installed elsewhere on Windows, set `TESSERACT_CMD` and, if needed,
  `TESSDATA_DIR` in `.env`.
- An OpenAI API key and a model that supports the structured extraction request.
  No API key or live API call is needed for offline tests.

In PowerShell, from the project root:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
Copy-Item .\.env.example .\.env
notepad .\.env
```

Set `OPENAI_API_KEY`, `OPENAI_MODEL`, and explicit pricing fields in `.env`.
`OPENAI_PRICING_MODEL` must exactly match `OPENAI_MODEL`. Set
`OPENAI_INPUT_COST_PER_MILLION` and `OPENAI_OUTPUT_COST_PER_MILLION` to the
current provider prices in USD per million tokens. There are deliberately no
implicit reference prices: an absent, malformed, or mismatched price
configuration fails before a model request.

The CLI applies a 60-second per-contract runtime budget, limits each extraction
request to 30 pages and 120,000 prompt/evidence characters, and uses an 8,192
token maximum completion. A conservative preflight estimate includes a
UTF-8-byte upper bound for input tokens, the structured response schema, a
request-wrapper allowance, and the maximum output tokens. Requests estimated
above $0.15 are rejected before submission. Where provider token usage is
returned, the application reports a separate cost estimate based on that usage
and the configured rates. These are estimates, not provider billing guarantees.

## Run the CLI

Pass one contract directory as the required positional argument. Quote paths
that contain spaces:

```powershell
python .\analyse.py ".\contracts\c01"
```

On success, stdout contains only the complete, schema-validated JSON object.
Progress, warnings, runtime, token usage, cost estimates, and errors go to
stderr. A failure exits non-zero and does not emit partial JSON.

To save the two streams separately:

```powershell
New-Item -ItemType Directory -Force .\out | Out-Null
python .\analyse.py ".\contracts\c01" 1> ".\out\c01.json" 2> ".\out\c01.log"
```

Run the evaluator after generating the output files:

```powershell
python .\evaluate.py .\out .\expected
```

Do not treat expected fixtures as extracted outputs. This repository does not
include fabricated `out/` results.

## Offline tests

Both opt-in live flags must remain disabled for the offline suite:

```powershell
$env:RUN_LIVE_EXTRACTION_C02 = "0"
$env:RUN_LIVE_RECONCILIATION = "0"
python -m pytest -v
```

The c02 live extraction comparison is opt-in:

```powershell
$env:RUN_LIVE_EXTRACTION_C02 = "1"
python -m pytest -v .\tests\test_extract_live.py
Remove-Item Env:RUN_LIVE_EXTRACTION_C02
```

The c02 report separates expected table rates from other commercial rates and
includes each rate's candidate ID, evidence assessment and issues, and source
file/page. Fixture-value matches are reported separately and do not imply that
the evidence is verified. Stage timings use milliseconds; rendering and OCR are
subtotals of PDF ingestion, and the OpenAI API duration includes the SDK's
structured response parsing.

The five c01–c05 live extraction/reconciliation-to-golden comparisons are
opt-in:

```powershell
$env:RUN_LIVE_RECONCILIATION = "1"
python -m pytest -v .\tests\test_reconcile.py
Remove-Item Env:RUN_LIVE_RECONCILIATION
```

Both live commands make paid API calls and require valid credentials, sufficient
credits, matching model pricing, and a model that supports the configured
structured response. A quota/insufficient-credit result is blocked, not a
passed golden comparison. Live comparisons pass only when extraction and
reconciliation complete and the output is actually compared with its expected
fixture.

The CLI's OpenAI error report distinguishes insufficient credits/quota,
requests-per-minute limits, tokens-per-minute limits, and other API errors.
For HTTP 429 responses it reports present rate-limit response headers,
including project-specific `x-ratelimit-*` headers, or states that none of
the tracked headers were present. It does not log request or authorization
headers. The SDK client has automatic retries disabled.

On a live golden mismatch, the comparison reports every differing JSON path,
including expected and actual values and whether a property is missing,
additional, or unequal. Saving the reconciled actual JSON is separately
opt-in; it is never enabled by the offline suite. Set
`SAVE_LIVE_FAILURE_OUTPUT=1` to save failed outputs outside `expected/`.
The default directory is `.\artifacts\live-failures`; optionally set
`LIVE_FAILURE_OUTPUT_DIR` to another directory outside `expected/`. Files use
unique names and do not overwrite existing outputs. For example:

```powershell
$env:RUN_LIVE_RECONCILIATION = "1"
$env:SAVE_LIVE_FAILURE_OUTPUT = "1"
$env:LIVE_FAILURE_OUTPUT_DIR = ".\artifacts\live-failures"
python -m pytest -v .\tests\test_reconcile.py
Remove-Item Env:RUN_LIVE_RECONCILIATION, Env:SAVE_LIVE_FAILURE_OUTPUT, Env:LIVE_FAILURE_OUTPUT_DIR
```

## Evidence and reconciliation

`ingest.ingest_folder(...)` returns a common page structure for selectable-text
and scanned pages: original text, filename, 1-based page number, reconstructed
lines, and (for OCR) word positions and confidence. The CLI runs ingestion in a
bounded worker and withholds all pages unless the complete folder succeeds;
OCR failures and deadline exhaustion are errors rather than silently skipped
pages.

`extract.extract_folder(...)` returns Pydantic candidate records and linked
evidence assessments (`verified`, `uncertain`, or `invalid`). Candidates retain
their source filename, page, and exact quote. The candidate content is treated
as untrusted data, and conflicting candidates are preserved. Call
`run.verified_candidates()` to access verified candidates; reconciliation
continues to use that verified-only gate for loaded values. A separate
assessment-aware path can ask a material clarification about a limited,
source-supported uncertainty without loading the uncertain value. Child-policy
terms that are relevant but have no company-schema field are retained in
`not_loaded` with their citations.

`reconcile.reconcile_run(run)` produces deterministic schema-shaped output,
sorts conflicts and questions canonically, and validates against the unchanged
company schema. It does not call the model or print diagnostics.

## Current acceptance limitations and results

Offline tests can verify deterministic reconciliation for identical candidate
inputs. They do not prove that repeated live LLM extractions are deterministic.
The c01–c05 runtime, token usage, configured-price estimate, and golden
evaluation results remain pending until successful live runs have actually
produced them. Cost values are estimates and do not establish actual provider
billing. API-credit availability and hidden-contract generalization have not
been verified in this offline run.

| Contract | Runtime | Input tokens | Output tokens | Estimated API cost | Evaluation result |
| --- | --- | --- | --- | --- | --- |
| c01 | Pending live run | Pending live run | Pending live run | Pending live run | Pending live comparison |
| c02 | Pending live run | Pending live run | Pending live run | Pending live run | Pending live comparison |
| c03 | Pending live run | Pending live run | Pending live run | Pending live run | Pending live comparison |
| c04 | Pending live run | Pending live run | Pending live run | Pending live run | Pending live comparison |
| c05 | Pending live run | Pending live run | Pending live run | Pending live run | Pending live comparison |

## AI-tool disclosure

An AI assistant using Copilot SDK in VS Code assisted with code inspection,
implementation, tests, and documentation during development. It did not make
live extraction calls, generate the sample outputs, or supply live measurements
for this results table. Offline tests and `evaluate.py` are project commands;
the evaluator only compares output files that have actually been generated.
