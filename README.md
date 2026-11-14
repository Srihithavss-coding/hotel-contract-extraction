# Hotel Contract Analyser

A Python command-line tool that reads the PDF documents for one hotel contract and produces structured JSON for a booking system. The focus is on traceable extraction: a value is loaded only when its source can be checked. When documents conflict or leave a material decision unresolved, the tool asks the loader instead of guessing.

**Assignment brief:** [`ASSIGNMENT.pdf`](ASSIGNMENT.pdf)

## 1. What the tool does

The input is one folder for one hotel. The folder may contain one or two PDFs, such as a rate letter and a related email. The program reads all PDFs in that folder and produces one JSON object that follows the supplied schema in `schema/`.

The output covers:

- **Contract details:** hotel name, validity dates, currency and tax-inclusion status.
- **Room rates:** date/season period, room, meal plan, occupancy and price. A normal room rate is per room per night; an extra-bed rate is per additional bed per night.
- **Policies:** cancellation charges and advance-payment instalments.
- **Questions:** material choices the loader must resolve, with the affected field, reason, options and supporting sources.
- **Not loaded:** relevant terms that cannot safely be represented in the schema or lack enough detail, with a reason and source.

Every loaded value includes a source object with the original filename, a 1-based page number and an exact supporting quotation. The quotation is intended to help a reviewer find the evidence quickly; it is not a model-written summary.

The project follows the assignment's scope. Cancellation and advance payment are the supported policy types. Group rates for 10 or more rooms are out of scope and are not mixed into individual room rates. Other terms are included only when they can be safely represented; otherwise they are omitted or recorded in `not_loaded` when relevant.

## 2. Design and processing flow

The main design principle is **AI proposes; Python verifies and decides**.

1. **PDF ingestion — `ingest.py`**  
   The program discovers the PDFs in the supplied folder and reads every page. PyMuPDF extracts selectable text. Pages with little or no usable text are rendered and processed with local Tesseract OCR. Page records retain the filename, 1-based page number, text, extraction method and, for OCR, word coordinates and confidence information.

2. **Candidate extraction — `extract.py`**  
   Google Gemini receives the page evidence and proposes structured candidate facts, rates and policy terms. The response is parsed into Pydantic models. Each candidate carries its proposed source filename, page and quotation. Document text is treated as data, not as instructions to the model.

3. **Evidence assessment — `extract.py`**  
   Python independently checks that the cited file and page exist and that the quotation is present in that page's extracted text. Additional checks examine whether the evidence supports the candidate's meaning. Rate checks also consider OCR word confidence and table geometry, including nearby room, meal-plan and occupancy headings. Candidates are classified as verified, uncertain or invalid. Only verified candidates can be loaded automatically.

4. **Deterministic reconciliation — `reconcile.py`**  
   Ordinary Python rules apply the booking-system rules to verified candidates. This stage does not call the model. It handles rate identities, supported occupancy expansion, date periods, extra beds, policy schedules, deduplication and conflicts. It does not silently choose between two supported but contradictory statements.

5. **Final schema validation — `analyse.py` and `reconcile.py`**  
   The final object is validated against the supplied JSON schema. If a stage fails, the CLI exits with a non-zero status and does not emit partial JSON.

### How uncertainty is handled

- **Clear and supported:** load the value with its source.
- **Conflicting or materially ambiguous:** ask a question and keep the affected value out of the loaded fields. Questions provide 2–4 options and cite the statements that created the choice.
- **Relevant but not safely loadable:** put the term in `not_loaded` with a short reason and source.

For example, a cancellation deadline that differs between a rate letter and an email should become a loader question, not an AI-selected answer. A reference-only rack rate should not be treated as a contracted price. A zero-penalty cancellation period is not emitted as a penalty tier. An advance-payment balance is not inferred unless the source supports it.

## 3. Repository layout

```text
hotel-contract-extraction/
├── ASSIGNMENT.pdf             # Original assignment brief
├── contracts/                 # Supplied sample contract folders (c01-c05)
├── expected/                  # Reference JSON fixtures for evaluation only
├── out/                       # Local generated JSON and diagnostic logs
├── prompts/                   # Extraction prompt material
├── schema/                    # Supplied output JSON schema
├── tests/                     # Offline tests and opt-in live tests
├── analyse.py                 # One-folder command-line entry point
├── ingest.py                  # PDF text extraction and OCR ingestion
├── extract.py                 # Gemini extraction and evidence checks
├── reconcile.py               # Deterministic business rules and validation
├── evaluate.py                # Compare generated JSON with expected fixtures
├── requirements.txt
├── .env.example
└── README.md
```

`expected/` is for comparison only. The production pipeline reads the PDFs in the input folder; it must not read or copy the expected JSON fixtures to create its output. The sample folder names (`c01`–`c05`) are not extraction rules: the CLI accepts a folder path, and the same pipeline is used for each input. Hidden-contract generalization still needs separate evaluation.

## 4. Setup

The documented development environment is Windows with Python 3.13.9. Python 3.10 or newer is required by the project.

### Install Python dependencies

Run these commands in PowerShell from the repository folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

### Install OCR support

Scanned pages require Tesseract OCR to be installed separately from the Python packages. Make sure the English language data (`eng`) is installed and that Tesseract is available on `PATH`.

If Tesseract is not on `PATH`, set `TESSERACT_CMD` and, when needed, `TESSDATA_DIR` in `.env`. The `.env.example` file shows the expected variable names and Windows path format.

### Configure Gemini

Create a local environment file and edit it:

```powershell
Copy-Item .\.env.example .\.env
notepad .\.env
```

Set the following values in `.env`:

```dotenv
GEMINI_API_KEY=your_google_ai_studio_api_key
GEMINI_MODEL=gemini-3.8-flash
GEMINI_PRICING_MODEL=gemini-3.8-flash
GEMINI_INPUT_COST_PER_MILLION=your_current_input_price_usd_per_million_tokens
GEMINI_OUTPUT_COST_PER_MILLION=your_current_output_price_usd_per_million_tokens
```

Also set `TESSERACT_CMD` and `TESSDATA_DIR` if your local Tesseract installation needs explicit paths. `GEMINI_PRICING_MODEL` must match `GEMINI_MODEL`. Enter the current applicable USD input and output prices for the selected model; use zero only when the applicable price is actually zero. Do not copy an example price from an old README or assume that a ChatGPT subscription includes API usage.

Keep `.env` private. It contains the API key and must not be committed. Commit `.env.example`, not `.env`.

## 5. Run one hotel folder

The CLI takes exactly one required argument: the folder containing that hotel's PDF documents. To run a sample and save JSON separately from diagnostics:

```powershell
New-Item -ItemType Directory -Force .\out | Out-Null
python .\analyse.py ".\contracts\c01" 1> ".\out\c01.json" 2> ".\out\c01.log"
```

Replace `c01` with `c02`, `c03`, `c04` or `c05` to run the other samples. Run each folder separately.

- Successful standard output contains the final JSON.
- Progress messages, token usage, cost estimates, warnings and errors are written to standard error.
- A non-zero exit code means the run did not produce a complete JSON result. Check the matching `.log` file.
- Do not evaluate a zero-byte JSON file left by a failed command; remove it and keep the diagnostic log for troubleshooting.

## 6. Runtime and cost controls

The assignment target is **no more than 60 seconds and USD 0.15 per contract**. The CLI tracks a 60-second end-to-end budget covering ingestion, extraction, reconciliation and final validation. The extraction request is also bounded by configured model pricing and a preflight cost estimate.

The implementation includes these guardrails:

- a bounded PDF-ingestion worker;
- a 30-page input limit;
- a 120,000-character request limit;
- an 8,192-token maximum completion setting;
- explicit input/output token prices that must match the configured model;
- a USD 0.15 preflight cost ceiling and a post-response estimated-cost check when usage is available;
- automatic SDK retries disabled, so hidden retry attempts do not inflate runtime or usage.

Tesseract OCR runs locally rather than through a paid OCR API. The model cost is an estimate calculated from configured prices and provider-reported token usage; it is not a provider billing guarantee. A provider-side delay or HTTP error can still prevent a run from finishing within the target.

## 7. Evaluation

After successful live runs have generated the five JSON files, compare them with the supplied fixtures:

```powershell
python .\evaluate.py .\out .\expected
```

The evaluator checks more than JSON syntax. It compares rate identities and prices, contract details, cancellation and advance-payment terms, source filename/page/quote presence, question topics and expected `not_loaded` items. A schema-valid JSON file can still contain missing or incorrect business values.

Do not copy files from `expected/` into `out/`. The `out/` files must be generated by `analyse.py` from the contract PDFs.

### Expected shape of the supplied samples

These are the expected fixture counts and scenarios, not claims that the live pipeline has completed all five comparisons.

| Sample | Main case | Expected rate rows | Expected questions | Expected `not_loaded` |
|---|---|---:|---|---:|
| c01 — Saffron Courtyard | Supported single/double expansion; incomplete festive surcharge | 24 | 0 | 1 |
| c02 — Coral Sands | Scanned rate table; individual and group-rate wording | 16 | 0 | 1 |
| c03 — Misty Ridge | Email supplies extra-bed rates; cancellation conflict | 54 | 1 (cancellation) | 0 |
| c04 — Lotus Pavilion | Overlapping periods and conflicting tax wording | 8 | 2 (tax and rates) | 0 |
| c05 — Banyan Grove | Long contract; reference-only rack rates | 24 | 0 | 1 |

## 8. Verification results and current status

The results below distinguish offline tests, a focused live extraction test and a complete contract-level run. They are not interchangeable.

| Check | Result |
|---|---|
| Offline test suite | **180 passed, 6 skipped** in 251.24 seconds. Live flags were disabled. |
| Focused live c02 rate-extraction test | **Passed twice**. Recorded runtimes: 145.52 seconds and 269.80 seconds; latest run: 269.80 seconds. This is an extraction-stage candidate comparison, not a complete c02 JSON run. |
| Full c02 CLI attempt | Not completed. The latest attempt ingested 2 pages, then received Gemini HTTP 504 after 61.281 seconds. `out/c02.json` was empty. |
| Full c01–c05 live golden comparison | Not completed successfully. A complete five-contract set of generated JSON files and final evaluator results is not available. |
| Three live runs per contract | Not completed. Deterministic reconciliation tests do not establish repeatability across separate live Gemini calls. |
| Hidden/unseen contracts | Not tested. |

### Per-sample runtime, token and cost record

The assignment asks for runtime, token usage and cost for each sample. The table intentionally records unavailable measurements rather than estimating or inventing them. A failed provider request is not an accuracy score.

| Sample | End-to-end runtime | Input / output tokens | Estimated API cost | Final evaluator result |
|---|---|---|---|---|
| c01 | No completed run recorded | Not recorded | Not available | Not evaluated |
| c02 | Latest CLI attempt: 61.281 s, blocked by HTTP 504; no final JSON. Focused extraction test: 269.80 s, not end-to-end. | Not available for the failed CLI run; focused-test usage not recorded here | Not available | Not evaluated |
| c03 | No completed run recorded | Not recorded | Not available | Not evaluated |
| c04 | No completed run recorded | Not recorded | Not available | Not evaluated |
| c05 | No completed run recorded | Not recorded | Not available | Not evaluated |

The focused c02 test demonstrates that the configured Gemini extraction path can complete the specific rate-candidate comparison. It does not demonstrate that the full pipeline meets the 60-second target, that the final JSON matches the fixture, or that all five contracts pass. The latest full c02 CLI attempt was blocked by a provider HTTP 504 after ingestion. This is an infrastructure failure, not a completed accuracy comparison.

## 9. Run the tests

### Offline tests

The offline suite does not make live Gemini requests. Run it with both live flags disabled:

```powershell
$env:RUN_LIVE_EXTRACTION_C02 = "0"
$env:RUN_LIVE_RECONCILIATION = "0"
python -m pytest -q
```

### Focused live c02 extraction test

This opt-in test makes a real Gemini API request and requires a valid `.env`, available provider quota and configured pricing:

```powershell
$env:RUN_LIVE_EXTRACTION_C02 = "1"
$env:GEMINI_MODEL = "gemini-3.8-flash"
$env:RUN_LIVE_RECONCILIATION = "0"
python -m pytest -q .\tests\test_extract_live.py::test_live_c02_rate_extraction_matches_expected_candidates
```

Live tests can fail before producing an extraction result because of authentication, quota, rate limits, provider availability or gateway timeouts. Such a blocked request should be reported as a blocked run, not as a successful or failed accuracy comparison. Avoid repeatedly retrying the same provider error.

## 10. Limitations

- OCR can misread digits, punctuation or table columns. Word confidence and geometry checks reduce risk but cannot guarantee perfect recognition.
- Gemini can omit or misinterpret a fact. Evidence validation can reject unsupported candidates, but it cannot recover facts the model did not propose.
- Evidence checks use deterministic validation rules and known patterns. Unfamiliar wording or document layouts may be marked uncertain or missed.
- The supplied schema supports only a limited set of booking fields and policy types. Some commercial terms therefore belong in `not_loaded` or are outside the assignment scope.
- Provider availability and rate limits affect live execution. A valid API configuration does not guarantee every request will complete within the runtime budget.
- Full live golden evaluation, three-run consistency across all five samples and testing on unseen contracts remain outstanding.

## 11. AI-tool disclosure

Google Gemini (`gemini-3.8-flash` in the recorded live tests) is the model provider used by the extraction pipeline. GitHub Copilot in VS Code and ChatGPT assisted with code inspection, debugging, test reasoning and documentation. Suggestions were reviewed against the implementation and local tests. Live-test statements in this README refer only to commands actually run; AI assistance is not presented as independent validation.

## 12. Submission summary

The repository contains a one-folder command-line pipeline, local PDF/OCR ingestion, structured Gemini candidate extraction, independent evidence checks, deterministic reconciliation, source-backed output fields and schema validation. The offline suite and the focused live c02 extraction test passed as recorded above. Complete five-contract live evaluation and three-run repeatability were not established because the full c02 command was blocked by a Gemini HTTP 504 and the remaining end-to-end runs were not completed.
