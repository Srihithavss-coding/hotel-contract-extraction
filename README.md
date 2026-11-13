# Hotel Contract Analyser

This command-line project reads the PDFs for one hotel contract and prepares JSON for a booking system. It aims to load only details supported by the documents. When documents disagree or leave a material detail unclear, it records a question for the loader instead of guessing.

## How it works

```text
PDFs -> read text / OCR -> AI suggests values and sources
     -> Python checks the evidence -> rules build the JSON -> schema validation
```

The AI suggests possible contract details. Python checks whether each suggestion is supported by the cited file, page and quotation, then applies the business rules. Only verified values are loaded. This is the main design choice: a plausible answer is not enough unless the source supports it.

## What the output contains

The output follows the supplied schema in `schema/output.schema.json`:

- `contract`: hotel name, validity dates, currency and tax status;
- `rates`: period, room, meal plan, occupancy and price;
- `policies`: cancellation charges and advance-payment instalments;
- `questions`: unresolved decisions, with options and cited sources;
- `not_loaded`: relevant terms that cannot safely be loaded, with a reason and source.

Each loaded contract value, rate and policy includes the source filename, page number and supporting words from the document.

The samples also exercise scanned pages, prices in a separate email, single/double price expansion, extra-bed rates, conflicting policies, overlapping date periods, conflicting tax statements, group rates and rack rates.

## Project files

| File or folder | Purpose |
| --- | --- |
| `analyse.py` | Runs the pipeline for one contract folder. |
| `ingest.py` | Reads PDF pages and uses Tesseract for scanned pages. |
| `extract.py` | Requests structured candidate details and source quotations. |
| `reconcile.py` | Checks evidence, applies rules and builds final JSON. |
| `schema/` | Supplied output schema. |
| `contracts/` | Five sample input folders. |
| `expected/` | Reference JSON files for comparison, not production input. |
| `evaluate.py` | Compares generated JSON with the reference files. |
| `tests/` | Offline tests and opt-in live evaluation tests. |

## Setup

Development was carried out on Windows with Python 3.13.9. Use Python 3.10 or newer. Tesseract OCR must be installed for scanned pages, with English language data available. If needed, set `TESSERACT_CMD` and `TESSDATA_DIR` in `.env`.

In PowerShell, from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
Copy-Item .\.env.example .\.env
notepad .\.env
```

Set `GEMINI_API_KEY` from Google AI Studio, `GEMINI_MODEL`,
`GEMINI_PRICING_MODEL`, `GEMINI_INPUT_COST_PER_MILLION` and
`GEMINI_OUTPUT_COST_PER_MILLION` in `.env`. The pricing model must exactly
match the selected Gemini model, and the two prices must be its current USD
prices per million tokens. Keep `.env` private; only `.env.example` belongs in
Git. The SDK request is configured for one attempt; rate-limit and network
failures are reported rather than automatically retried.

The Gemini integration has not been validated with a live Google AI Studio
request. Passing offline tests does not establish live extraction accuracy.

The program checks its estimated request cost before calling the model and stops if the estimate exceeds USD 0.15. Reported cost is an estimate, not a provider billing guarantee. The CLI also enforces the 60-second per-contract limit.

## Run a contract

Create the output folder, then run one contract at a time. JSON is written to the `.json` file; progress and diagnostics go to the `.log` file:

```powershell
New-Item -ItemType Directory -Force .\out | Out-Null
python .\analyse.py .\contracts\c01 1> .\out\c01.json 2> .\out\c01.log
```

Repeat for `c02` through `c05`, changing the input folder and output filename. Only keep a JSON file when the command finishes successfully.

## Evaluate the five samples

After all five real output files have been generated, run:

```powershell
python .\evaluate.py .\out .\expected
```

The evaluator checks rates, contract details, policies, sources, questions and `not_loaded` items. Schema-valid JSON can still contain wrong or missing values. Do not copy files from `expected/` into `out/`; the output files must come from `analyse.py`.

## Tests

The offline test suite does not make paid model calls. Run it with both live flags disabled:

```powershell
$env:RUN_LIVE_EXTRACTION_C02 = "0"
$env:RUN_LIVE_RECONCILIATION = "0"
python -m pytest -v
```

### Latest local verification

With both live flags set to `0`, `python -m pytest -q` completed successfully:
**173 passed, 6 skipped in 45.09 seconds**. The opt-in live extraction and
live reconciliation tests were not run. This offline result does not establish
live model accuracy, c01–c05 evaluation scores, or consistency across three
live runs. The previously reported c02 16/16 result remains a separate,
focused extraction-stage test, not part of this offline result.

Live tests are separate and make paid API requests. They need working credentials, available quota and a compatible model. If the provider returns a quota or rate-limit error, the test is blocked; it is not evidence that extraction accuracy passed or failed. Avoid repeatedly retrying after the same provider error.

## Validation status

| Check | Current evidence |
| --- | --- |
| Offline tests | See **Latest local verification** above. |
| Focused c02 OCR/rate extraction | One recorded run matched **16/16** expected table-rate candidates and verified those 16. This was an extraction-stage check, not a full final-JSON comparison; another attempt did not reproduce the same result consistently. |
| Full c01-c05 comparison | Not completed successfully. A c01 mismatch was reported; other attempts were interrupted by provider quota/rate limits. |
| Three runs per contract | Not completed. Deterministic reconciliation for identical candidates does not prove repeatability across separate live model calls. |
| Generated `out/` files | `out/c01.json` through `out/c05.json` are not present in this repository snapshot. They are required by the assignment and must be generated by successful runs. |
| Hidden contracts | Not tested. |

The assignment asks for a time, token and cost table for every sample. Those figures have not been established for the five final outputs, so they are not invented here:

| Contract | Runtime | Input/output tokens | Estimated cost | Final `evaluate.py` result |
| --- | --- | --- | --- | --- |
| c01 | Not verified | Not verified | Not verified | No output file |
| c02 | Not verified for final JSON | Not verified for final JSON | Not verified for final JSON | No output file |
| c03 | Not verified | Not verified | Not verified | No output file |
| c04 | Not verified | Not verified | Not verified | No output file |
| c05 | Not verified | Not verified | Not verified | No output file |

`evaluate.py` was checked against the current repository snapshot, but it found no `out/c01.json` through `out/c05.json`. Its missing-file report is not an accuracy score. Add the real evaluator output after successful runs generate those files.

## Limitations and next steps

OCR can misread digits or table columns. A model can omit or misinterpret a detail. Evidence checks can reject unsupported suggestions, but they cannot recover every missed value. Unusual layouts or wording may also fall outside the current extraction and verification rules.

With another week, I would complete repeatable live runs on all five samples, review each evaluator miss, improve only the rules responsible for those misses, and test the same workflow on additional contracts. I would prioritise extraction accuracy and repeatability over adding features.

## AI-tool disclosure

An AI assistant using Copilot SDK in VS Code assisted with code inspection, implementation, tests and documentation. The live extraction results above came from runs of the project tests. `evaluate.py` compares generated output files; it does not create those files.
