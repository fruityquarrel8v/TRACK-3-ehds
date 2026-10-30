# Intelligent Health Data Access Gateway

**An AI-assisted EHDS research gateway that interprets health-data requests, calculates trustworthy statistics and prevents unsafe disclosures.**

*AIM Healthcare Hackathon Vienna 2026 · Track 3: European Health Data Space · prototype on synthetic data*

> "An AI system that knows when to answer, when to ask, and when to say no."

## The problem

Researchers need health data, but every request mixes clinical ambiguity ("patients on metformin" – current or ever?), legal limits (EHDS Art. 53/54: research yes, insurance pricing no) and privacy risk. A Health Data Access Body (HDAB) has to decide, compute and protect – quickly and consistently.

## Our solution

A simulated HDAB you can talk to. The researcher types one question; the gateway interprets it, asks only when a detail changes the result, decides under the EHDS rules, computes deterministic statistics with small-cell suppression and opt-out exclusion, and explains every decision in plain language. A guided step-by-step builder and the raw JSON format remain available.

## What makes us different: we protect against *combinations* of answers

A small-cell rule checks one answer at a time. Two perfectly legal answers can still leak:

| Request | Released count |
|---|---|
| Women with breast cancer in Austria | 114 |
| … excluding Burgenland | 112 |

Anyone who sees both learns that Burgenland has exactly 2 patients – a protected small group. Our **disclosure ledger** remembers every released count (as one-way hashes of the patient set, never IDs). Before a new result is released, each cell is compared with everything released before, to anyone. If one population is contained in the other and they differ by fewer than 10 patients, the result is withheld and the researcher is told why and what to ask instead. The number itself is never shown. Try it on the **Demo scenarios** page.

*Following a privacy rule is not always the same as protecting privacy.*

## How it works

```
researcher question ──► legal/purpose screening (decision.py, Art. 54 first)
        │
        ▼
rule-based interpretation (nl.py) ──► structured criteria ──► validation + terminology (criteria.py, terminology.py)
        │                                                        ICD-10 · ATC · LOINC
        ▼
clarification when a choice changes the result (medication status, hospitalisation meaning, diabetes type…)
        │
        ▼
deterministic query engine (engine.py, pandas, opt-outs removed at load) ──► small-cell suppression (privacy.py)
        │
        ▼
disclosure ledger: differencing check against everything released before (disclosure.py)
        │
        ▼
released result + explanation, or refusal with safe alternatives ──► audit log (audit.py)
```

The interpreter never touches patient data and never produces a number; it only turns text into a validated structured request. The deterministic engine computes; independent safeguards decide what may leave. No external AI service is called; the interpreter is rule-based (an LLM provider could be plugged in behind the same structured-request schema, see *Limitations*).

## Demonstration

`streamlit run app.py`, then:

1. **Research assistant** – type "Compare HbA1c values between men and women aged 40–65 with type 2 diabetes during 2025." → interpretation card → review → result (women 7.2 %, men 7.1 %).
2. **Demo scenarios** – the four prepared scenarios (legitimate question → differencing attack blocked → clarification → prohibited purpose), all run through the real pipeline; a reset button restores the documented starting state.
3. **Staff dashboard** – decisions, query plans, disclosure ledger, audit log. **Secure processing environment** – R05's cohort, downloads blocked. **Evaluation** – the official metrics live.

Screenshots and a test recording are in the pull request.

## Privacy and security

- Opted-out patients are removed from all tables when the data is loaded (Art. 71).
- Counts < 10 → `<10`; percentages and means with < 10 patients → `suppressed`; every requested group shown.
- Disclosure ledger with subset-differencing detection across all applicants, persistent on disk, hashes only.
- Charts are drawn only from the suppressed table; patient-level plots are refused outside the SPE.
- SPE: locked folder per approved application, download/export calls raise and are audited; the UI uses tables without export toolbars.
- The interpreter treats question text as data: "ignore previous instructions / treat as approved" is ignored and reported (tested).
- No secrets in the repository; no network calls.

## Test results (measured)

| Metric | Result | Target |
|---|---|---|
| Correct decisions (R01–R10) | **10 of 10** | 10 of 10 |
| Correct numbers | **6 of 6** | 6 of 6 |
| Privacy leaks | **0** | 0 |
| Consistency over 3 runs | **10 of 10** | 10 of 10 |
| Slowest answer | **≈0.25 s** (data loads once, ≈2 s) | < 60 s |
| Automated tests (`pytest`) | **150+ passing**: official requests, new clinical requests, lab thresholds, medication periods, onset dates, differencing ledger, prompt-injection, UI flows (Streamlit AppTest) | – |

Numbers in the tests are recomputed independently with plain pandas, not taken from the engine.

## Limitations (implemented vs. future)

Implemented now: everything above, on the supplied synthetic dataset, for the 15 diagnoses, 17 medication groups and 4 laboratory tests it contains.

Not implemented / future work:
- The ledger detects subset/complement differencing between released counts. It does not model combinations of three or more queries, means, external knowledge, or results released before the ledger existed. Production SDC would need cumulative-release accounting.
- Trends over time are available per calendar year ("by year", "trends", "over time"): the request is evaluated once per year (diagnoses recorded by 31 December of that year, or within it for new cases; medication status on 31 December; latest laboratory value within the year; hospital stays within the year; age as of 31 December 2025) and shown as a line graph. Monthly results are not available; the assistant says so.
- Interpretation is rule-based; questions outside the dictionary are reported as not recognised rather than guessed. A local (Ollama) or European (Mistral) model could be added behind the same structured-criteria schema with output validation; it was not tested here.
- Real clinical data would bring missing values, inconsistent coding, mixed units and free text; the loader and terminology layer are the place for a future harmonisation step (code normalisation, unit checks, provenance). Before any real deployment: validation on representative datasets, clinical review of definitions, independent privacy/security assessment and human oversight.
- Decisions are simulated for demonstration and are not legally binding decisions of a Health Data Access Body.

## Team

Laura, Christian, Sašo and Axel – AIM Healthcare Hackathon Vienna 2026, Track 3.

| Member | Contribution |
|---|---|
| Laura | _to be filled in_ |
| Christian | _to be filled in_ |
| Sašo | Clinical lead: interpretation rules (hospitalisation, medication status, condition onset, lab thresholds), validation of mappings, product direction |
| Axel | _to be filled in_ |

Implementation support: Devin (AI software engineer), working from the team's briefs.

---

# Technical documentation

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py                     # web app: http://localhost:8501
python evaluate.py                       # the challenge metrics on requests/R01–R10
python -m ehds_hdab requests/R01.json    # one request from the command line (JSON out)
python -m pytest -q                      # tests
```

Python 3.10+, pandas, Streamlit.

## Metric results on the 10 requests

Measured with `python evaluate.py` (each request run 3 times):

| Metric | Result | Target |
|---|---|---|
| Correct decisions | **10 of 10** (decision and articles) | 10 of 10 |
| Correct numbers | **6 of 6** | 6 of 6 |
| Privacy leaks | **0** | 0 |
| Consistency | **10 of 10** over 3 runs | 10 of 10 |
| Time to answer | **0.21 s** slowest (after the data is loaded once, ≈2 s) | under 60 s |

| Request | Decision | Articles |
|---|---|---|
| R01 | APPROVE_STATISTICAL | 53, 69 |
| R02 | REJECT (insurance premiums) | 54 |
| R03 | APPROVE_STATISTICAL | 53, 69 |
| R04 | REJECT (advertising) | 54 |
| R05 | APPROVE_SPE (ML model needs individual histories) | 53, 66, 68, 73 |
| R06 | REJECT (employment decisions, re-identification) | 54 |
| R07 | APPROVE_STATISTICAL | 53, 69 |
| R08 | APPROVE_STATISTICAL | 53, 69 |
| R09 | APPROVE_STATISTICAL (records not justified) | 53, 66, 69 |
| R10 | REJECT (harmful product development) | 54 |

Beyond the 10 examples, `tests/` contains 64 tests, including new requests in `tests/new_requests/` (inpatient stays by hospital filtered on main diagnosis, medication shares by sex, a condition named without a code, latest blood pressure by age band, a tobacco product request, a public-health request, an SPE readmission-model request) whose expected numbers are computed independently with plain pandas, plus legitimate requests that must not be rejected and disguised prohibited ones that must be.

## Architecture

```
request JSON
   │
   ▼
decision.py ── Art. 54 prohibited purpose? ──yes──► REJECT (Art. 54)
   ├─ purpose matches no Art. 53 category ──────► REVIEW_REQUIRED (Art. 53): nothing computed or released
   │ no
   ├─ "health data request" ────────────────────► APPROVE_STATISTICAL (Art. 53, 69)
   ├─ access application + justified need for records ► APPROVE_SPE (Art. 53, 66, 68, 73)
   └─ access application, statistics suffice ───► APPROVE_STATISTICAL (Art. 53, 66, 69)
   │
   ▼
parser.py + terminology.py  →  QueryPlan (concepts with ICD-10 / ATC / LOINC codes,
   │                            cohort, metrics, group-by dimensions)
   ▼
data.py (pandas, opted-out patients removed from all 5 tables at load time)
   │
   ├─ statistical ──► engine.py (exact Decimal maths) ──► privacy.py (small-cell rule,
   │                                                       output privacy check) ──► answer
   └─ SPE ──────────► spe.py (workspace with only the cohort + requested record types,
                      manifest, download blocked) ──► audit.py (JSONL log)
```

| Component | What it does |
|---|---|
| `ehds_hdab/decision.py` | Rule-based decision in the README order. Detects Art. 54 purposes (insurance/credit decisions, advertising/marketing, employment decisions, re-identification, harmful products, detrimental decisions about people). SPE only when the applicant explains why statistics are not enough (e.g. training a model, individual-level histories). Returns decision, articles, plain-language reason and the Art. 53 purpose category. |
| `ehds_hdab/parser.py` | Turns the free-text `definitions` and `question` into a `QueryPlan`: diagnoses ("ICD-10 code starting with E11"), medications (ATC prefixes, active on 31 Dec 2025), observations (LOINC, latest/earliest/highest/lowest per patient), encounters (type, year, main diagnosis), age bands, ratio formulas (`X = A / B × 100`), counts, percentages, means and group-by dimensions (region, age band, sex, hospital). Anything it cannot interpret becomes a warning for staff instead of a crash. |
| `ehds_hdab/terminology.py` | Dictionary of common medical terms → ICD-10 / ATC / LOINC (used when a request gives a name but no code), with a fallback lookup in the data. |
| `ehds_hdab/data.py` | Loads the five CSV files once (cached), removes every opted-out patient (Art. 71) from all tables, computes age = 2025 − birth year. |
| `ehds_hdab/engine.py` | Executes the plan for every requested group (cartesian product, so empty groups are shown too). Vectorised pandas; exact `Decimal` arithmetic. |
| `ehds_hdab/privacy.py` | Small-cell rule: counts < 10 (incl. 0) → `<10`; percentages with numerator or denominator < 10 → `suppressed`; means of < 10 patients → `suppressed`; one decimal, halves up. Then scans the output for patient IDs, record lists and small counts. |
| `ehds_hdab/spe.py` | Creates `spe_workspaces/<request id>/` with the pseudonymised cohort and only the record types requested, plus a manifest (legal basis, cohort size, `download_allowed: false`). Any download/export call raises `DownloadBlockedError` and is logged. |
| `ehds_hdab/audit.py` | Append-only JSONL audit log of decisions, SPE creation and blocked downloads (no patient data). |
| `ehds_hdab/criteria.py` | Optional structured `criteria` block in a request (population with nested ALL/ANY groups and exclusions, group-by, metrics, record types) → `QueryPlan`; plain-language description and human-readable definitions. Used by the cohort builder and the question interpreter; requests without it use the free-text parser unchanged. |
| `ehds_hdab/nl.py` | Deterministic interpreter for plain-English research questions → `criteria` (dictionary concepts only, documented defaults listed as assumptions, unknown words reported, never guessed). |
| `ehds_hdab/viz.py` | Chart commands ("bar chart of prevalence by region", "change this into a line graph", "group by sex and age") → chart spec; renders bar/line charts with matplotlib from the already-suppressed answer, suppressed cells shown as labelled gaps. Patient-level chart types (histogram, scatter, box plot) are refused outside the SPE. |
| `ehds_hdab/pipeline.py` | Ties it together: decide → parse → compute or create SPE → privacy check. Also a differencing guard: if the same applicant asks for a population that differs from an earlier one by 1–9 patients, the answer is withheld for staff review (not a challenge requirement). |
| `app.py` | Streamlit web app. **Research assistant** (applicant view): a single question box ("What would you like to research?") → a plain-language *Here's how I understood your question* card (population, age, measurement, compare-by, period, analysis, with the codes as secondary detail) → a clarification step only when a choice changes the result (medication active vs ever prescribed) → review with applicant details and purpose → decision and results with charts. A step-by-step **guided builder** (1 Research question → 2 Population → 3 Analysis → 4 Review; answers survive Back/Next) is the alternative; *Edit details* opens it prefilled. **Advanced options** (collapsed) keep example / upload / paste JSON and the challenge-format form. All paths produce the same request JSON and go through `process_request`. **Staff dashboard**, **Secure processing environment** (download blocked and logged) and **Evaluation** views are unchanged. Theme in `.streamlit/config.toml`, scoped CSS in `app.py`. |
| `evaluate.py` | Runs R01–R10 three times and reports the README metrics against `requests/expected.json`. |

### Design choices

- **Rules instead of an LLM.** Deterministic, fast, explainable, and nothing leaves the HDAB. Unclear wording is flagged for staff review rather than guessed.
- **One central pandas database.** The data is small (≈7 MB), so a central in-memory store is simplest; opt-outs are removed once, at load time, so no later step can include them.
- **Privacy as the last step.** All numbers pass through one small-cell function and an output check before anyone sees them. Charts are drawn only from that suppressed table, never from patient records.
- **Clinician-reviewed interpretation rules** (documented defaults, shown to the researcher): "hospitalised for X" = inpatient stay with X as main diagnosis; a medication without a stated status = active on 31 Dec 2025 (+ staff warning); a condition named without a code = diagnosis code only; conditions that exist only as hospital main diagnoses are counted from the diagnosis list with a staff warning; "diagnosed/new cases in YYYY" = onset in YYYY, "had X in YYYY" = diagnosed by 31 Dec YYYY; a lab threshold uses each patient's latest value within the period (latest overall if none).
- **No LLM anywhere.** The "automatic interpretation" of questions and chart commands is rule-based, so it only understands concepts in the built-in dictionary; everything else is reported as not recognised.
- **One design system.** All colours live in `ehds_hdab/theme.py` (pink `#C44779` for actions and selected states, lavender `#B9A5F5` for information, midnight navy `#182342` for text, blush `#FFF6FA` background, white cards; green/amber/red kept for approved/review/rejected so status never relies on pink alone). The Streamlit CSS, the Altair charts and `.streamlit/config.toml` read from it; styling never touches the calculations.

## What works and what does not

- Works: all 10 example requests; new requests with other conditions, medications, lab values, encounter filters (incl. main diagnosis `primary_icd10`), grouping by hospital/sex/age band/region, ratio definitions, and decision wording beyond the examples (tested in `tests/`).
- Limits: trends are per calendar year only (no monthly results; age is as of 31 December 2025 for every year); the guided builder shows one level of AND/OR grouping (deeper nesting only via the JSON input); the disclosure ledger compares count cells for nested populations differing by 1–9 patients, not percentages, means or combinations of three or more queries. The parser understands the definition style used in the challenge requests, not arbitrary prose; unusual phrasing produces a staff-review warning. Requested groupings ("regions with the highest rates", "which hospitals"), prevalence ("rates of"), the access pathway ("pseudonymised records", "secure processing environment") and purpose sentences ("we are …", "so that …") are carried from the question onto the review screen, where the applicant confirms them; analyses the engine cannot compute (odds ratios, regression, prediction) are refused with an explanation rather than silently replaced by counts, and a relationship between two conditions is offered as a co-occurrence count only after the applicant chooses it. Every new question clears the previous request's purpose, access type, clarification answers and decision; only the applicant's name, organisation and country are kept. Art. 54 detection is keyword/pattern based, so a cleverly disguised purpose could pass and needs a human reviewer. A purpose that matches neither Art. 54 nor an Art. 53 category is held as `REVIEW_REQUIRED` and releases nothing; the staff dashboard shows such requests but has no approve/reject action yet, so the hold is final within the prototype. The SPE is a simulation (a locked folder + blocked export API), not a real isolated compute environment.


---

# Challenge description (from the organisers)

The European Health Data Space (EHDS) lets researchers, agencies and companies reuse health data. In each EU country, a Health Data Access Body decides every request. Approved applicants get anonymised statistics or pseudonymised records inside a secure processing environment (SPE) that they cannot download from.

## What to build

An app that takes a request as JSON and does the following automatically:

1. Decides `REJECT`, `APPROVE_STATISTICAL` or `APPROVE_SPE` and cites the articles.
2. For statistics, computes the numbers from the data using the small-cell rule.
3. For an SPE, copies the selected patients' requested records into it, reports the exact patient count and blocks downloads.
4. Shows the decision and the reason to the applicant and to staff.

Decide in this order:

1. Prohibited purpose: `REJECT`, Art. 54.
2. Health data request: `APPROVE_STATISTICAL`, Art. 53 and 69.
3. Access application that explains why statistics are not enough, like R05: `APPROVE_SPE`, Art. 53, 66, 68 and 73.
4. Any other access application, like R09: `APPROVE_STATISTICAL`, Art. 53, 66 and 69.

Small-cell rule:

- Show every requested group, even if empty.
- Show counts below 10, including 0, as `<10`.
- Show `suppressed` for a percentage whose numerator or denominator is below 10 and for a mean of fewer than 10 patients.
- Round to one decimal, halves up.

Also:

- Generate the queries from each request's definitions. Nothing is written by hand per request.
- The definitions name conditions, drugs and measurements in plain English. The data codes them as ICD-10, ATC and LOINC, so your app has to map one to the other.
- Leave out opted-out patients everywhere.
- An SPE can be a separate schema or folder. Copy in only the selected patients and the record types the request names and keep their patient IDs. Show a blocked download in your demo.
- Patient-level data stays inside your app. You may send the request text to external services such as a cloud LLM.

## What you get

- `data/`: 20,000 synthetic patients from three Austrian hospitals in five CSV files: patients, conditions (ICD-10), medications (ATC), observations (LOINC) and encounters. `data/README.md` describes the columns.
- `requests/`: 10 requests as JSON, each with an applicant, a purpose and plain-English definitions of the numbers. `requested_as` is `health data request` (statistics) or `health data access application` (records).
- `requests/expected.json`: the expected decision, articles and numbers for each request.
- The law: [Regulation (EU) 2025/327](https://eur-lex.europa.eu/eli/reg/2025/327/oj/eng), Chapter IV.

## Metrics

Measure these on the 10 requests. The jury also tests new requests on the same data. They can filter inpatient stays by main diagnosis (`primary_icd10`) or group by hospital (`data_holder`), which none of the 10 does.

| Metric | How to measure | Target |
|---|---|---|
| Correct decisions | Decision and articles match `expected.json`. Adding Art. 67 or 71 is fine | 10 of 10 |
| Correct numbers | Approved requests where every expected value appears under its group. Layout does not matter | 6 of 6 |
| Privacy leaks | Statistical answers with a patient-level record, a count below 10, an unsuppressed value or opted-out patients | 0 |
| Consistency | Requests with the same result across 3 runs | 10 of 10 |
| Time to answer | Slowest of the 10, from submitting to the answer | Under 60 s |

## You decide

- Rules, an LLM or both.
- What applicants and staff see.
- One central database or federated queries per hospital.
- Whether to convert the data to OMOP or FHIR.

## Rules

Use only synthetic data.

## Judging

| Criterion | Weight | Question |
|---|---|---|
| Clinical approach and impact | 33.3% | Would an access body use this? Can an applicant understand the decision? |
| Technical quality and AI | 33.3% | Are decisions and numbers correct? Are the small-cell rule, opt-outs and download block enforced? If you use AI, does it help? |
| Execution and prototype | 33.3% | Does the app work end to end, on the 10 requests and on new ones? |

The jury scores each criterion from 1 to 10 and takes your metric results into account.

## Submission

1. A video of up to 2 minutes showing your demo. Say what works and what does not.
2. A link to your public repository with instructions to run it. Its README has to explain your technical architecture: which components you built, how they work together and which models or services you use.
3. Your metric results on the 10 requests.
