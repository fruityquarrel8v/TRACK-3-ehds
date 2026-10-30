# Compliance audit — AIM Health Data Navigator (Track 3, EHDS)

Audit of the prototype as pushed on branch `devin/1793301500-ehds-prototype` (commit 2efbd65). Every finding below was
produced by running the real backend (`ehds_hdab.pipeline.process_request`) on the official synthetic data in a fresh
Python process with an isolated disclosure ledger and audit log, exactly as the Streamlit app calls it. No code was
changed for this audit. "AI" in this report means the optional language-model adapters; the default interpreter is
rule-based (`provider_name()` returned `rules`), so **no AI model was active in any test**.

## Summary table

| Area | Status | Evidence | Severity | Recommended action |
|---|---|---|---|---|
| Prohibited-purpose rejection | **PARTIALLY IMPLEMENTED** | A (insurance) and C (employment) → `REJECT`, Art. 54. **B (advertising) → `APPROVE_STATISTICAL` and 2,436 obesity patients released.** Purpose "decide which patients to drop from our insurance" → approved, 1,152 released. `ehds_hdab/decision.py::_prohibited` | **High** (challenge metric + Art. 54) | Quick fix: broaden the advertising/insurance patterns (`advertisement(s)`, `residents`, `drop/cancel … insurance`) and add these five sentences as tests. |
| Statistics / SPE distinction | **IMPLEMENTED AND TESTED** (as a simulated SPE) | Applicant chooses the route on the review screen (`REQUEST_TYPES`, `app.py`); `decision.py::decide` downgrades an access application to statistics unless the purpose justifies records (`_records_justified`); SPE = folder per request with cohort-only CSVs, manifest, download raises `DownloadBlockedError` (`ehds_hdab/spe.py`). Tests: `tests/test_requests.py::test_spe_includes_only_requested_tables_and_hba1c_observations`, `tests/test_generalisation.py::test_heart_failure_spe_contains_only_the_cohort_and_blocks_downloads`. | Low | Keep, but label it honestly in the demo and README as a *simulated* SPE (a folder plus a blocked-download button, not an isolated computing environment). |
| Authorization and human review | **PARTIALLY IMPLEMENTED** | Backend has only two outcomes (`REJECT` / `APPROVE_*`) plus a boolean `needs_staff_review` flag (`pipeline.py` l.133). **The flag does not stop release**: a request flagged for review still gets its numbers. There is no pending state, no staff approve/reject action, and the staff page only lists decisions (`app.py::page_staff`). The applicant screen shows "Approved for statistics" and "A staff member will review it" together. | **High** (misleading UI; numbers released before any human looks) | Moderate fix: add a third decision `REVIEW_REQUIRED` that withholds numbers, a staff approve/reject button that records the human decision in the audit log, and reword the applicant screen. Quick partial fix: when `needs_staff_review` is true, withhold the answer and change the badge. |
| Differencing protection | **PARTIALLY IMPLEMENTED** | Query A → 72 released; Query B → withheld, audited, in both orders, across restarts and across applicants (`ehds_hdab/disclosure.py::assess`, `tests/test_differencing.py`, 5 tests). Refusal text does not contain 71 or 1. **Gaps:** your exact wording "excluding patients from Burgenland" is misread as "in Burgenland"; only pairwise nested *count* cells are compared (not percentages/means, not 3-query combinations, not total-minus-shown-categories); ledger can be reset from the staff page. | Medium | Quick fix: exclusion phrasing in `nl.py`. Moderate fix: compare percentage numerators/denominators and the sum of shown categories against totals. Too substantial before submission: general multi-query inference control. |

Ranking by impact on challenge compliance and patient privacy:
1. Advertising purpose approved and numbers released (Task 1-B) — a prohibited purpose releases data.
2. "Needs review" does not withhold results, and the UI says "approved" and "staff review" at once (Task 3).
3. Exclusion phrasing misread (Task 4) — a researcher gets an answer to a different question than asked.
4. Ledger covers only pairwise nested counts (Task 4) — documented limitation, not a demonstrated leak.
5. SPE is a simulated workflow (Task 2) — acceptable for a hackathon if stated.

---

## Task 1 — Prohibited purposes

Method: each sentence was run through the same two functions the app uses, `ehds_hdab.providers.interpret` (turns the
sentence into criteria) and `ehds_hdab.pipeline.process_request` (decides, computes, privacy-checks). Because the app
asks for a separate "Research purpose" text box on the review screen, each sentence was tested three ways: as the
purpose, with an empty purpose, and with the purpose "research". `decide()` reads purpose **and** question together
(`decision.py` l.17-21), so the result was the same in all three variants.

| Test | Decision | Articles | Expected met? | Decided by | Rules or AI | Statistics released before authorization? |
|---|---|---|---|---|---|---|
| A insurance premiums | `REJECT` | 54 | Yes | `decision.py::_prohibited` → `insurance_credit` (`insurance` + `premiums`) | Deterministic regex | No: `process_request` returns before any query runs (`pipeline.py` l.83-89) |
| B advertisements for weight-loss products | `APPROVE_STATISTICAL` | 53, 69 | **No** | `_prohibited` returned `None`; `decide` fell through to the statistical branch | Deterministic regex | **Yes: `{"All": {"patients": 2436}}` (obesity, E66) was computed and returned.** Flagged `needs_staff_review=True` only because words were unrecognised, not because of the purpose. |
| C employer hiring decisions | `REJECT` | 54 | Yes | `_prohibited` → `employment` (`employer` + `hiring`) | Deterministic regex | No |
| D university research, counts by region | `APPROVE_STATISTICAL` | 53, 69 | Yes | `decide` statistical branch; purpose category "scientific research" | Deterministic regex | Released after the decision, through small-cell rule, output check and ledger (`pipeline.py` l.111-129) |
| E "commercial project" | `APPROVE_STATISTICAL` | 53, 69 | **Partly** | `decide` statistical branch; `needs_staff_review=True` only because the interpreter found no population criteria | Deterministic regex | No numbers (answer `{"All": {}}`), but the decision label is "approved" and the reason text says "Scientific research is an allowed purpose", which is not what the applicant wrote. |

Why B slips through (`decision.py` l.42-86): the advertising rule needs "advertising/marketing" preceded by *for/our/its/their*
or followed by *purposes/materials/to/of*, or the word "promot…", or an audience word from the list
*prescribers, GPs, physicians, doctors, consumers, customers*. The sentence uses the plural noun "advertisements" and the
audience "residents", so none of the patterns match. This is a rule-coverage gap, not a design error; the purpose is
evaluated separately from the clinical question, but the vocabulary is too narrow.

Additional probe: question "How many patients have type 2 diabetes?" with purpose "We will decide which patients to drop
from our insurance." → `APPROVE_STATISTICAL`, 1,152 released, `needs_staff_review=True` only because the purpose matched
no Art. 53 category. The insurance rule requires a decision word such as *premiums, pricing, underwriting, eligibility,
deny/exclude applicants*; "drop from our insurance" is not covered.

Existing tests cover ten prohibited and legitimate Art. 54 sentences (`tests/test_probes.py` l.234-275) and the four
official rejections R02/R04/R06/R10 (`tests/test_requests.py::test_decisions_match_all_ten_requests`); none of the five
audit sentences were in the suite.

## Task 2 — Statistics versus secure processing environment

1. **Route identification:** the applicant picks "Statistics only (health data request)" or "Access to pseudonymised
   records in a secure processing environment (health data access application)" on the review screen
   (`app.py` `REQUEST_TYPES`, `_screen_review`). The backend then applies the challenge order: Art. 54 → statistics
   (Art. 53, 69) → access application with justification (Art. 53, 66, 68, 73) → access application without
   justification downgraded to statistics (Art. 53, 66, 69) (`decision.py::decide`, l.217-264).
2. **Explicit request:** yes, via that selector; for an access application the applicant also ticks which record types
   (conditions, medications, observations, encounters) go into the SPE.
3. **Explanation of the difference:** short caption on the review screen ("Explain in the purpose why anonymised
   statistics are not enough … Otherwise the request is downgraded to statistics") and `DECISION_EXPLANATIONS` on the
   result screen. Adequate but brief; no side-by-side comparison.
4. **Implemented?** Yes, as a file-system workspace: `spe.py::create_workspace` creates `spe_workspaces/<request-id>/`
   with `cohort.csv` (patient IDs only), one CSV per requested record type filtered to the cohort (observations further
   filtered to the requested LOINC codes), and `manifest.json` (`download_allowed: false`, articles, counts).
5. **Restrictions enforced:** path traversal guard on the request id and table name; `request_download()` always raises
   `DownloadBlockedError` and writes a `download_blocked` audit event; the Streamlit SPE page uses `st.table` (no export
   toolbar) and shows only the first 50 rows. Verified by `tests/test_generalisation.py` l.143-157 and
   `tests/test_requests.py` l.80-89.
6. **Does patient-level data leave the boundary?** Not through the app's download paths. It is, however, *displayed* in
   the browser on the SPE page (first 50 rows of pseudonymised records), which an applicant could screenshot or retype.
   That is inherent to any SPE with a screen; it should be stated in the demo. No patient-level data reaches the
   statistics path (`privacy.py::check_statistical_output` scans every statistical answer for record lists and `p-…` IDs).
7. **Simulated or functioning?** Simulated. It is a folder with CSVs and an in-app viewer. There is no isolated compute,
   no user authentication, no output checking of analysis results, and the files are readable by anyone with access to
   the server's disk. **It must not be described as a production-ready SPE.**
8. **Decision ↔ route linkage:** correct. R05 (justified) → `APPROVE_SPE` with a workspace of the exact cohort; R09
   (not justified) → `APPROVE_STATISTICAL` with Art. 66 cited; health data requests never create a workspace
   (`pipeline.py` l.100-110).

Smallest honest demonstration: show R05 approved → open the SPE page → cohort count, record tables, click "Download
this table" → blocked with the audit entry; say out loud that this is a simulated workspace.

## Task 3 — Authorization decisions and human review

1. **Does a human-review status exist in the backend?** Only as a boolean flag `needs_staff_review` set when the
   decision is not `REJECT` and there are warnings, an empty query plan, or no matched Art. 53 category
   (`pipeline.py` l.133-136). The decision field itself is always one of `REJECT`, `APPROVE_STATISTICAL`, `APPROVE_SPE`.
   There is no `REVIEW_REQUIRED`/pending decision.
2. **Are pending requests prevented from releasing results?** **No.** The flag is informational. Observed:
   `needs_staff_review=True` with `{"All": {"patients": 1152}}` released (unmatched purpose) and
   `{"All": {"patients": 2436}}` released (unrecognised words). The only mechanism that withholds numbers is the
   disclosure ledger (`withheld=True`), which is a privacy check, not a human-review gate.
3. **Staff review interface?** `app.py::page_staff` shows an overview table (decision, articles, purpose category,
   needs-review flag, privacy check), lets staff inspect a request, see triggered rules and the generated query plan,
   view the disclosure ledger and audit log. It is a **read-only dashboard**.
4. **Can staff approve or reject?** **No.** There are no approve/reject controls; the only action is the demo-only
   "Reset disclosure ledger" button.
5. **Are decisions and explanations recorded?** Decisions, articles, purpose category and the review flag are appended
   to `audit_log.jsonl` (`pipeline.py` l.154-164); the human-readable reason is kept only in the Streamlit session
   (`store()`), so it is lost when the browser session ends. Withheld releases and blocked downloads are audited.
6. **EHDS provisions cited?** Yes: Art. 54 for rejections; 53 + 69 for statistics; 53, 66, 68, 73 for SPE; 53, 66, 69
   for downgraded applications; Art. 71 is referenced in the opt-out caption. Each article is shown with a one-line
   description and a link to Regulation (EU) 2025/327 (`app.py::show_articles`). Matches `expected.json` for all ten
   requests (`tests/test_requests.py`).
7. **Can the user understand why?** For clear cases yes: badge, reason sentence, legal basis, "How this was defined".
   For flagged cases the screen is contradictory: a green "Approved for statistics" badge, then the amber "Parts of this
   request could not be interpreted automatically. A staff member will review it", then the numbers
   (`app.py::show_response` l.104-115). The reason text for an unmatched purpose reads "No prohibited purpose was found
   … staff review is required" while the badge says approved. For test E the reason even says "Scientific research is an
   allowed purpose" because the word "research" appeared in the probe's substitute purpose; with the applicant's own
   words it says the purpose was not matched.

Automated recommendation vs final authorization: the app does not distinguish them. Everything it shows is an
automated outcome presented as final. The footer disclaimer ("decisions are generated automatically … not legally
binding") is the only caveat.

## Task 4 — Protection against differencing attacks

Independent reference (pandas on the CSVs, previous session): A = 72, B = 71, Burgenland = 1, opt-outs excluded.

| Step | Interpretation shown to the researcher | Outcome |
|---|---|---|
| Query A "How many female breast cancer patients aged 45–74 live in Austria?" | female, C50, aged 45–74 | `APPROVE_STATISTICAL`, `{"All": {"patients": 72}}` released and recorded |
| Query B "…, excluding Burgenland?" | …and **not in region Burgenland** | `withheld=True`, answer `null`, audit event `differencing_withheld`, 4 safe alternatives offered |
| Query B2 "…, excluding **patients from** Burgenland?" | …and **in region Burgenland** (wrong) | `<10` released for the wrong population; the attack never reaches the ledger |

1. **Interpreted correctly?** A yes; B yes; B2 (your other wording) **no** — `nl.py` l.207-219 only recognises an
   exclusion word directly before the region name.
2. **Tracks previous disclosures?** Yes: every released count cell is appended to `disclosure_ledger.jsonl` with its
   size and salted SHA-256 tokens of the patient IDs (`disclosure.py::record`). Three rows were written for A, B, B2.
3. **Second result blocked?** Yes, in both orders (B first then A is also withheld). Blocking is all-or-nothing for the
   whole result; no generalisation (e.g. coarser regions) is attempted, only suggested in text.
4. **Bypass via other accounts/sessions?** No in the default `shared` audience: a different applicant name and a fresh
   process were also withheld (previous session). The optional `EHDS_DISCLOSURE_AUDIENCE=applicant` mode *would* allow
   it; nothing in the app sets it, but it should be removed or hard-disabled.
5. **Graphs, tables, tooltips, exports?** All derive from the released answer only: `charts.py::plan_inputs` reads
   `resp["answer"]` (post-small-cell), tooltips show the `shown` column, suppressed cells are plotted as gaps with a
   `<10`/`suppressed` marker, the JSON download serialises the same `answer`, and the PNG is rendered from the same table
   (`app.py` l.116-137, 313-316). Withheld results have `answer=None`, so nothing can be charted or downloaded.
6. **Small-cell bypass via totals or complementary categories?** The engine emits no totals row, so a single grouped
   answer cannot be subtracted internally. Across queries the ledger only compares **pairs** of nested count cells
   whose sizes differ by 1–9. Demonstrated: "by region" (4 regions shown, 5 shown as `<10`) followed by the total (72)
   were both released; here the difference (23) spans five hidden regions, so no single region is exposed, but if only
   one region had been hidden its exact count would be total − sum of shown cells and the pairwise check would not
   notice (each pairwise gap ≥ 10). Likewise percentages/means are never compared. **Not a comprehensive differencing
   protection.**
7. **Rounding and other protections?** Counts `<10` incl. 0 → `<10`; percentages with numerator or denominator `<10`
   and means of `<10` patients → `suppressed`; one decimal, halves up via `Decimal.quantize(ROUND_HALF_UP)`
   (`privacy.py`). Verified by `tests/test_privacy.py` and the official evaluation (0 leaks).

Does the refusal reveal the count? The applicant-facing text, alternatives and UI contain neither 71 nor 1 (the only
"71" in the whole response object was inside the processing-time float). The `conflicts` field names the earlier
request ID and relation "subset" but is not rendered. Residual, unavoidable: being refused tells the researcher the
excluded group has 1–9 patients, which the `<10` rule reveals anyway.

Verdict for this attack: blocked. Verdict for differencing in general: **partially implemented** (nested count pairs
only).

## Task 5 — Requirements checklist (official Track 3 README)

| Requirement | Status | Evidence |
|---|---|---|
| Takes a request as JSON | IMPLEMENTED AND TESTED | `ehds_hdab/cli.py`, `pipeline.py::process_file`; `tests/test_requests.py` runs `requests/R01–R10.json` |
| Decides REJECT / APPROVE_STATISTICAL / APPROVE_SPE and cites articles | IMPLEMENTED AND TESTED | `decision.py::decide`; `test_decisions_match_all_ten_requests` (10/10 vs `expected.json`) |
| Decision order (Art. 54 → request → justified application → other application) | IMPLEMENTED AND TESTED | `decision.py` l.217-264; `tests/test_probes.py` SPE-justification cases |
| Prohibited purposes beyond the 10 examples | PARTIALLY IMPLEMENTED | regex vocabulary misses audit sentences B and the "drop from insurance" probe (this report, Task 1) |
| Compute numbers from the data with the small-cell rule | IMPLEMENTED AND TESTED | `engine.py`, `privacy.py::apply_small_cell`; `test_expected_values_appear_under_each_group` (6/6), `tests/test_privacy.py` |
| Show every requested group, even if empty; `<10` incl. 0 | IMPLEMENTED AND TESTED | `apply_small_cell`; `test_counts_suppress_below_ten_including_zero` |
| `suppressed` for %/mean with <10; one decimal, halves up | IMPLEMENTED AND TESTED | `privacy.py`; `test_percentage_and_mean_small_cells_are_suppressed`, `test_round_half_up_and_trailing_decimal` |
| SPE: copy only selected patients + named record types, keep patient IDs, report exact count | IMPLEMENTED AND TESTED | `spe.py::create_workspace`; `test_spe_includes_only_requested_tables_and_hba1c_observations` |
| SPE blocks downloads, shown in demo | IMPLEMENTED AND TESTED (simulated) | `SecureProcessingEnvironment.request_download`; `page_spe` button; `test_heart_failure_spe_contains_only_the_cohort_and_blocks_downloads` |
| Show decision and reason to applicant and staff | PARTIALLY IMPLEMENTED | shown (`show_response`, `page_staff`), but review-flagged cases are presented contradictorily and staff cannot act |
| Queries generated from each request's definitions, nothing hand-written per request | IMPLEMENTED AND TESTED | `parser.py`, `criteria.py`; `tests/test_generalisation.py` runs 7 unseen requests (incl. `primary_icd10` filter and `data_holder` grouping) against direct pandas |
| Map plain English to ICD-10 / ATC / LOINC | IMPLEMENTED AND TESTED | `terminology.py`, `nl.py`; `tests/test_nl.py`, `tests/test_criteria.py` |
| Leave out opted-out patients everywhere | IMPLEMENTED AND TESTED | `data.py` removes them at load; `test_opted_out_patients_are_removed_from_every_loaded_table` |
| Patient-level data stays inside the app | IMPLEMENTED AND TESTED | default provider `rules` sends nothing anywhere; optional model adapters send only the request text (`providers.py`); `check_statistical_output` |
| Metrics: 10/10 decisions, 6/6 numbers, 0 leaks, 10/10 consistency, <60 s | IMPLEMENTED AND TESTED | `evaluate.py` → 10/10, 6/6, 0, 10/10, slowest 0.24 s |
| Works on new requests (jury) | IMPLEMENTED BUT NOT VERIFIED beyond the 7 in `tests/new_requests` | rule-based interpreter; this audit found two misinterpretations ("excluding patients from", "commercial project" → no criteria) |
| README explains architecture, components, models/services | IMPLEMENTED AND TESTED (manually read) | `README.md` Architecture / Design choices / Limitations |
| ≤2-minute video saying what works and what does not | NOT IMPLEMENTED (submission artefact) | — |
| Use only synthetic data | IMPLEMENTED | `data/` only |
| Human review workflow (not an explicit README requirement, but implied by "Health Data Access Body decides every request") | PARTIALLY IMPLEMENTED | flag only, no gate, no staff action (Task 3) |
| Differencing / inference control (not an explicit README requirement) | PARTIALLY IMPLEMENTED | `disclosure.py`; `tests/test_differencing.py`; limits in Task 4 |

## Task 6 — Fix estimates

| Issue | Effort | Risk to existing results |
|---|---|---|
| Advertising / insurance vocabulary gaps (Task 1-B, "drop from insurance") + regression tests for the five audit sentences | Quick (regex + tests) | Low; must re-run the legitimate Art. 54 cases in `test_probes.py` to avoid over-rejecting e.g. "post-marketing surveillance" |
| Withhold numbers when `needs_staff_review` is true; change badge to "Review required" | Quick (pipeline + two UI strings) | Low; some probe tests assert `needs_staff_review` with answers present and would need updating |
| Proper `REVIEW_REQUIRED` decision + staff approve/reject with audit entries + applicant sees final decision | Moderate (half a day) | Medium; touches decision schema, evaluation harness and app flows; the 10 official requests must stay 10/10 |
| "excluding patients from X" and similar exclusion phrasings | Quick (one regex in `nl.py` + tests) | Low |
| Ledger: compare percentage numerators/denominators; compare sum of shown categories vs totals | Moderate | Medium; could withhold legitimate results, needs careful tests |
| Remove `EHDS_DISCLOSURE_AUDIENCE=applicant` mode and the staff "Reset ledger" button (or hide behind an env flag) | Quick | Low |
| General multi-query inference control (3+ queries, overlapping non-nested populations) | Too substantial before submission | — |
| Real SPE (isolation, auth, output checking) | Too substantial before submission; describe as simulated | — |

## How the tests were run

```
PYTHONPATH=. EHDS_LEDGER_PATH=/tmp/audit/ledger.jsonl EHDS_AUDIT_LOG=/tmp/audit/audit.jsonl python3 /tmp/audit/run_purpose.py
PYTHONPATH=. python3 /tmp/audit/run_diff.py A ; … B ; … B2      # one process per query, shared isolated ledger
PYTHONPATH=. python3 /tmp/audit/run_comp.py R T  /  T R B        # by-region vs total, both orders
python -m pytest -q   → 176 passed ;  python evaluate.py → 10/10, 6/6, 0 leaks, 10/10, 0.24 s
```
The running app's `disclosure_ledger.jsonl` and `audit_log.jsonl` were not touched.
