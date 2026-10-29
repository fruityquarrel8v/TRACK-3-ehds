import json
import time
from decimal import Decimal as Dd, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
import pytest

from ehds_hdab import config
from ehds_hdab.decision import decide
from ehds_hdab.engine import _selected_observations
from ehds_hdab.pipeline import process_request
from ehds_hdab.plan import QueryPlan
from ehds_hdab.parser import parse_request

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
p = pd.read_csv(D / "patients.csv", dtype=str)
p = p[p.opted_out.str.lower() != "true"].copy()
p["age"] = 2025 - p.birth_date.str[:4].astype(int)
ids = set(p.patient_id)
c = pd.read_csv(D / "conditions.csv", dtype=str)
c = c[c.patient_id.isin(ids)]
m = pd.read_csv(D / "medications.csv", dtype=str, keep_default_na=False)
m = m[m.patient_id.isin(ids)]
o = pd.read_csv(D / "observations.csv", dtype=str)
o = o[o.patient_id.isin(ids)]
e = pd.read_csv(D / "encounters.csv", dtype=str, keep_default_na=False)
e = e[e.patient_id.isin(ids)]


def cond(pref):
    return set(c[c.icd10.str.startswith(tuple(pref))].patient_id)


def active(pref):
    x=m[m.atc.str.startswith(tuple(pref))]; x=x[(x.start_date<='2025-12-31')&((x.end_date=='')|(x.end_date>='2025-12-31'))]; return set(x.patient_id)


def latest(loinc):
    x=o[o.loinc==loinc].sort_values('date').drop_duplicates('patient_id',keep='last'); return dict(zip(x.patient_id,x.value.astype(float)))


def sc(n):
    n=int(n); return '<10' if n<10 else n


def pct(a, b):
    a,b=int(a),int(b); return 'suppressed' if a<10 or b<10 else float((Dd(a*100)/Dd(b)).quantize(Dd('0.1'),ROUND_HALF_UP))


def mean(vals):
    return 'suppressed' if len(vals)<10 else float((sum(Dd(str(v)) for v in vals)/Dd(len(vals))).quantize(Dd('0.1'),ROUND_HALF_UP))


base = {
    "applicant": {"name": "X", "organisation": "Uni", "country": "AT"},
    "requested_as": "health data request",
    "requested_format": "anonymised statistics",
}


def _probe_cases():
    probes = []
    x = e[(e.type == "inpatient") & e.primary_icd10.str.startswith("I50") & e.start_date.str.startswith("2024")]
    probes.append((
        "P1",
        {**base, "id": "P1", "purpose": "Research on hospital burden of heart failure.", "question": "How many inpatient stays with a main diagnosis of heart failure started in 2024, by hospital?",
         "definitions": ["Heart failure stay = encounter with type 'inpatient' and primary_icd10 starting with I50.", "Count stays (not patients) whose start_date is in 2024.", "Report per hospital: heart failure stays."]},
        "APPROVE_STATISTICAL",
        {h: sc(n) for h, n in x.groupby("data_holder").size().items()},
    ))

    a = p[p.age >= 18]
    ob = cond(["E66"])
    probes.append((
        "P2",
        {**base, "id": "P2", "purpose": "Public health monitoring of obesity.", "question": "What share of adults has obesity, per hospital?",
         "definitions": ["Adult = aged 18 or older on 31 December 2025.", "Obesity = any condition with an ICD-10 code starting with E66.", "Obesity rate = adults with obesity / all adults × 100, one decimal.", "Report per hospital: adults, adults with obesity and the obesity rate."]},
        "APPROVE_STATISTICAL",
        {h: [sc(len(g)), sc(g.patient_id.isin(ob).sum()), pct(g.patient_id.isin(ob).sum(), len(g))] for h, g in a.groupby("data_holder")},
    ))

    eg = latest("62238-1")
    ck = cond(["N18"])
    exp = {}
    for s, g in p.groupby("sex"):
        v = [eg[i] for i in g.patient_id if i in ck and i in eg]
        exp[s] = [sc(len(v)), mean(v)]
    probes.append((
        "P3",
        {**base, "id": "P3", "purpose": "Research on kidney function in chronic kidney disease.", "question": "What is the mean eGFR of patients with chronic kidney disease by sex?",
         "definitions": ["Chronic kidney disease = any condition of chronic kidney disease.", "Use the latest eGFR (LOINC 62238-1) per patient; patients without an eGFR are left out.", "Report per sex: patients with chronic kidney disease who have an eGFR value and their mean eGFR, one decimal."]},
        "APPROVE_STATISTICAL",
        exp,
    ))

    st = active(["C10AA"])
    ht = cond(["I10"])
    exp = {}
    for r, g in a.groupby("region"):
        h = g[g.patient_id.isin(ht)]
        n = h.patient_id.isin(st).sum()
        exp[r] = [sc(len(h)), pct(n, len(h))]
    probes.append((
        "P4",
        {**base, "id": "P4", "purpose": "Quality of care research on cardiovascular prevention.", "question": "Among adults with hypertension, what share has an active statin, by region?",
         "definitions": ["Adult = aged 18 or older on 31 December 2025.", "Hypertension = any condition with an ICD-10 code starting with I10.", "Statin = any statin.", "A medication is active on 31 December 2025 if start_date is on or before that day and end_date is empty or on or after that day.", "Report per region: adults with hypertension and % with an active statin, one decimal."]},
        "APPROVE_STATISTICAL",
        exp,
    ))

    pn = set(e[(e.type == "inpatient") & e.primary_icd10.str.startswith("J18")].patient_id)
    bands = [(0, 17, "0-17"), (18, 64, "18-64"), (65, 200, "65+")]
    exp = {lab: [sc(p[(p.age >= lo) & (p.age <= hi)].patient_id.isin(pn).sum())] for lo, hi, lab in bands}
    probes.append((
        "P5",
        {**base, "id": "P5", "purpose": "Public health research on pneumonia hospitalisations.", "question": "How many patients had an inpatient stay for pneumonia, by age band?",
         "definitions": ["Pneumonia stay = encounter with type 'inpatient' and primary_icd10 starting with J18.", "Age bands on 31 December 2025: 0-17, 18-64, 65+.", "Report per band: patients with a pneumonia stay."]},
        "APPROVE_STATISTICAL",
        exp,
    ))

    cp = cond(["J44"])
    exp = {}
    for r, g in p.groupby("region"):
        exp[r] = {}
        for lo in range(0, 100, 10):
            hi = lo + 9 if lo < 90 else 200
            lab = f"{lo}-{lo + 9}" if lo < 90 else "90+"
            exp[r][lab] = [sc(g[(g.age >= lo) & (g.age <= hi)].patient_id.isin(cp).sum())]
    probes.append((
        "P6",
        {**base, "id": "P6", "purpose": "Research on COPD epidemiology to plan pulmonary rehabilitation services.", "question": "How many patients have COPD, by 10-year age band and region?",
         "definitions": ["COPD = any condition with an ICD-10 code starting with J44.", "Age bands on 31 December 2025: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90+.", "Report per region and age band: patients with COPD."]},
        "APPROVE_STATISTICAL",
        exp,
    ))

    sb = o[o.loinc == "8480-6"].copy()
    sb["v"] = sb.value.astype(float)
    mx = sb.groupby("patient_id").v.max()
    exp = {}
    for s, g in p[p.patient_id.isin(cond(["E11"]))].groupby("sex"):
        v = [mx[i] for i in g.patient_id if i in mx.index]
        exp[s] = [sc(len(v)), mean(v)]
    probes.append((
        "P7",
        {**base, "id": "P7", "purpose": "Research on blood pressure control in diabetes.", "question": "What is the mean of the highest systolic blood pressure of patients with type 2 diabetes, by sex?",
         "definitions": ["Type 2 diabetes = any condition with an ICD-10 code starting with E11.", "Use the highest systolic blood pressure (LOINC 8480-6) per patient; patients without one are left out.", "Report per sex: patients with type 2 diabetes who have a systolic blood pressure value and their mean systolic blood pressure, one decimal."]},
        "APPROVE_STATISTICAL",
        exp,
    ))
    return probes


def _leaves(node):
    if isinstance(node, dict):
        return [value for child in node.values() for value in _leaves(child)]
    return [node]


def _assert_expected(expected, actual, path, failures):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if not isinstance(actual, dict) or key not in actual:
                failures.append(f"{path}/{key} missing")
                continue
            _assert_expected(value, actual[key], path + "/" + key, failures)
    else:
        actual_values = _leaves(actual)
        for value in expected if isinstance(expected, list) else [expected]:
            if not any(
                isinstance(value, (int, float))
                and isinstance(found, (int, float))
                and abs(value - found) < 1e-9
                or value == found
                for found in actual_values
            ):
                failures.append(f"{path}: want {value} got {actual_values}")


def test_independent_probe_aggregations_match():
    for probe_id, request, expected_decision, expected_answer in _probe_cases():
        started = time.perf_counter()
        response = process_request(request)
        elapsed = time.perf_counter() - started
        assert response["decision"] == expected_decision, probe_id
        failures = []
        _assert_expected(expected_answer, response["answer"], "", failures)
        assert not failures, f"{probe_id}: {failures}"
        if probe_id == "P1":
            assert response["query_plan"]["metrics"][0]["label"] == "heart_failure_stays"
        if probe_id == "P2":
            assert [metric["label"] for metric in response["query_plan"]["metrics"]] == [
                "adults",
                "adults_with_obesity",
                "obesity_rate",
            ]
        if probe_id == "P6":
            assert elapsed < 1.0, f"P6 took {elapsed:.3f}s"


_DECISION_CASES = [
    ("D1", "health data request", "A bank wants to use diagnoses to decide on credit scoring for loan applicants.", "How many patients have heart failure per region?", "REJECT"),
    ("D2", "health data request", "A pharmaceutical company wants to target its sales representatives at GPs who prescribe the most statins.", "How many patients have an active statin per region?", "REJECT"),
    ("D3", "health data access application", "We want to validate whether guideline recommendations are followed in GP practices for a quality report.", "Provide pseudonymised records of patients with asthma and their medications.", "APPROVE_STATISTICAL"),
    ("D4", "health data access application", "We develop a machine learning model to predict heart failure readmission; the model needs individual-level data.", "Provide pseudonymised conditions and encounters of patients with heart failure.", "APPROVE_SPE"),
    ("D5", "health data request", "A brewery wants to develop a new alcohol-free beer marketed to people with diabetes.", "How many adults have type 2 diabetes per region?", "REJECT"),
    ("D6", "health data request", "University research on alcohol-related liver disease to inform prevention policy.", "How many adults have type 2 diabetes per region?", "APPROVE_STATISTICAL"),
    ("D7", "health data request", "Our insurance company's research unit studies prevention programmes for asthma for public health.", "How many patients have asthma per region?", "APPROVE_STATISTICAL"),
    ("D8", "health data access application", "Training of medical students on interpreting HbA1c trends; a teaching module.", "Provide pseudonymised HbA1c values of patients with type 2 diabetes.", "APPROVE_STATISTICAL"),
    ("D9", "health data request", "An employer wants to know which of its staff have depression before promotions.", "How many adults have depression per region?", "REJECT"),
]


@pytest.mark.parametrize(
    "probe_id,requested_as,purpose,question,expected",
    _DECISION_CASES,
)
def test_decision_probes(probe_id, requested_as, purpose, question, expected):
    response = process_request({
        **base,
        "id": probe_id,
        "requested_as": requested_as,
        "purpose": purpose,
        "question": question,
        "definitions": ["Report per region: patients."],
    })
    assert response["decision"] == expected, (probe_id, response["reason"])


_LEGITIMATE_ART54_CASES = [
    "Which patient groups should be prioritised for influenza vaccination to reduce hospitalisations?",
    "Estimate diabetes prevalence by region; exclude patients under 18 from the analysis.",
    "Assess whether the burden of COPD differs across regions to plan public health insurance coverage of rehabilitation.",
    "A bank-funded foundation wants to assess heart failure prevalence for public health planning.",
    "Study prescribing of SGLT2 inhibitors by physicians since market authorisation to evaluate uptake of guidelines.",
    "The ministry plans a public-health campaign targeting GPs to improve statin prescribing in line with guidelines.",
    "A hospital quality report assesses staff performance in clinical documentation.",
]


@pytest.mark.parametrize("text", _LEGITIMATE_ART54_CASES)
def test_legitimate_public_health_requests_are_not_rejected(text):
    decision = decide({
        **base,
        "purpose": text,
        "question": text,
    })
    assert decision.decision == "APPROVE_STATISTICAL", decision.triggered_rules


_PROHIBITED_CREDIT_CASES = [
    "We will use the data to deny loan applications from people with chronic disease.",
    "The insurer wants to raise premiums for policyholders with heart failure.",
]


@pytest.mark.parametrize("text", _PROHIBITED_CREDIT_CASES)
def test_specific_credit_decisions_are_rejected(text):
    decision = decide({
        **base,
        "purpose": text,
        "question": text,
    })
    assert decision.decision == "REJECT"
    assert decision.articles == ["54"]


@pytest.mark.parametrize(
    "word,pick",
    [
        ("highest", "max"),
        ("maximum", "max"),
        ("max", "max"),
        ("lowest", "min"),
        ("minimum", "min"),
        ("min", "min"),
        ("first", "earliest"),
        ("earliest", "earliest"),
        ("last", "latest"),
        ("latest", "latest"),
        ("most recent", "latest"),
    ],
)
def test_observation_pick_synonyms_are_parsed(word, pick):
    request = {
        "question": f"What is the {word} systolic blood pressure?",
        "definitions": ["Use the latest systolic blood pressure (LOINC 8480-6) per patient."],
    }
    plan = parse_request(request)
    spec = next(spec for spec in plan.concepts.values() if spec.get("kind") == "observation")
    assert spec["pick"] == pick


def test_maximum_observation_selection_keeps_highest_value():
    observations = pd.DataFrame({
        "patient_id": ["p-000001", "p-000001", "p-000001"],
        "loinc": ["8480-6", "8480-6", "8480-6"],
        "value": ["120", "180", "130"],
        "date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
    })
    plan = QueryPlan(concepts={
        "systolic blood pressure": {
            "kind": "observation",
            "loinc": "8480-6",
            "pick": "max",
        }
    })
    selected = _selected_observations(
        {"observation": "systolic blood pressure"},
        {"p-000001"},
        plan,
        {"observations": observations},
    )
    assert selected.iloc[0]["_numeric"] == Dd("180")


def test_unmatched_purpose_requires_staff_review_and_is_explicit():
    request = {
        **base,
        "purpose": "A descriptive analysis for a newly established internal initiative.",
        "question": "How many patients have asthma per region?",
        "definitions": ["Report per region: patients with asthma."],
    }
    decision = decide(request)
    response = process_request(request)
    assert decision.purpose_category == ""
    assert "No prohibited purpose was found" in decision.reason
    assert "not matched to a specific Art. 53 category" in decision.reason
    assert response["needs_staff_review"] is True


def test_purpose_reason_uses_singular_and_plural_grammar():
    singular = decide({
        **base,
        "purpose": "Public health monitoring of obesity.",
        "question": "How many adults have obesity?",
    })
    plural = decide({
        **base,
        "purpose": "Public health research on obesity.",
        "question": "How many adults have obesity?",
    })
    assert "Public health is an allowed purpose" in singular.reason
    assert "Scientific research and public health are allowed purposes" in plural.reason


def test_statistical_reason_branches_end_with_a_period():
    requests = [
        {**base, "purpose": "Public health planning.", "question": "Count patients."},
        {
            **base,
            "requested_as": "health data access application",
            "purpose": "Policy-making.",
            "question": "Count patients by region.",
        },
        {
            **base,
            "requested_as": "other",
            "purpose": "Scientific research.",
            "question": "Count patients by region.",
        },
    ]
    for request in requests:
        decision = decide(request)
        assert decision.decision == "APPROVE_STATISTICAL"
        assert decision.reason.endswith(".")


def test_generic_detrimental_decisions_against_identified_groups_are_rejected():
    decision = decide({
        **base,
        "purpose": "Use diagnoses to make detrimental decisions affecting identified patient groups.",
        "question": "Which groups should be excluded from access to services?",
    })
    assert decision.decision == "REJECT"
    assert decision.articles == ["54"]


@pytest.mark.parametrize("scale", ["× 100", "x 100", "* 100", "times 100", "per 100"])
def test_named_ratio_definitions_support_scale_synonyms(scale):
    request = {
        "question": "What is the obesity rate by region?",
        "definitions": [
            "Obesity = any condition with an ICD-10 code starting with E66.",
            f"Obesity rate = adults with obesity / all adults {scale}.",
            "Report per region: the obesity rate.",
        ],
    }
    plan = parse_request(request)
    assert len(plan.metrics) == 1
    assert plan.metrics[0]["label"] == "obesity_rate"
    assert plan.metrics[0]["type"] == "percentage"
    assert plan.metrics[0]["numerator_population"]
    assert plan.metrics[0]["denominator_population"]


def test_existing_requests_have_a_recognized_purpose_category():
    for path in sorted((ROOT / "requests").glob("R*.json")):
        request = json.loads(path.read_text(encoding="utf-8"))
        assert decide(request).purpose_category, request["id"]


def test_rejected_requests_are_final_and_audit_event_has_no_answer():
    request = {
        **base,
        "id": "AUDIT-REJECT",
        "purpose": _PROHIBITED_CREDIT_CASES[0],
        "question": _PROHIBITED_CREDIT_CASES[0],
    }
    response = process_request(request)
    assert response["decision"] == "REJECT"
    assert response["needs_staff_review"] is False
    events = [
        json.loads(line)
        for line in config.AUDIT_LOG.read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["event"] == "request_processed"
    assert events[-1]["request_id"] == "AUDIT-REJECT"
    details = events[-1]["details"]
    assert set(details) == {
        "decision",
        "articles",
        "purpose_category",
        "needs_staff_review",
        "processing_time_s",
    }
    assert "answer" not in details


def test_spe_workspace_creation_is_audited():
    request = json.loads((ROOT / "requests" / "R05.json").read_text(encoding="utf-8"))
    response = process_request(request)
    assert response["decision"] == "APPROVE_SPE"
    events = [
        json.loads(line)
        for line in config.AUDIT_LOG.read_text(encoding="utf-8").splitlines()
    ]
    created = next(event for event in events if event["event"] == "spe_workspace_created")
    assert created["request_id"] == request["id"]
    assert created["details"] == {"cohort_size": response["spe"]["cohort_size"]}
