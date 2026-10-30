import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

from ehds_hdab.criteria import criteria_to_definitions
from ehds_hdab.data import load_data
from ehds_hdab.nl import interpret_question
from ehds_hdab.pipeline import process_request


ROOT = Path(__file__).resolve().parents[1]
PURPOSE = "Public health research on chronic disease burden."


def _submit(text, request_id):
    interpretation = interpret_question(text)
    response = process_request({
        "id": request_id,
        "applicant": {
            "name": f"NL {request_id}",
            "organisation": "Public Health Research Institute",
        },
        "requested_as": "health data request",
        "requested_format": "anonymised statistics",
        "purpose": PURPOSE,
        "question": text,
        "definitions": criteria_to_definitions(interpretation.criteria),
        "criteria": interpretation.criteria,
    })
    return interpretation, response


def _condition_ids(data, prefix):
    rows = data["conditions"]
    return set(rows.loc[rows["icd10"].astype(str).str.startswith(prefix), "patient_id"])


def _latest_lab(data, loinc, start=None, end=None):
    rows = data["observations"].loc[
        data["observations"]["loinc"].eq(loinc)
    ].copy()
    if start:
        rows = rows.loc[rows["date"].ge(pd.Timestamp(start))]
    if end:
        rows = rows.loc[rows["date"].le(pd.Timestamp(end))]
    return rows.sort_values("date").drop_duplicates("patient_id", keep="last")


def _count(response, label="patients"):
    return response["answer"]["All"][label]


def test_hba1c_outcome_question_has_hand_built_criteria():
    text = (
        "Compare HbA1c levels between male and female patients with type 2 "
        "diabetes aged 40-65 during 2025."
    )

    interpretation = interpret_question(text)

    assert interpretation.criteria == {
        "population": {
            "match": "all",
            "items": [
                {
                    "type": "condition",
                    "label": "Type 2 diabetes",
                    "icd10": ["E11"],
                    "onset_end": "2025-12-31",
                },
                {"type": "age", "label": "age", "min": 40, "max": 65},
            ],
        },
        "group_by": [{"dimension": "sex"}],
        "metrics": [
            {"type": "count_patients", "label": "patients"},
            {
                "type": "mean_latest_lab",
                "label": "mean latest HbA1c (%)",
                "loinc": "4548-4",
                "period_start": "2025-01-01",
                "period_end": "2025-12-31",
            },
        ],
        "record_types": [],
    }
    assert interpretation.assumptions == [
        "Compared as the mean of each patient's latest HbA1c within the period."
    ]
    assert interpretation.unrecognised == []


def test_hba1c_outcome_means_match_direct_pandas_by_sex():
    data = load_data()
    text = (
        "Compare HbA1c levels between male and female patients with type 2 "
        "diabetes aged 40-65 during 2025."
    )
    interpretation, response = _submit(text, "NL-MEAN")
    t2d = _condition_ids(data, "E11")
    patients = data["patients"]
    cohort = set(patients.loc[
        patients["patient_id"].isin(t2d)
        & patients["age"].between(40, 65),
        "patient_id",
    ])
    latest = _latest_lab(
        data, "4548-4", "2025-01-01", "2025-12-31"
    )
    latest = latest.loc[latest["patient_id"].isin(cohort)].copy()
    latest["numeric"] = pd.to_numeric(latest["value"], errors="coerce")
    latest = latest.loc[latest["numeric"].notna()].merge(
        patients[["patient_id", "sex"]], on="patient_id", how="left"
    )
    expected = {}
    for sex, frame in latest.groupby("sex"):
        mean = Decimal(str(frame["numeric"].mean())).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
        expected[sex] = str(mean)

    assert interpretation.ok()
    assert response["decision"] == "APPROVE_STATISTICAL"
    for sex, mean in expected.items():
        actual = response["answer"][sex]["mean latest HbA1c (%)"]
        assert Decimal(str(actual)) == Decimal(mean)


def test_hospitalisation_question_targets_inpatient_primary_diagnosis():
    data = load_data()
    text = "How many patients were hospitalised for heart failure in 2025, by hospital?"
    interpretation, response = _submit(text, "NL-HOSPITAL")
    encounter_rows = data["encounters"]
    selected = encounter_rows.loc[
        encounter_rows["type"].str.casefold().eq("inpatient")
        & encounter_rows["primary_icd10"].astype(str).str.startswith("I50")
        & encounter_rows["start_date"].between(
            pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
        )
    ]
    selected_ids = set(selected["patient_id"])
    patients = data["patients"].loc[
        data["patients"]["patient_id"].isin(selected_ids)
    ]
    expected = patients.groupby("data_holder")["patient_id"].nunique().to_dict()

    assert interpretation.criteria["population"]["items"] == [
        {
            "type": "hospital_stay",
            "label": "Heart failure",
            "icd10": ["I50"],
            "period_start": "2025-01-01",
            "period_end": "2025-12-31",
        }
    ]
    assert interpretation.criteria["group_by"] == [{"dimension": "data_holder"}]
    assert sum(expected.values()) == 103
    assert response["decision"] == "APPROVE_STATISTICAL"
    for holder, count in expected.items():
        actual = response["answer"][holder]["patients"]
        assert actual == (count if count >= 10 else "<10")


def test_t2d_latest_hba1c_threshold_questions_match_pandas():
    data = load_data()
    t2d = _condition_ids(data, "E11")
    cases = [
        (
            "Patients with type 2 diabetes and HbA1c above 8% in 2025",
            "gt", "2025-01-01", "2025-12-31", 155,
        ),
        (
            "Patients with type 2 diabetes and HbA1c above 8%",
            "gt", None, None, 251,
        ),
        (
            "Patients with type 2 diabetes and HbA1c at least 8",
            "ge", None, None, 282,
        ),
    ]
    for index, (text, operator, start, end, anchor) in enumerate(cases):
        interpretation, response = _submit(text, f"NL-HBA1C-{index}")
        latest = _latest_lab(data, "4548-4", start, end)
        values = pd.to_numeric(latest["value"], errors="coerce")
        comparison = values.gt(8) if operator == "gt" else values.ge(8)
        threshold_ids = set(latest.loc[comparison, "patient_id"])
        expected = len(t2d & threshold_ids)

        assert expected == anchor
        assert interpretation.ok()
        assert response["decision"] == "APPROVE_STATISTICAL"
        assert _count(response) == expected


def test_adult_t2d_medication_status_questions_match_pandas():
    data = load_data()
    adults = set(data["patients"].loc[data["patients"]["age"].ge(18), "patient_id"])
    t2d = _condition_ids(data, "E11")
    rows = data["medications"].loc[
        data["medications"]["atc"].astype(str).str.startswith("A10BA02")
    ].copy()
    ever_ids = set(rows["patient_id"])
    reference = pd.Timestamp("2025-12-31")
    active_rows = rows.loc[
        rows["start_date"].le(reference)
        & (rows["end_date"].isna() | rows["end_date"].ge(reference))
    ]
    active_ids = set(active_rows["patient_id"])

    ever_text = "Adults with type 2 diabetes ever prescribed metformin"
    ever_interpretation, ever_response = _submit(ever_text, "NL-MET-EVER")
    active_text = "Adults with type 2 diabetes taking metformin"
    active_interpretation, active_response = _submit(active_text, "NL-MET-ACTIVE")
    assert len(adults & t2d & ever_ids) == 970
    assert len(adults & t2d & active_ids) == 885
    assert _count(ever_response) == len(adults & t2d & ever_ids)
    assert _count(active_response) == len(adults & t2d & active_ids)
    assert ever_interpretation.criteria["population"]["items"][-1]["status"] == "ever"
    assert not any("Medication status" in item for item in ever_interpretation.assumptions)
    assert active_interpretation.criteria["population"]["items"][-1]["status"] == "active"
    assert any("counted as active on 31 December 2025" in item
               for item in active_interpretation.assumptions)


def test_condition_or_and_exclusion_match_pandas():
    data = load_data()
    text = (
        "Patients with type 2 diabetes or obesity without hypertension, by sex"
    )
    interpretation, response = _submit(text, "NL-OR-EXCLUDE")
    t2d = _condition_ids(data, "E11")
    obesity = _condition_ids(data, "E66")
    hypertension = _condition_ids(data, "I10")
    patients = data["patients"]
    expected = (t2d | obesity) - hypertension
    assert interpretation.criteria["population"]["match"] == "all"
    assert interpretation.criteria["population"]["items"][0]["match"] == "any"
    assert interpretation.criteria["population"]["items"][1]["exclude"] is True
    assert interpretation.criteria["group_by"] == [{"dimension": "sex"}]
    for sex, frame in patients.loc[patients["patient_id"].isin(expected)].groupby("sex"):
        actual = response["answer"][sex]["patients"]
        count = frame["patient_id"].nunique()
        assert actual == (count if count >= 10 else "<10")


def test_medication_negation_phrases_exclude_the_concept():
    phrases = [
        "not taking LAMA",
        "not taking a LAMA",
        "not taking an LAMA",
        "not taking any LAMA",
        "not on LAMA",
        "not prescribed LAMA",
        "no LAMA",
        "without LAMA",
        "without a LAMA",
        "without an LAMA",
        "who do not take LAMA",
        "never prescribed LAMA",
    ]
    for phrase in phrases:
        interpretation = interpret_question(f"Patients with COPD {phrase}")
        medication = next(
            item for item in interpretation.criteria["population"]["items"]
            if item.get("type") == "medication"
        )
        assert medication["exclude"] is True, phrase
        if phrase.startswith("never prescribed"):
            assert medication["status"] == "ever"
            assert "never" not in interpretation.unrecognised


def test_copd_without_active_lama_in_klinikum_nord_matches_pandas():
    data = load_data()
    text = "Patients with COPD not taking a LAMA in Klinikum Nord"
    interpretation, response = _submit(text, "NL-COPD-NO-LAMA")

    copd = _condition_ids(data, "J44")
    nord_patients = set(
        data["patients"].loc[
            data["patients"]["data_holder"].eq("Klinikum Nord"),
            "patient_id",
        ]
    )
    medications = data["medications"].loc[
        data["medications"]["atc"].astype(str).str.startswith("R03BB")
    ]
    reference = pd.Timestamp("2025-12-31")
    active_lama = set(
        medications.loc[
            medications["start_date"].le(reference)
            & (
                medications["end_date"].isna()
                | medications["end_date"].ge(reference)
            ),
            "patient_id",
        ]
    )
    expected_ids = (copd & nord_patients) - active_lama
    medication = next(
        item for item in interpretation.criteria["population"]["items"]
        if item.get("type") == "medication"
    )

    assert medication["label"] == "LAMA"
    assert medication["exclude"] is True
    assert medication["status"] == "active"
    assert _count(response) == (len(expected_ids) if len(expected_ids) >= 10 else "<10")


def test_natural_language_display_names_for_bmi_lama_and_hba1c():
    bmi = interpret_question("Patients with BMI above 30")
    lama = interpret_question("Patients with COPD and LAMA")
    hba1c = interpret_question(
        "Compare HbA1c levels in patients with type 2 diabetes"
    )

    assert next(
        item for item in bmi.criteria["population"]["items"]
        if item.get("type") == "lab"
    )["label"] == "BMI"
    assert next(
        item for item in lama.criteria["population"]["items"]
        if item.get("type") == "medication"
    )["label"] == "LAMA"
    assert hba1c.criteria["metrics"][1]["label"] == "mean latest HbA1c (%)"


def test_loop_diuretic_and_bmi_distribution_have_no_unrecognised_words():
    loop_diuretic = interpret_question(
        "Compare the distribution of loop-diuretic use among individuals "
        "with COPD, by region."
    )
    bmi_distribution = interpret_question(
        "Show the BMI distribution across people with COPD, by region."
    )

    assert loop_diuretic.unrecognised == []
    assert bmi_distribution.unrecognised == []


def test_unknown_condition_is_not_invented():
    interpretation = interpret_question("Number of patients with sepsis by sex")

    assert not interpretation.ok()
    assert "sepsis" in interpretation.unrecognised
    assert interpretation.criteria["population"]["items"] == []
    assert interpretation.criteria["group_by"] == [{"dimension": "sex"}]


def test_diagnosis_ranges_and_prescription_periods_are_structured():
    diagnosed = interpret_question(
        "How many patients were diagnosed with asthma between 2023 and 2024?"
    )
    medication = interpret_question(
        "Patients with type 2 diabetes prescribed metformin during 2025"
    )

    asthma = diagnosed.criteria["population"]["items"][0]
    metformin = medication.criteria["population"]["items"][1]
    assert asthma["onset_start"] == "2023-01-01"
    assert asthma["onset_end"] == "2024-12-31"
    assert metformin["status"] == "during_period"
    assert metformin["period_start"] == "2025-01-01"
    assert metformin["period_end"] == "2025-12-31"


def test_all_official_questions_are_safe_to_interpret():
    for path in sorted((ROOT / "requests").glob("R*.json")):
        request = json.loads(path.read_text(encoding="utf-8"))
        interpretation = interpret_question(request.get("question", ""))
        assert interpretation is not None


def test_excluding_region_is_an_exclusion():
    from ehds_hdab.nl import interpret_question

    criteria = interpret_question(
        "How many women with breast cancer are there in Austria, excluding Burgenland?").criteria
    items = criteria["population"]["items"]
    region = next(item for item in items if item["type"] == "region")
    assert region["values"] == ["Burgenland"] and region.get("exclude") is True
    assert any(item["type"] == "sex" and item["value"] == "female" for item in items)
    assert any(item["type"] == "condition" and item["icd10"] == ["C50"] for item in items)
