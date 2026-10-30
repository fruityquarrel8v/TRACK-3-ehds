import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
import pytest

from ehds_hdab.criteria import criteria_to_definitions, describe_criteria
from ehds_hdab.data import load_data
from ehds_hdab.parser import parse_request
from ehds_hdab.pipeline import process_request
from ehds_hdab.privacy import check_statistical_output


ROOT = Path(__file__).resolve().parents[1]
BASE = {
    "applicant": {"name": "Test Applicant", "organisation": "Public Health Institute", "country": "AT"},
    "requested_as": "health data request",
    "requested_format": "anonymised statistics",
}


def _request(criteria, request_id="CRITERIA", purpose="Public health research on chronic disease burden."):
    return {
        **BASE,
        "id": request_id,
        "purpose": purpose,
        "question": describe_criteria(criteria),
        "definitions": criteria_to_definitions(criteria),
        "criteria": criteria,
    }


def _condition_ids(data, prefix):
    rows = data["conditions"]
    return set(rows.loc[rows["icd10"].astype(str).str.startswith(prefix), "patient_id"])


def _latest_lab_ids(data, loinc, op, threshold=None, start=None, end=None):
    rows = data["observations"]
    rows = rows.loc[rows["loinc"].eq(loinc)].copy()
    if start is not None:
        rows = rows.loc[rows["date"].ge(pd.Timestamp(start))]
    if end is not None:
        rows = rows.loc[rows["date"].le(pd.Timestamp(end))]
    rows = rows.sort_values("date").drop_duplicates("patient_id", keep="last")
    values = pd.to_numeric(rows["value"], errors="coerce")
    if op == "present":
        return set(rows["patient_id"])
    if op == ">":
        mask = values.gt(threshold)
    elif op == ">=":
        mask = values.ge(threshold)
    elif op == "<":
        mask = values.lt(threshold)
    else:
        mask = values.le(threshold)
    return set(rows.loc[mask, "patient_id"])


def _all_count(response, label="patients"):
    return response["answer"]["All"][label]


def _criteria(items, match="all", group_by=None, metrics=None, record_types=None):
    return {
        "population": {"match": match, "items": items},
        "group_by": group_by or [],
        "metrics": metrics or [{"type": "count_patients", "label": "patients"}],
        "record_types": record_types or [],
    }


@pytest.mark.parametrize(
    ("operator", "threshold", "start", "end", "anchor"),
    [
        (">", 8.0, None, None, 251),
        (">", 8.0, "2025-01-01", "2025-12-31", 155),
        (">=", 8.0, None, None, 282),
    ],
)
def test_t2d_latest_hba1c_thresholds_match_pandas(operator, threshold, start, end, anchor):
    data = load_data()
    t2d = _condition_ids(data, "E11")
    lab_ids = _latest_lab_ids(
        data,
        "4548-4",
        operator,
        threshold,
        start=start,
        end=end,
    )
    expected = len(t2d & lab_ids)
    item = {
        "type": "lab",
        "label": "HbA1c",
        "loinc": "4548-4",
        "op": operator,
        "value": threshold,
    }
    if start:
        item.update({"period_start": start, "period_end": end})
    criteria = _criteria([
        {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]},
        item,
    ])

    response = process_request(_request(criteria, request_id=f"LAB-{operator}-{start or 'all'}"))

    assert expected == anchor
    assert _all_count(response) == expected


@pytest.mark.parametrize(("status", "anchor"), [("active", 885), ("ever", 970)])
def test_adult_t2d_metformin_status_matches_pandas(status, anchor):
    data = load_data()
    patients = data["patients"]
    adults = set(patients.loc[patients["age"].ge(18), "patient_id"])
    t2d = _condition_ids(data, "E11")
    rows = data["medications"]
    rows = rows.loc[rows["atc"].astype(str).str.startswith("A10BA02")].copy()
    if status == "active":
        reference = pd.Timestamp("2025-12-31")
        rows = rows.loc[
            rows["start_date"].le(reference)
            & (rows["end_date"].isna() | rows["end_date"].ge(reference))
        ]
    expected = len(adults & t2d & set(rows["patient_id"]))
    criteria = _criteria([
        {"type": "age", "label": "Adults", "min": 18},
        {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]},
        {
            "type": "medication",
            "label": "Metformin",
            "atc": ["A10BA02"],
            "status": status,
        },
    ])

    response = process_request(_request(criteria, request_id=f"MET-{status}"))

    assert expected == anchor
    assert _all_count(response) == expected


def test_hospital_stay_uses_inpatient_primary_diagnosis_and_period():
    data = load_data()
    encounters = data["encounters"]
    rows = encounters.loc[
        encounters["type"].str.casefold().eq("inpatient")
        & encounters["primary_icd10"].astype(str).str.startswith("I50")
        & encounters["start_date"].between("2025-01-01", "2025-12-31")
    ]
    expected = rows["patient_id"].nunique()
    criteria = _criteria([{
        "type": "hospital_stay",
        "label": "Heart failure",
        "icd10": ["I50"],
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
    }])

    response = process_request(_request(criteria, request_id="HOSPITAL-STAY"))

    assert expected == 103
    assert _all_count(response) == expected


def test_any_population_is_a_pandas_union():
    data = load_data()
    expected = len(_condition_ids(data, "E11") | _condition_ids(data, "I10"))
    criteria = _criteria([
        {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]},
        {"type": "condition", "label": "Hypertension", "icd10": ["I10"]},
    ], match="any")

    response = process_request(_request(criteria, request_id="ANY-GROUP"))

    assert _all_count(response) == expected


def test_excluded_active_medication_matches_pandas_set_difference():
    data = load_data()
    t2d = _condition_ids(data, "E11")
    rows = data["medications"].loc[
        data["medications"]["atc"].astype(str).str.startswith("A10BA02")
    ]
    reference = pd.Timestamp("2025-12-31")
    active = rows.loc[
        rows["start_date"].le(reference)
        & (rows["end_date"].isna() | rows["end_date"].ge(reference))
    ]
    expected = len(t2d - set(active["patient_id"]))
    criteria = _criteria([
        {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]},
        {
            "type": "medication",
            "label": "Metformin",
            "atc": ["A10BA02"],
            "status": "active",
            "exclude": True,
        },
    ])

    response = process_request(_request(criteria, request_id="EXCLUDE-MED"))

    assert _all_count(response) == expected


def test_nested_groups_match_pandas():
    data = load_data()
    patients = data["patients"]
    age_ids = set(
        patients.loc[patients["age"].between(40, 65, inclusive="both"), "patient_id"]
    )
    expected = len(
        age_ids
        & (_condition_ids(data, "E11") | _condition_ids(data, "E66"))
    )
    criteria = _criteria([{
        "match": "all",
        "items": [
            {"type": "age", "label": "Age 40 to 65", "min": 40, "max": 65},
            {
                "match": "any",
                "items": [
                    {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]},
                    {"type": "condition", "label": "Obesity", "icd10": ["E66"]},
                ],
            },
        ],
    }])

    response = process_request(_request(criteria, request_id="NESTED-GROUP"))

    assert _all_count(response) == expected


def _rounded_mean(values):
    mean = sum((Decimal(str(value)) for value in values), Decimal(0)) / Decimal(len(values))
    return float(mean.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def test_mean_latest_hba1c_in_2025_by_sex_matches_pandas():
    data = load_data()
    t2d = _condition_ids(data, "E11")
    observations = data["observations"]
    latest = observations.loc[
        observations["loinc"].eq("4548-4")
        & observations["date"].between("2025-01-01", "2025-12-31")
    ].sort_values("date").drop_duplicates("patient_id", keep="last")
    latest["numeric"] = pd.to_numeric(latest["value"], errors="coerce")
    latest = latest.loc[latest["patient_id"].isin(t2d) & latest["numeric"].notna()]
    patients = data["patients"].set_index("patient_id")
    expected = {}
    for sex in ("female", "male"):
        values = latest.loc[
            latest["patient_id"].map(patients["sex"]).eq(sex), "numeric"
        ].tolist()
        expected[sex] = _rounded_mean(values)
    criteria = _criteria(
        [{"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]}],
        group_by=[{"dimension": "sex"}],
        metrics=[
            {"type": "count_patients", "label": "patients"},
            {
                "type": "mean_latest_lab",
                "label": "mean latest HbA1c (%)",
                "loinc": "4548-4",
                "period_start": "2025-01-01",
                "period_end": "2025-12-31",
            },
        ],
    )

    response = process_request(_request(criteria, request_id="MEAN-LAB"))

    assert response["answer"]["female"]["mean latest HbA1c (%)"] == expected["female"]
    assert response["answer"]["male"]["mean latest HbA1c (%)"] == expected["male"]


def test_rare_condition_grouping_is_small_cell_safe():
    data = load_data()
    cases = _condition_ids(data, "C50")
    patients = data["patients"]
    male_cases = set(patients.loc[patients["sex"].eq("male"), "patient_id"]) & cases
    demographics = patients.set_index("patient_id")[["region", "sex"]]
    raw_counts = (
        demographics.loc[list(male_cases)]
        .groupby(["region", "sex"])
        .size()
        .to_dict()
    )
    criteria = _criteria(
        [
            {"type": "condition", "label": "Breast cancer", "icd10": ["C50"]},
            {"type": "sex", "label": "male", "value": "male"},
        ],
        group_by=[{"dimension": "region"}, {"dimension": "sex"}],
    )

    response = process_request(_request(criteria, request_id="RARE-GROUP"))
    displayed = []
    for region, region_group in response["answer"].items():
        for sex, sex_group in region_group.items():
            raw = raw_counts.get((region, sex), 0)
            expected = "<10" if raw < 10 else raw
            assert sex_group["patients"] == expected
            displayed.append(sex_group["patients"])

    assert "<10" in displayed
    assert response["privacy_check"]["passed"] is True
    assert check_statistical_output(response["answer"]) == []


def test_malformed_criterion_warns_without_crashing():
    criteria = _criteria([{"type": "unknown_kind", "label": "unrecognised"}])

    response = process_request(_request(criteria, request_id="BAD-CRITERION"))

    assert any("Could not apply criterion 'unrecognised'" in warning for warning in response["warnings"])
    assert response["needs_staff_review"] is True


def test_empty_population_requires_staff_review():
    response = process_request(_request(_criteria([]), request_id="EMPTY-POPULATION"))

    assert response["needs_staff_review"] is True
    assert any("No applicable population criteria" in warning for warning in response["warnings"])


def test_invalid_criteria_values_are_skipped_with_warnings():
    criteria = _criteria([
        {"type": "condition", "label": "Missing code", "icd10": []},
        {"type": "lab", "label": "Bad lab", "loinc": "4548-4", "op": "==", "value": 8},
        {"type": "lab", "label": "Bad value", "loinc": "4548-4", "op": ">", "value": "eight"},
        {
            "type": "hospital_stay",
            "label": "Bad date",
            "period_start": "2025-02-30",
            "period_end": "2025-12-31",
        },
    ])

    plan = parse_request({
        **BASE,
        "question": "A structured criteria request.",
        "criteria": criteria,
    })

    assert len(plan.warnings) == 4
    assert all("Could not apply criterion" in warning for warning in plan.warnings)


def test_reversed_condition_onset_dates_warn_without_crashing():
    criteria = _criteria([{
        "type": "condition",
        "label": "Asthma",
        "icd10": ["J45"],
        "onset_start": "2025-12-31",
        "onset_end": "2025-01-01",
    }])

    response = process_request(_request(criteria, request_id="REVERSED-ONSET"))

    assert any("Could not apply criterion 'Asthma'" in warning for warning in response["warnings"])
    assert response["needs_staff_review"] is True


@pytest.mark.parametrize("status", [None, "unknown"])
def test_missing_or_unknown_medication_status_defaults_active_and_warns(status):
    data = load_data()
    t2d = _condition_ids(data, "E11")
    rows = data["medications"].loc[
        data["medications"]["atc"].astype(str).str.startswith("A10BA02")
    ]
    reference = pd.Timestamp("2025-12-31")
    active = rows.loc[
        rows["start_date"].le(reference)
        & (rows["end_date"].isna() | rows["end_date"].ge(reference))
    ]
    expected = len(t2d & set(active["patient_id"]))
    medication = {
        "type": "medication",
        "label": "Metformin",
        "atc": ["A10BA02"],
    }
    if status is not None:
        medication["status"] = status
    criteria = _criteria([medication, dict(medication)])

    response = process_request(_request(criteria, request_id=f"STATUS-{status or 'missing'}"))

    assert _all_count(response) == expected
    medication_warnings = [
        warning for warning in response["warnings"]
        if "counted as active on 31 December 2025" in warning
    ]
    assert len(medication_warnings) == 1
    assert response["needs_staff_review"] is True


def test_lab_present_matches_any_measurement_in_the_period():
    data = load_data()
    observations = data["observations"]
    rows = observations.loc[
        observations["loinc"].eq("4548-4")
        & observations["date"].between(
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31"),
        )
    ]
    expected = rows["patient_id"].nunique()
    criteria = _criteria([{
        "type": "lab",
        "label": "HbA1c",
        "loinc": "4548-4",
        "op": "present",
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
    }])

    response = process_request(_request(criteria, request_id="LAB-PRESENT"))

    assert _all_count(response) == expected


def test_medication_during_period_matches_pandas_overlap():
    data = load_data()
    rows = data["medications"].loc[
        data["medications"]["atc"].astype(str).str.startswith("A10BA02")
    ]
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")
    expected = rows.loc[
        rows["start_date"].le(end)
        & (rows["end_date"].isna() | rows["end_date"].ge(start)),
        "patient_id",
    ].nunique()
    criteria = _criteria([{
        "type": "medication",
        "label": "Metformin",
        "atc": ["A10BA02"],
        "status": "during_period",
        "period_start": start.date().isoformat(),
        "period_end": end.date().isoformat(),
    }])

    response = process_request(_request(criteria, request_id="DURING-PERIOD"))

    assert _all_count(response) == expected


def test_builder_condition_warning_recurses_through_groups():
    criteria = _criteria([
        {
            "match": "any",
            "items": [
                {"type": "condition", "label": "Pneumonia", "icd10": ["J18"]},
                {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]},
            ],
        },
    ])

    response = process_request(_request(criteria, request_id="GROUP-WARNING"))

    assert any("encounter main diagnosis" in warning for warning in response["warnings"])
    assert response["needs_staff_review"] is True


def test_builder_prohibited_purpose_is_rejected_under_article_54():
    criteria = _criteria([
        {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]},
    ])

    response = process_request(_request(
        criteria,
        request_id="PROHIBITED-PURPOSE",
        purpose="Insurance premium pricing for people with diabetes.",
    ))

    assert response["decision"] == "REJECT"
    assert "54" in response["articles"]


def test_builder_spe_cohort_size_matches_pandas():
    data = load_data()
    adults = set(data["patients"].loc[data["patients"]["age"].ge(18), "patient_id"])
    expected = len(adults & _condition_ids(data, "E11"))
    criteria = _criteria(
        [
            {"type": "age", "label": "Adults", "min": 18},
            {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]},
        ],
        record_types=["conditions", "medications"],
    )
    request = json.loads((ROOT / "requests" / "R05.json").read_text(encoding="utf-8"))
    request["id"] = "CRITERIA-SPE"
    request["criteria"] = criteria

    response = process_request(request)

    assert response["decision"] == "APPROVE_SPE"
    assert response["answer"]["cohort_size"] == expected


def test_describe_criteria_includes_labels_codes_and_exclusion():
    criteria = _criteria([
        {"type": "age", "label": "Age", "min": 40, "max": 65},
        {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]},
        {
            "type": "medication",
            "label": "Metformin",
            "atc": ["A10BA02"],
            "status": "active",
            "exclude": True,
        },
    ])

    description = describe_criteria(criteria)

    assert "Type 2 diabetes" in description
    assert "E11" in description
    assert "A10BA02" in description
    assert "without" in description


def test_describe_condition_onset_timing_with_human_readable_dates():
    cases = [
        (
            {"onset_start": "2025-01-01", "onset_end": "2025-12-31"},
            "newly diagnosed in 2025",
        ),
        (
            {"onset_start": "2023-01-01", "onset_end": "2024-12-31"},
            "newly diagnosed from 1 January 2023 to 31 December 2024",
        ),
        (
            {"onset_end": "2025-12-31"},
            "diagnosed on or before 31 December 2025",
        ),
        (
            {"onset_start": "2024-01-01"},
            "diagnosed since 1 January 2024",
        ),
    ]
    for timing, expected in cases:
        description = describe_criteria(_criteria([{
            "type": "condition",
            "label": "Depression",
            "icd10": ["F32"],
            **timing,
        }]))
        assert expected in description


def test_describe_lab_without_unit_and_hospital_stay_without_diagnosis():
    criteria = _criteria([
        {
            "type": "lab",
            "label": "BMI",
            "loinc": "39156-5",
            "op": ">",
            "value": 30,
        },
        {
            "type": "hospital_stay",
            "label": "Any main diagnosis",
            "period_start": "2025-01-01",
            "period_end": "2025-12-31",
        },
    ])

    description = describe_criteria(criteria)

    assert "above 30" in description
    assert "30 %" not in description
    assert "with an inpatient stay in 2025" in description
    assert "for Any main diagnosis" not in description
