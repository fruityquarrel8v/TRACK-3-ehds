import json

from ehds_hdab.criteria import criteria_to_definitions, describe_criteria
from ehds_hdab.data import load_data
from ehds_hdab.pipeline import process_request


def _criteria(excluded_age=None):
    items = [
        {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]},
        {"type": "age", "label": "Adults", "min": 18},
    ]
    if excluded_age is not None:
        items.append({
            "type": "age",
            "label": f"Age {excluded_age}",
            "min": excluded_age,
            "max": excluded_age,
            "exclude": True,
        })
    return {
        "population": {"match": "all", "items": items},
        "group_by": [],
        "metrics": [{"type": "count_patients", "label": "patients"}],
        "record_types": [],
    }


def _request(request_id, excluded_age=None, applicant=None):
    criteria = _criteria(excluded_age)
    return {
        "id": request_id,
        "applicant": applicant if applicant is not None else {
            "name": "Differencing Applicant",
            "organisation": "Public Health Institute",
        },
        "requested_as": "health data request",
        "requested_format": "anonymised statistics",
        "purpose": "Public health research on type 2 diabetes.",
        "question": describe_criteria(criteria),
        "definitions": criteria_to_definitions(criteria),
        "criteria": criteria,
    }


def _adult_t2d_ids(data):
    patients = data["patients"]
    conditions = data["conditions"]
    t2d = set(conditions.loc[
        conditions["icd10"].astype(str).str.startswith("E11"),
        "patient_id",
    ])
    adults = set(patients.loc[patients["age"].ge(18), "patient_id"])
    return adults & t2d


def test_small_population_difference_is_withheld_and_audited(tmp_path):
    data = load_data()
    patients = data["patients"]
    adults_with_t2d = _adult_t2d_ids(data)
    age_38 = set(patients.loc[patients["age"].eq(38), "patient_id"])
    excluded = adults_with_t2d & age_38
    assert 0 < len(excluded) < 10

    first = process_request(_request("DIFF-BASE"))
    second = process_request(_request(
        "DIFF-SMALL",
        excluded_age=38,
        applicant={
            "name": "differencing applicant",
            "organisation": "public health institute",
        },
    ))

    assert first["decision"] == "APPROVE_STATISTICAL"
    assert not first.get("withheld")
    assert second["withheld"] is True
    assert second["answer"] is None
    assert second["needs_staff_review"] is True
    reason = next(
        warning for warning in second["warnings"]
        if warning.startswith("Results withheld:")
    )
    assert "cannot be released safely" in reason

    audit_records = [
        json.loads(line)
        for line in (tmp_path / "audit_log.jsonl").read_text().splitlines()
    ]
    withheld_event = next(
        event for event in audit_records
        if event["event"] == "differencing_withheld"
    )
    assert withheld_event["details"]["earlier_request"] == "DIFF-BASE"
    assert "patient_id" not in json.dumps(withheld_event)


def test_difference_of_ten_or_more_is_not_withheld():
    data = load_data()
    patients = data["patients"]
    adults_with_t2d = _adult_t2d_ids(data)
    age_42 = set(patients.loc[patients["age"].eq(42), "patient_id"])
    assert len(adults_with_t2d & age_42) >= 10

    process_request(_request("DIFF-LARGE-BASE"))
    response = process_request(_request("DIFF-LARGE", excluded_age=42))

    assert not response.get("withheld")
    assert response["answer"] is not None


def test_other_applicants_results_count_as_released():
    process_request(_request("DIFF-OTHER-BASE"))
    response = process_request(_request(
        "DIFF-OTHER-SMALL",
        excluded_age=38,
        applicant={
            "name": "Another Applicant",
            "organisation": "Public Health Institute",
        },
    ))

    assert response.get("withheld") is True
    assert response["answer"] is None


def test_identical_rerun_is_allowed():
    process_request(_request("DIFF-IDENTICAL-1"))
    response = process_request(_request("DIFF-IDENTICAL-2"))

    assert not response.get("withheld")
    assert response["answer"] is not None


def test_missing_applicant_is_still_protected():
    process_request(_request("DIFF-NO-APPLICANT-1", applicant={}))
    response = process_request(_request(
        "DIFF-NO-APPLICANT-2",
        excluded_age=38,
        applicant={},
    ))

    assert response.get("withheld") is True
    assert response["answer"] is None
