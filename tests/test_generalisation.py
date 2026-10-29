import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
import pytest

from ehds_hdab.pipeline import process_request
from ehds_hdab.spe import DownloadBlockedError, SecureProcessingEnvironment

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "tests" / "new_requests"
REFERENCE = pd.Timestamp("2025-12-31")


def _request(filename):
    return json.loads((NEW / filename).read_text(encoding="utf-8"))


def _source_data():
    patients = pd.read_csv(ROOT / "data" / "patients.csv", dtype=str)
    patients = patients.loc[patients["opted_out"].str.casefold() != "true"].copy()
    patients["birth_date"] = pd.to_datetime(patients["birth_date"])
    patients["age"] = REFERENCE.year - patients["birth_date"].dt.year
    patient_ids = set(patients["patient_id"])
    tables = {}
    for name in ("conditions", "medications", "observations", "encounters"):
        frame = pd.read_csv(ROOT / "data" / f"{name}.csv", dtype=str, keep_default_na=False)
        tables[name] = frame.loc[frame["patient_id"].isin(patient_ids)].copy()
    return patients, tables


def _values(group):
    node = group
    while isinstance(node, dict):
        node = next(iter(node.values()))
    return node


def _check_values(actual, expected_by_group):
    assert set(actual) == set(expected_by_group)
    for group, expected_values in expected_by_group.items():
        got = list(actual[group].values())
        for expected in expected_values:
            assert any(
                isinstance(expected, (int, float))
                and isinstance(value, (int, float))
                and abs(float(expected) - float(value)) < 1e-9
                or expected == value
                for value in got
            ), (group, expected, got)


def _small_count(value):
    return "<10" if value < 10 else int(value)


def _small_percentage(numerator, denominator):
    if denominator < 10 or numerator < 10:
        return "suppressed"
    return float((Decimal(numerator) * Decimal(100) / Decimal(denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def test_inpatient_stays_in_2025_by_hospital_match_direct_pandas():
    patients, tables = _source_data()
    encounters = tables["encounters"]
    encounters["start_date"] = pd.to_datetime(encounters["start_date"])
    selected = encounters.loc[
        encounters["type"].str.casefold().eq("inpatient")
        & encounters["primary_icd10"].str.startswith("I50")
        & encounters["start_date"].dt.year.eq(2025)
    ]
    expected = {str(k): [_small_count(int(v))] for k, v in selected.groupby("data_holder").size().items()}
    result = process_request(_request("a_inpatient_by_hospital.json"))
    assert result["decision"] == "APPROVE_STATISTICAL"
    _check_values(result["answer"], expected)


def test_asthma_active_salmeterol_fluticasone_by_sex_match_direct_pandas():
    patients, tables = _source_data()
    condition_ids = set(tables["conditions"].loc[tables["conditions"]["icd10"].str.startswith("J45"), "patient_id"])
    cohort = patients.loc[patients["age"].ge(18) & patients["patient_id"].isin(condition_ids)]
    meds = tables["medications"].copy()
    meds["start_date"] = pd.to_datetime(meds["start_date"])
    meds["end_date"] = pd.to_datetime(meds["end_date"], errors="coerce")
    active_ids = set(meds.loc[
        meds["atc"].str.startswith("R03AK")
        & meds["start_date"].le(REFERENCE)
        & (meds["end_date"].isna() | meds["end_date"].ge(REFERENCE)),
        "patient_id",
    ])
    expected = {}
    for sex in patients["sex"].unique():
        group = cohort.loc[cohort["sex"].eq(sex)]
        denominator = len(group)
        numerator = int(group["patient_id"].isin(active_ids).sum())
        expected[sex] = [
            _small_count(denominator),
            _small_percentage(numerator, denominator),
        ]
    result = process_request(_request("b_asthma_medication_by_sex.json"))
    _check_values(result["answer"], expected)


def test_heart_failure_no_code_definition_by_region_matches_direct_pandas():
    patients, tables = _source_data()
    condition_ids = set(tables["conditions"].loc[tables["conditions"]["icd10"].str.startswith("I50"), "patient_id"])
    expected = {}
    for region in patients["region"].unique():
        count = int((patients["region"].eq(region) & patients["patient_id"].isin(condition_ids)).sum())
        expected[region] = [_small_count(count)]
    result = process_request(_request("c_heart_failure_no_code.json"))
    _check_values(result["answer"], expected)


def test_mean_latest_systolic_pressure_by_age_band_matches_direct_pandas():
    patients, tables = _source_data()
    condition_ids = set(tables["conditions"].loc[tables["conditions"]["icd10"].str.startswith("I10"), "patient_id"])
    obs = tables["observations"].loc[tables["observations"]["loinc"].eq("8480-6")].copy()
    obs["date"] = pd.to_datetime(obs["date"])
    latest = obs.sort_values("date").drop_duplicates("patient_id", keep="last")
    eligible = patients.loc[patients["patient_id"].isin(condition_ids)].merge(latest, on="patient_id")
    expected = {}
    for low in range(0, 100, 10):
        high = low + 9 if low < 90 else None
        label = f"{low}-{high}" if high is not None else "90+"
        rows = eligible.loc[eligible["age"].ge(low) & (eligible["age"].le(high) if high is not None else True)]
        if len(rows) < 10:
            value = "suppressed"
        else:
            mean = sum(Decimal(v) for v in rows["value"]) / Decimal(len(rows))
            value = float(mean.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        expected[label] = [value]
    result = process_request(_request("d_systolic_bp_age_bands.json"))
    _check_values(result["answer"], expected)


def test_prohibited_tobacco_product_development_and_smoking_research_distinction():
    assert process_request(_request("e_tobacco_product_development.json"))["decision"] == "REJECT"
    assert process_request(_request("f_smoking_copd_public_health.json"))["decision"] == "APPROVE_STATISTICAL"


def test_heart_failure_spe_contains_only_the_cohort_and_blocks_downloads():
    response = process_request(_request("g_heart_failure_readmission_spe.json"))
    assert response["decision"] == "APPROVE_SPE"
    workspace = Path(response["spe"]["path"])
    assert set(response["spe"]["files"]) == {"cohort.csv", "conditions.csv", "medications.csv", "encounters.csv"}
    cohort = pd.read_csv(workspace / "cohort.csv", dtype=str)
    cohort_ids = set(cohort["patient_id"])
    for table in ("conditions", "medications", "encounters"):
        frame = pd.read_csv(workspace / f"{table}.csv", dtype=str)
        assert set(frame["patient_id"]).issubset(cohort_ids)
    spe = SecureProcessingEnvironment(workspace, request_id="NEW-G")
    assert set(spe.list_tables()) == {"cohort", "conditions", "medications", "encounters"}
    assert len(spe.read_table("cohort")) == response["answer"]["cohort_size"]
    with pytest.raises(DownloadBlockedError):
        spe.request_download("cohort.csv")
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["download_allowed"] is False
    assert manifest["cohort_size"] == response["answer"]["cohort_size"]
