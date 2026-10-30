"""Cumulative ('documented diagnosis by the end of each year') versus newly recorded ('newly diagnosed during each
year') must be interpreted, labelled and computed differently. Reference values come from pandas on the dataset."""
import pandas as pd
import pytest

from ehds_hdab.config import DATA_DIR
from ehds_hdab.criteria import criteria_to_definitions
from ehds_hdab.nl import interpret_question
from ehds_hdab.pipeline import process_request

CUMULATIVE_Q = ("Show the number of patients with a documented type 2 diabetes diagnosis by the end of each year from "
                "2023 to 2025 as a line graph.")
NEW_Q = "Show the number of patients newly diagnosed with type 2 diabetes during each year from 2023 to 2025."
AMBIGUOUS_Q = "Show patients diagnosed with type 2 diabetes per year from 2023 to 2025."


def _reference():
    patients = pd.read_csv(DATA_DIR / "patients.csv", dtype=str)
    kept = set(patients.loc[patients["opted_out"].str.lower() != "true", "patient_id"])
    cond = pd.read_csv(DATA_DIR / "conditions.csv", dtype=str)
    t2d = cond[cond["icd10"].str.startswith("E11") & cond["patient_id"].isin(kept)].copy()
    t2d["onset"] = pd.to_datetime(t2d["onset_date"])
    first = t2d.groupby("patient_id")["onset"].min()
    cumulative = {str(y): int((first <= pd.Timestamp(f"{y}-12-31")).sum()) for y in (2023, 2024, 2025)}
    new = {str(y): int(((first >= pd.Timestamp(f"{y}-01-01")) & (first <= pd.Timestamp(f"{y}-12-31"))).sum())
           for y in (2023, 2024, 2025)}
    return cumulative, new


def _run(question, criteria=None):
    interp = interpret_question(question)
    criteria = criteria or interp.criteria
    return interp, process_request({
        "id": "TIMING", "applicant": {"name": "t", "organisation": "Uni", "country": "AT"},
        "requested_as": "health data request", "purpose": "Scientific research on diabetes epidemiology.",
        "question": question, "definitions": criteria_to_definitions(criteria), "criteria": criteria})


def test_documented_by_year_end_is_cumulative_and_labelled_so():
    cumulative, new = _reference()
    interp, resp = _run(CUMULATIVE_Q)
    item = interp.criteria["population"]["items"][0]
    assert not (item.get("onset_start") and item.get("onset_end")), "a date range must not become a new-diagnosis window"
    assert interp.criteria["group_by"][0]["dimension"] == "year"
    assert interp.clarifications == []
    assert any("cumulative" in a.casefold() for a in interp.assumptions)
    label = next(m["label"] for m in interp.criteria["metrics"] if m["type"] == "count_patients")
    assert "documented" in label.casefold() and "new" not in label.casefold()
    assert resp["decision"] == "APPROVE_STATISTICAL"
    assert {year: cell[label] for year, cell in resp["answer"].items()} == cumulative
    assert cumulative != new


def test_newly_diagnosed_during_each_year_is_incidence_and_labelled_so():
    cumulative, new = _reference()
    interp, resp = _run(NEW_Q)
    item = interp.criteria["population"]["items"][0]
    assert item.get("onset_start") and item.get("onset_end")
    assert interp.clarifications == []
    label = next(m["label"] for m in interp.criteria["metrics"] if m["type"] == "count_patients")
    assert "newly" in label.casefold()
    assert {year: cell[label] for year, cell in resp["answer"].items()} == new


def test_new_cases_per_year_without_an_explicit_range_is_still_incidence():
    _, new = _reference()
    interp, resp = _run("New cases of type 2 diabetes per year.")
    label = next(m["label"] for m in interp.criteria["metrics"] if m["type"] == "count_patients")
    assert "newly" in label.casefold()
    assert {year: cell[label] for year, cell in resp["answer"].items()} == new


def test_ambiguous_diagnosed_per_year_asks_and_both_answers_work():
    cumulative, new = _reference()
    interp = interpret_question(AMBIGUOUS_Q)
    question = next(q for q in interp.clarifications if q["id"] == "diagnosis_timing")
    modes = {dict(question["options"])[label]["diagnosis_timing"]: label for label, _ in question["options"]}
    assert set(modes) == {"cumulative", "new"}
    from ehds_hdab.nl import apply_diagnosis_timing
    for mode, expected in (("cumulative", cumulative), ("new", new)):
        criteria = apply_diagnosis_timing(interpret_question(AMBIGUOUS_Q).criteria, mode)
        _, resp = _run(AMBIGUOUS_Q, criteria)
        label = next(m["label"] for m in criteria["metrics"] if m["type"] == "count_patients")
        assert {year: cell[label] for year, cell in resp["answer"].items()} == expected


def test_non_trend_single_year_diagnosed_in_keeps_its_meaning():
    interp = interpret_question("How many patients were newly diagnosed with type 2 diabetes in 2025?")
    item = interp.criteria["population"]["items"][0]
    assert item["onset_start"] == "2025-01-01" and item["onset_end"] == "2025-12-31"
    assert interp.criteria["group_by"] == []
