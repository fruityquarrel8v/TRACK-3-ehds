import pandas as pd
import pytest

from ehds_hdab.config import ROOT
from ehds_hdab.criteria import plan_from_criteria
from ehds_hdab.data import load_data
from ehds_hdab.engine import execute
from ehds_hdab.nl import interpret_question
from ehds_hdab.pipeline import process_request
from ehds_hdab.privacy import apply_small_cell
from ehds_hdab.viz import interpret_chart_command

T2D = {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]}


def _request(criteria, rid="TR-1"):
    return {"id": rid, "applicant": {"name": "Trend Tester", "organisation": "Uni"}, "country": "AT",
            "requested_as": "health data request", "requested_format": "anonymised statistics",
            "purpose": "Scientific research on diabetes care", "question": "trend", "criteria": criteria}


@pytest.fixture(scope="module")
def independent():
    patients = pd.read_csv(ROOT / "data" / "patients.csv")
    conditions = pd.read_csv(ROOT / "data" / "conditions.csv", parse_dates=["onset_date"])
    observations = pd.read_csv(ROOT / "data" / "observations.csv", parse_dates=["date"])
    keep = set(patients.loc[~patients["opted_out"], "patient_id"])
    t2d = conditions[conditions["icd10"].astype(str).str.startswith("E11") & conditions["patient_id"].isin(keep)]
    out = {}
    for year in (2023, 2024, 2025):
        ids = set(t2d.loc[t2d["onset_date"] <= f"{year}-12-31", "patient_id"])
        obs = observations[(observations["loinc"] == "4548-4") & observations["patient_id"].isin(ids)
                           & (observations["date"] >= f"{year}-01-01") & (observations["date"] <= f"{year}-12-31")]
        latest = obs.sort_values("date").drop_duplicates("patient_id", keep="last")
        out[str(year)] = (len(ids), round(float(latest["value"].mean()), 1))
    return out


def test_year_grouping_matches_independent_pandas(independent):
    criteria = {"population": {"match": "all", "items": [T2D]},
                "group_by": [{"dimension": "year", "from": 2023, "to": 2025}],
                "metrics": [{"type": "count_patients", "label": "patients"},
                            {"type": "mean_latest_lab", "label": "mean HbA1c", "loinc": "4548-4"}]}
    plan = plan_from_criteria(criteria)
    assert plan.group_by[0]["dimension"] == "year" and not plan.warnings
    answer = apply_small_cell(execute(plan, load_data()))
    assert list(answer) == ["2023", "2024", "2025"]
    for year, (count, mean) in independent.items():
        assert answer[year]["patients"] == count
        assert float(answer[year]["mean HbA1c"]) == mean


def test_year_is_first_dimension_and_defaults_to_request_period():
    criteria = {"population": {"match": "all", "items": [dict(T2D, onset_start="2021-01-01", onset_end="2025-12-31")]},
                "group_by": [{"dimension": "sex"}, {"dimension": "year"}],
                "metrics": [{"type": "count_patients", "label": "patients"}]}
    plan = plan_from_criteria(criteria)
    assert [d["dimension"] for d in plan.group_by] == ["year", "sex"]
    assert (plan.group_by[0]["from"], plan.group_by[0]["to"]) == (2021, 2025)
    resp = process_request(_request(criteria))
    assert resp["decision"] == "APPROVE_STATISTICAL" and not resp.get("withheld")
    assert list(resp["answer"]) == ["2021", "2022", "2023", "2024", "2025"]
    assert set(resp["answer"]["2025"]) == {"female", "male"}


def test_new_cases_per_year_count_onsets_within_each_year():
    data = load_data()
    criteria = {"population": {"match": "all", "items": [dict(T2D, onset_start="2024-01-01", onset_end="2025-12-31")]},
                "group_by": [{"dimension": "year"}], "metrics": [{"type": "count_patients", "label": "patients"}]}
    answer = apply_small_cell(execute(plan_from_criteria(criteria), data))
    conditions = data["conditions"]
    for year in ("2024", "2025"):
        expected = conditions.loc[conditions["icd10"].astype(str).str.startswith("E11")
                                  & conditions["onset_date"].dt.year.eq(int(year)), "patient_id"].nunique()
        assert answer[year]["patients"] == expected


def test_invalid_year_grouping_is_flagged_not_executed():
    plan = plan_from_criteria({"population": {"match": "all", "items": [T2D]},
                               "group_by": [{"dimension": "year", "from": 2025, "to": 2020}],
                               "metrics": [{"type": "count_patients", "label": "patients"}]})
    assert not plan.group_by and any("year" in w for w in plan.warnings)


def test_question_with_trend_words_groups_by_year():
    interpretation = interpret_question("Show HbA1c trends in patients with type 2 diabetes")
    assert interpretation.ok()
    assert {"dimension": "year"} in interpretation.criteria["group_by"]
    assert not any("not available" in a for a in interpretation.assumptions)
    assert "trends" not in interpretation.unrecognised
    monthly = interpret_question("Monthly number of patients with asthma")
    assert {"dimension": "year"} in monthly.criteria["group_by"]
    assert any("Monthly" in a for a in monthly.assumptions)


def test_chart_commands_use_year_as_a_line_graph():
    dims, metrics = ["Sex"], ["patients"]
    result = interpret_chart_command("show this by year as a line graph", {}, metrics, dims)
    assert result["ok"] and result["regroup"] == ["year"] and result["spec"]["type"] == "line"
    result = interpret_chart_command("line graph over time", {}, metrics, dims)
    assert result["regroup"] == ["year"]
    result = interpret_chart_command("bar chart of patients by year", {}, metrics, ["Year"])
    assert result["ok"] and result["regroup"] is None and result["spec"] == {
        "type": "bar", "metric": "patients", "x": "Year", "color": None}
    refused = interpret_chart_command("plot monthly trends", {}, metrics, dims)
    assert not refused["ok"] and "monthly" in refused["message"].casefold()
