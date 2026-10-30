import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from ehds_hdab.data import load_data
from ehds_hdab.pipeline import process_request
from ehds_hdab.viz import answer_table, interpret_chart_command, plotted_data, render_chart

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def r01():
    resp = process_request(json.loads((ROOT / "requests" / "R01.json").read_text(encoding="utf-8")))
    table = answer_table(resp["answer"], resp["query_plan"]["group_by"])
    dims = ["Region"]
    metrics = [c for c in table.columns if c not in dims]
    return resp, table, dims, metrics


def test_bar_is_default_and_metric_chosen_by_word(r01):
    _, _, dims, metrics = r01
    result = interpret_chart_command("bar chart of prevalence by region", {}, metrics, dims)
    assert result["ok"] and result["regroup"] is None
    assert result["spec"] == {"type": "bar", "metric": "prevalence", "x": "Region", "color": None}


def test_line_refinement_keeps_metric(r01):
    _, _, dims, metrics = r01
    spec = {"type": "bar", "metric": "prevalence", "x": "Region", "color": None}
    result = interpret_chart_command("change this into a line graph", spec, metrics, dims)
    assert result["ok"] and result["spec"]["type"] == "line" and result["spec"]["metric"] == "prevalence"


def test_regroup_detected_when_dimension_missing(r01):
    _, _, dims, metrics = r01
    result = interpret_chart_command("group the results by sex and age", {}, metrics, dims)
    assert result["ok"] and result["regroup"] == ["sex", "age_band"]


@pytest.mark.parametrize("command", ["histogram of HbA1c", "scatter plot of BMI", "box plot by sex"])
def test_patient_level_charts_are_refused(r01, command):
    _, _, dims, metrics = r01
    result = interpret_chart_command(command, {}, metrics, dims)
    assert not result["ok"] and "secure processing environment" in result["message"]


def test_trends_and_unknown_commands(r01):
    _, _, dims, metrics = r01
    assert "Trends over time" in interpret_chart_command("plot monthly trends", {}, metrics, dims)["message"]
    unknown = interpret_chart_command("make it pretty", {}, metrics, dims)
    assert not unknown["ok"] and "Try" in unknown["message"]


def test_render_returns_png_and_hides_suppressed_cells():
    resp = process_request(json.loads((ROOT / "requests" / "R08.json").read_text(encoding="utf-8")))
    table = answer_table(resp["answer"], resp["query_plan"]["group_by"])
    metric = "mean_bmi"
    spec = {"type": "bar", "metric": metric, "x": "Region", "color": None}
    _, png = render_chart(table, spec, "t", "women with breast cancer", "none", "kg/m2")
    assert png.startswith(b"\x89PNG")
    plotted = plotted_data(table, spec)
    suppressed = table[metric].astype(str).isin(["<10", "suppressed"])
    assert plotted[metric].isna().tolist() == suppressed.tolist()
    assert suppressed.any()


def test_regroup_end_to_end_matches_pandas():
    request = {
        "id": "VIZ-1",
        "applicant": {"name": "A", "organisation": "Viz University"},
        "requested_as": "health data request",
        "requested_format": "anonymised statistics",
        "purpose": "Public health research on type 2 diabetes.",
        "question": "Patients with type 2 diabetes",
        "criteria": {"population": {"match": "all", "items": [
            {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]}]},
            "group_by": [], "metrics": [{"type": "count_patients", "label": "patients"}]},
    }
    first = process_request(request)
    assert first["decision"] == "APPROVE_STATISTICAL"
    regrouped = copy.deepcopy(request)
    regrouped["id"] = "VIZ-1-G1"
    bands = [{"lo": 18, "hi": 39, "label": "18-39"}, {"lo": 40, "hi": 64, "label": "40-64"},
             {"lo": 65, "hi": None, "label": "65+"}]
    regrouped["criteria"]["group_by"] = [{"dimension": "sex"}, {"dimension": "age_band", "bands": bands}]
    second = process_request(regrouped)
    assert second["decision"] == "APPROVE_STATISTICAL" and not second.get("withheld")
    data = load_data()
    t2d = data["patients"][data["patients"]["patient_id"].isin(
        data["conditions"].loc[data["conditions"]["icd10"].str.startswith("E11"), "patient_id"])]
    for sex in ("female", "male"):
        for band in bands:
            ages = t2d.loc[t2d["sex"] == sex, "age"]
            expected = int(((ages >= band["lo"]) & (ages <= (band["hi"] if band["hi"] is not None else 999))).sum())
            got = second["answer"][sex][band["label"]]["patients"]
            assert got == (expected if expected >= 10 else "<10")
