"""Visualisation plan: deterministic chart selection, clinically labelled measures, privacy-safe rendering."""
import json

import pytest

from ehds_hdab.charts import (altair_chart, build_plan, display_table, is_suppressed, png_table_and_spec,
                              select_chart_type)
from ehds_hdab.criteria import criteria_to_definitions
from ehds_hdab.nl import interpret_question
from ehds_hdab.pipeline import process_request
from ehds_hdab.viz import interpret_chart_command, render_chart


def ask(text, rid="VIZ"):
    interpretation = interpret_question(text)
    assert interpretation.ok(), (text, interpretation.problems, interpretation.unrecognised)
    request = {"id": rid, "applicant": {"name": "Viz Tester", "organisation": "Uni"}, "country": "AT",
               "requested_as": "health data request", "requested_format": "anonymised statistics",
               "purpose": "Scientific research on chronic disease care", "question": text,
               "definitions": criteria_to_definitions(interpretation.criteria), "criteria": interpretation.criteria}
    resp = process_request(request)
    assert resp["decision"] == "APPROVE_STATISTICAL" and resp.get("answer"), text
    return resp, interpretation


def chart_values(chart):
    """Every value that can reach the browser (data rows of all layers)."""
    spec = json.loads(chart.to_json())
    values = []
    for layer in spec.get("layer", [spec]):
        values.extend(spec.get("datasets", {}).get(layer.get("data", {}).get("name"), []))
    return values


def test_changes_over_time_produce_a_line_graph():
    resp, _ = ask("Show changes in type 2 diabetes diagnoses over time.")
    plan = build_plan(resp, "patients with type 2 diabetes", "Per calendar year, 2023–2025")
    assert plan["chart_type"] == "line" and plan["x"] == "Year" and plan["time_axis"]
    assert plan["x_order"] == ["2023", "2024", "2025"]
    assert plan["measure_type"] == "count_patients"


def test_hospital_comparison_produces_a_bar_chart_of_prevalence():
    resp, interpretation = ask("Compare type 2 diabetes prevalence across hospitals.")
    plan = build_plan(resp, "patients with type 2 diabetes", "No time period applied.")
    assert plan["chart_type"] == "bar" and plan["x"] == "Hospital" and plan["allowed_types"] == ["bar"]
    assert plan["measure_type"] == "percentage" and plan["unit"] == "%"
    assert any(a.startswith("Prevalence =") for a in interpretation.assumptions)


def test_hba1c_trend_uses_latest_value_mean_and_units():
    resp, _ = ask("Show HbA1c trends in patients with type 2 diabetes")
    plan = build_plan(resp, "patients with type 2 diabetes", "Per calendar year, 2023–2025",
                      {"metric": "mean latest HbA1c (%)"})
    assert plan["metric"] == "mean latest HbA1c (%)" and plan["unit"] == "%" and plan["measure_type"] == "mean"
    assert "latest" in plan["explanation"] and "mean" in plan["explanation"]
    assert plan["chart_type"] == "line"


def test_multi_group_trend_has_one_series_per_sex_with_fixed_colours():
    resp, _ = ask("Compare HbA1c trends between men and women with type 2 diabetes.")
    plan = build_plan(resp, "patients", "Per calendar year, 2023–2025", {"metric": "mean latest HbA1c (%)"})
    assert plan["chart_type"] == "line" and plan["x"] == "Year" and plan["color"] == "Sex"
    assert set(plan["colors"]) == {"female", "male"} and plan["colors"]["female"] != plan["colors"]["male"]
    assert len(plan["data"]) == 6


def test_distribution_request_gives_counts_per_clinical_category():
    resp, interpretation = ask("Show the distribution of BMI among patients with type 2 diabetes.")
    plan = build_plan(resp, "patients with type 2 diabetes", "No time period applied.")
    assert plan["category_axis"] and plan["chart_type"] == "bar"
    assert plan["x_order"][:2] == ["<18.5 (underweight)", "18.5–24.9 (normal)"]
    released = plan["data"]["value"].dropna().tolist()
    assert released and sum(released) <= resp["answer"]["All"]["patients"]
    assert any("Individual values are never shown" in a for a in interpretation.assumptions)


def test_relationship_request_is_aggregated_not_a_scatter():
    resp, interpretation = ask("Show the relationship between BMI and HbA1c.")
    plan = build_plan(resp, "patients with measurements", "No time period applied.")
    assert plan["category_axis"] and plan["measure_type"] == "mean" and plan["x_label"] == "BMI category"
    assert any("not a patient-level scatter plot" in a for a in interpretation.assumptions)
    refused = interpret_chart_command("scatter plot of BMI against HbA1c", {}, ["patients"], [])
    assert not refused["ok"] and "secure processing environment" in refused["message"]


def test_hospitalisations_are_counted_as_stays_and_labelled():
    resp, interpretation = ask("How many heart failure hospitalizations occurred each year?")
    plan = build_plan(resp, "patients", "Per calendar year, 2023–2025")
    assert plan["metric"] == "inpatient stays for Heart failure" and plan["measure_type"] == "count_encounters"
    assert "once per stay" in plan["measure_text"]
    stays = resp["answer"]["2025"]["inpatient stays for Heart failure"]
    patients = resp["answer"]["2025"]["patients hospitalised"]
    assert stays >= patients


def test_suppressed_cells_never_reach_chart_tooltips_or_table():
    resp, _ = ask("Show the distribution of BMI among patients with type 2 diabetes.")
    plan = build_plan(resp, "patients with type 2 diabetes", "No time period applied.")
    assert plan["suppressed_cells"] >= 1
    rows = chart_values(altair_chart(plan))
    hidden = [r for r in rows if "marker" in r]
    assert hidden and all(r["marker"] == "<10" and r.get("zero") == 0 for r in hidden)
    assert all(r.get("value") in (None, "NaN") or not is_suppressed(r["shown"]) for r in rows)
    table = display_table(plan)
    assert "<10" in table.iloc[:, -1].tolist()
    frame, spec = png_table_and_spec(plan)
    assert "<10" in frame[plan["metric"]].tolist()
    render_chart(frame, spec, plan["title"], "pop", "period", "")


def test_missing_time_points_are_gaps_not_zeros():
    resp, _ = ask("Trends in new cases of depression since 2015")
    plan = build_plan(resp, "patients", "Per calendar year")
    assert plan["x_order"][0] == "2015" and plan["x_order"][-1] == "2025"
    assert all(v >= 10 for v in plan["data"]["value"].dropna())
    assert 0 not in plan["data"]["value"].dropna().tolist()


def test_switching_chart_types_only_reuses_released_values():
    resp, _ = ask("Compare HbA1c trends between men and women with type 2 diabetes.")
    line = build_plan(resp, "p", "t", {"metric": "mean latest HbA1c (%)"})
    bar = build_plan(resp, "p", "t", {"metric": "mean latest HbA1c (%)", "type": "bar"})
    assert line["chart_type"] == "line" and bar["chart_type"] == "bar"
    assert line["data"]["shown"].tolist() == bar["data"]["shown"].tolist()


def test_unsupported_chart_types_are_explained_with_an_alternative():
    chart_type, allowed, note = select_chart_type(True, False, "pie")
    assert chart_type == "line" and "pie" in note and "line graph" in note
    chart_type, allowed, note = select_chart_type(False, False, "line")
    assert chart_type == "bar" and "bar chart" in note
    result = interpret_chart_command("make this a pie chart", {}, ["patients"], ["Region"])
    assert result["refused_type"] == "pie"


def test_filter_and_table_commands_are_recognised():
    result = interpret_chart_command("Only patients aged 40-65", {}, ["patients"], ["Region"])
    assert result["filter"] == {"type": "age", "label": "age", "min": 40, "max": 65}
    assert interpret_chart_command("show a table instead", {}, ["patients"], ["Region"])["table"] is True
    assert interpret_chart_command("change the time interval to months", {}, ["patients"], ["Year"])["ok"] is False
    percent = interpret_chart_command("use percentages instead of absolute numbers", {}, ["patients"], ["Region"])
    assert not percent["ok"] and "prevalence" in percent["message"]


def test_graph_values_match_the_released_answer():
    resp, _ = ask("Compare type 2 diabetes prevalence across hospitals.")
    plan = build_plan(resp, "p", "t")
    for _, row in plan["data"].iterrows():
        assert row["shown"] == str(resp["answer"][row["Hospital"]]["prevalence of Type 2 diabetes (%)"])
