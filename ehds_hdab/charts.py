"""Structured visualisation plans built only from approved, privacy-checked results.

The chart layer never queries patient data. It receives the released answer (already passed through
the small-cell rule and the disclosure ledger), decides deterministically which chart communicates it
correctly, and renders it with Altair (interactive) or matplotlib (PNG download). Suppressed cells
("<10", "suppressed") are never plotted as numbers: they appear as gaps with a text marker, and tooltips
and the data table show the marker, not a value.
"""
import re

import altair as alt
import pandas as pd

from .config import SMALL_CELL_THRESHOLD
from .viz import DIMENSION_LABELS, answer_table

TIME_DIMENSION = "Year"
SUPPRESSED = {"<10", "suppressed", f"<{SMALL_CELL_THRESHOLD}"}
from .theme import CHART_PALETTE as PALETTE, CHART_TEXT, FIXED_COLOURS
REFUSED_TYPES = {
    "pie": "a pie chart hides small differences and cannot show time or more than a handful of groups",
    "donut": "a donut chart hides small differences and cannot show time or more than a handful of groups",
    "area": "a stacked area chart would imply the groups add up to a total, which these statistics do not",
    "3d": "3-D effects distort the comparison of values",
    "scatter": "a scatter plot needs one point per patient, which may not leave the secure processing environment",
    "box": "a box plot needs individual patients' values, which may not leave the secure processing environment",
    "histogram": "a histogram of raw values needs individual measurements; the app shows counts per clinical "
                 "category instead when you ask for a distribution",
}
MEASURE_TEXT = {
    "count_patients": "number of distinct patients",
    "count_encounters": "number of inpatient stays (a patient with several stays is counted once per stay)",
    "percentage": "percentage of the base population",
    "mean": "mean of each patient's latest value",
}


def is_suppressed(value):
    return isinstance(value, str) and value.strip().casefold() in SUPPRESSED


def _numeric(value):
    if is_suppressed(value) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unit_of(label):
    match = re.search(r"\(([^()]+)\)\s*$", str(label))
    return match.group(1) if match else ""


def _metric_info(resp):
    """Released metric columns with their type/category metadata from the approved query plan."""
    plan = resp.get("query_plan") or {}
    info = {}
    for metric in plan.get("metrics", []):
        info[metric.get("label")] = metric
    return info


def plan_inputs(resp):
    group_by = (resp.get("query_plan") or {}).get("group_by", [])
    dims = [DIMENSION_LABELS.get(d.get("dimension"), d.get("dimension")) for d in group_by]
    table = answer_table(resp.get("answer") or {}, group_by)
    info = _metric_info(resp)
    metrics = [c for c in table.columns if c not in dims]
    series = {}
    for column in metrics:
        meta = info.get(column) or {}
        if meta.get("series"):
            series.setdefault(meta["series"], []).append((column, meta.get("category"), meta.get("type")))
    return table, dims, metrics, info, series


def select_chart_type(x_is_time, category_axis, requested=None):
    """Deterministic chart choice. Returns (type, allowed_types, note)."""
    allowed = ["line", "bar"] if x_is_time else ["bar"]
    if requested in REFUSED_TYPES:
        suggestion = "line graph" if x_is_time else "bar chart"
        return allowed[0], allowed, (f"You asked for a {requested} chart, but {REFUSED_TYPES[requested]}. "
                                     f"A {suggestion} is shown instead.")
    if requested == "line" and not x_is_time:
        return "bar", allowed, ("A line graph implies a continuous order between categories; these groups are not "
                                "ordered in time, so a bar chart is shown. Ask for results by year to get a line graph.")
    if requested in allowed:
        return requested, allowed, ""
    return allowed[0], allowed, ""


def build_plan(resp, population_text, period_text, overrides=None):
    """A structured, validated description of the chart for one approved result."""
    overrides = dict(overrides or {})
    table, dims, metrics, info, series = plan_inputs(resp)
    if table.empty or not metrics:
        return None
    x_is_time = TIME_DIMENSION in dims
    category_axis = bool(series)

    if category_axis:
        series_name = overrides.get("metric") if overrides.get("metric") in series else next(iter(series))
        columns = series[series_name]
        order = [category for _, category, _ in columns]
        measure_type = columns[0][2]
        colour = dims[0] if dims else None
        rows = []
        for _, row in table.iterrows():
            for column, category, _ in columns:
                shown = row[column]
                rows.append({"Category": category, "value": _numeric(shown), "shown": str(shown),
                             **({colour: row[colour]} if colour else {})})
        data = pd.DataFrame(rows)
        first_column = columns[0][0]
        if " | " in str(first_column):
            # stratified mean: "<outcome> | <by-variable> <category>" -> x axis is the by-variable
            by_name = str(first_column).split(" | ", 1)[1].replace(str(columns[0][1]), "").strip()
            x_label = f"{by_name} category"
        else:
            x_label = series_name
        x = "Category"
        metric_label = ("patients" if measure_type == "count_patients" else series_name)
        metric_options = list(series)
    else:
        preferred = next((m for kind in ("percentage", "mean", "count_encounters") for m in metrics
                          if (info.get(m) or {}).get("type") == kind), metrics[0])
        metric_label = overrides.get("metric") if overrides.get("metric") in metrics else preferred
        measure_type = (info.get(metric_label) or {}).get("type", "count_patients")
        x = overrides.get("x") if overrides.get("x") in dims else (TIME_DIMENSION if x_is_time else (dims[0] if dims else None))
        colour = overrides.get("color") if overrides.get("color") in dims and overrides.get("color") != x else (
            next((d for d in dims if d != x), None))
        order = list(dict.fromkeys(table[x].astype(str))) if x else ["All"]
        rows = []
        for _, row in table.iterrows():
            shown = row[metric_label]
            rows.append({"x": str(row[x]) if x else "All", "value": _numeric(shown), "shown": str(shown),
                         **({colour: row[colour]} if colour else {})})
        data = pd.DataFrame(rows).rename(columns={"x": x or "Result"})
        x, x_label = (x or "Result"), (x or "")
        metric_options = metrics

    chart_type, allowed, note = select_chart_type(x == TIME_DIMENSION, category_axis, overrides.get("type"))
    unit = _unit_of(metric_label) or ("%" if measure_type == "percentage" else "")
    y_label = metric_label if not unit or unit in metric_label else f"{metric_label} ({unit})"
    colour_values = list(dict.fromkeys(data[colour].astype(str))) if colour else []
    colours = {value: FIXED_COLOURS.get(value.casefold(), PALETTE[index % len(PALETTE)])
               for index, value in enumerate(colour_values)}
    suppressed = int(data["shown"].map(is_suppressed).sum())
    title = f"{metric_label} by {x_label}" if x_label and x != "Result" else metric_label
    if colour:
        title += f" and {colour}"
    plan = {
        "chart_type": chart_type, "allowed_types": allowed, "selection_note": note,
        "x": x, "x_label": x_label, "x_order": order, "time_axis": x == TIME_DIMENSION,
        "color": colour, "colors": colours, "metric": metric_label, "metric_options": metric_options,
        "measure_type": measure_type, "measure_text": MEASURE_TEXT.get(measure_type, measure_type),
        "unit": unit, "y_label": y_label, "title": title, "category_axis": category_axis,
        "population": population_text, "period": period_text,
        "time_grouping": "calendar year" if x_is_time else None,
        "suppressed_cells": suppressed, "data": data,
        "privacy": (f"Counts below {SMALL_CELL_THRESHOLD} are shown as '<{SMALL_CELL_THRESHOLD}' and plotted as gaps, "
                    "never as zero; percentages and means based on fewer than 10 patients are suppressed. "
                    "Patients who opted out are excluded. Tooltips and the data table show only released values."),
    }
    plan["explanation"] = explanation(plan)
    return plan


def explanation(plan):
    measure = plan["measure_text"]
    if plan["measure_type"] == "mean":
        measure += (" (where a patient had several measurements in the same year, the latest was used)"
                    if plan["time_axis"] else " (within the period if one was given)")
    if plan["measure_type"] == "percentage":
        measure += (" (patients with the recorded diagnosis divided by all registered, non-opted-out patients with "
                    "the same demographic filters)")
    if plan["category_axis"]:
        measure = (f"{measure} per {plan['x_label'].split(' (')[0]}, using each patient's latest value")
    parts = [f"Shows the {measure} among {plan['population'].rstrip('.')}."]
    if plan["time_axis"]:
        timing = ("diagnoses first recorded within that year (newly recorded patients)" if "newly" in str(plan).casefold()
                  else "diagnoses recorded by 31 December of that year (cumulative, documented patients)")
        parts.append(f"Each calendar year is evaluated separately: {timing}, medication status on 31 December, the "
                     "latest laboratory value "
                     "within the year and hospital stays within the year; age is as of 31 December 2025.")
    elif plan["period"] and plan["period"] != "No time period applied.":
        parts.append(f"Time period: {plan['period']}.")
    if plan["color"]:
        parts.append(f"Separate series per {plan['color'].casefold()}; colours are fixed per group across charts.")
    if plan["suppressed_cells"]:
        parts.append(f"{plan['suppressed_cells']} cell(s) are hidden by the small-cell rule and appear as gaps.")
    parts.append("Descriptive statistics on synthetic data; differences between groups or years are not evidence "
                 "of causation.")
    return " ".join(parts)


def altair_chart(plan):
    data = plan["data"].copy()
    x, colour = plan["x"], plan["color"]
    x_enc = alt.X(f"{x}:N", sort=plan["x_order"], title=plan["x_label"] or None,
                  axis=alt.Axis(labelAngle=0 if len(plan["x_order"]) <= 6 else -30))
    y_enc = alt.Y("value:Q", title=plan["y_label"], scale=alt.Scale(zero=True))
    tooltip = [alt.Tooltip(f"{x}:N", title=plan["x_label"] or "Group")]
    if colour:
        tooltip.append(alt.Tooltip(f"{colour}:N", title=colour))
    tooltip.append(alt.Tooltip("shown:N", title=plan["metric"]))
    encodings = {"x": x_enc, "y": y_enc, "tooltip": tooltip}
    if colour:
        domain = list(plan["colors"])
        encodings["color"] = alt.Color(f"{colour}:N", scale=alt.Scale(domain=domain, range=[plan["colors"][d] for d in domain]),
                                       legend=alt.Legend(title=colour))
        if plan["chart_type"] == "bar":
            encodings["xOffset"] = alt.XOffset(f"{colour}:N", sort=domain)
    released = data[data["value"].notna()]
    base = alt.Chart(released).encode(**encodings)
    single = {} if colour else {"color": PALETTE[0]}
    if plan["chart_type"] == "line":
        chart = base.mark_line(point=alt.OverlayMarkDef(size=70, filled=True, **single), strokeWidth=2.5, **single)
        if colour:
            encodings["detail"] = alt.Detail(f"{colour}:N")
    else:
        chart = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, **single)
    hidden = data[data["value"].isna()].copy()
    layers = [chart]
    if not hidden.empty:
        hidden["marker"] = hidden["shown"].map(lambda s: s if is_suppressed(s) else "n/a")
        hidden["zero"] = 0
        marker_enc = {"x": x_enc, "y": alt.Y("zero:Q"), "text": "marker:N",
                      "tooltip": [alt.Tooltip(f"{x}:N", title=plan["x_label"] or "Group"), alt.Tooltip("marker:N", title="Hidden")]}
        if colour:
            marker_enc["xOffset"] = alt.XOffset(f"{colour}:N", sort=list(plan["colors"])) if plan["chart_type"] == "bar" else alt.XOffset(value=0)
            marker_enc["color"] = alt.Color(f"{colour}:N", scale=alt.Scale(domain=list(plan["colors"]), range=list(plan["colors"].values())), legend=None)
        layers.append(alt.Chart(hidden).mark_text(dy=-8, fontSize=11, fontWeight="bold").encode(**marker_enc))
    figure = alt.layer(*layers).properties(title=alt.TitleParams(plan["title"], anchor="start", fontSize=16,
                                                                 color=CHART_TEXT["title"]), height=360,
                                           background="white", padding={"left": 8, "right": 8, "top": 8, "bottom": 8})
    return figure.configure_axis(grid=True, gridColor=CHART_TEXT["grid"], gridOpacity=1, domainColor=CHART_TEXT["grid"],
                                 tickColor=CHART_TEXT["grid"], labelColor=CHART_TEXT["label"],
                                 titleColor=CHART_TEXT["title"], labelFontSize=12, titleFontSize=12
                                 ).configure_view(strokeOpacity=0).configure_legend(
        labelColor=CHART_TEXT["label"], titleColor=CHART_TEXT["title"])


def display_table(plan):
    """Exact released values behind the chart (markers kept for hidden cells)."""
    data = plan["data"]
    columns = [plan["x"]] + ([plan["color"]] if plan["color"] else [])
    table = data[columns + ["shown"]].rename(columns={"shown": plan["y_label"], plan["x"]: plan["x_label"] or plan["x"]})
    return table.reset_index(drop=True)


def png_table_and_spec(plan):
    """Inputs for the matplotlib PNG renderer (viz.render_chart) from the same plan."""
    data = plan["data"].copy()
    x_col = plan["x"]
    frame = pd.DataFrame({x_col: data[x_col]})
    if plan["color"]:
        frame[plan["color"]] = data[plan["color"]]
    frame[plan["metric"]] = data["shown"]
    spec = {"type": plan["chart_type"], "metric": plan["metric"], "x": x_col, "color": plan["color"]}
    return frame, spec


def requested_type(command):
    text = str(command or "").casefold()
    for key in ("pie", "donut", "area", "3d", "scatter", "box", "histogram"):
        if re.search(rf"\b{key}\b", text):
            return key
    if re.search(r"\bline\b", text):
        return "line"
    if re.search(r"\b(?:bar|columns?)\b", text):
        return "bar"
    return None
