import io
import re

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


DIMENSION_LABELS = {
    "sex": "Sex",
    "age_band": "Age band",
    "region": "Region",
    "data_holder": "Hospital",
    "year": "Year",
}
_DIMENSION_PATTERNS = {
    "year": re.compile(r"\byears?\b|\byearly\b|\bannual(?:ly)?\b|\bover\s+time\b|\btrends?\b|\btime\b", re.I),
    "sex": re.compile(r"\b(?:sex|gender)\b", re.I),
    "age_band": re.compile(r"\bage(?:\s+(?:band|group))?\b", re.I),
    "region": re.compile(r"\bregions?\b", re.I),
    "data_holder": re.compile(r"\bhospitals?\b", re.I),
}
_GROUP_CUE = re.compile(r"\b(?:by|across|per|between)\b", re.I)
_REFUSED_CHART = re.compile(
    r"\bhistogram\b|\bscatter\b|\bbox\s?plot\b|"
    r"\bdistribution\s+of\s+(?:an?\s+)?(?:individuals?|patients?)\b",
    re.I,
)
_REFUSED_MONTHLY = re.compile(r"\bmonthly\b|\bper\s+month\b|\bby\s+month\b|\bmonths?\b", re.I)
_TIME_CUE = re.compile(r"\bover\s+time\b|\btrends?\b|\byearly\b|\bannual(?:ly)?\b|\bper\s+year\b|"
                       r"\bby\s+year\b|\beach\s+year\b|\btime\s+series\b", re.I)
_TABLE_CUE = re.compile(r"\b(?:as\s+a\s+table|show\s+(?:a\s+|the\s+)?table|table\s+instead|in\s+a\s+table)\b", re.I)
_AGE_FILTER = re.compile(r"\b(?:aged?|ages?)\s+(\d{1,3})\s*(?:-|–|to)\s*(\d{1,3})\b|\b(?:aged?|over|older\s+than)\s+(\d{1,3})\s*\+?\b", re.I)
_PERCENT_CUE = re.compile(r"\bpercent(?:age)?s?\b|\bproportions?\b|\bshares?\b|%", re.I)
_REFUSED_TYPE = re.compile(r"\b(pie|donut|area|3d)\b", re.I)
_HELP_MESSAGE = (
    "Try a bar chart of patients by sex, a line graph of mean HbA1c by age band, "
    "group the results by region, or show the results by year as a line graph."
)
_PATIENT_COUNT_WORDS = {"patient", "patients", "number", "count", "prevalence"}
_METRIC_WORDS = {
    "mean", "average", "hba1c", "bmi", "egfr", "prevalence", "number",
    "count", "patient", "patients", "adults", "diabetes", "systolic",
    "pressure",
}


def _metric_words(value):
    return set(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _dimension_column(dimension, columns):
    target = DIMENSION_LABELS[dimension].casefold()
    return next(
        (column for column in columns if str(column).casefold() in {target, dimension}),
        None,
    )


def answer_table(answer, group_by):
    dimension_columns = [
        DIMENSION_LABELS.get(item.get("dimension"), item.get("dimension"))
        for item in group_by or []
    ]
    metric_columns = []
    rows = []

    def walk(node, path):
        if isinstance(node, dict) and node and all(
            not isinstance(value, dict) for value in node.values()
        ):
            row = {
                column: path[index]
                for index, column in enumerate(dimension_columns)
                if index < len(path)
            }
            for metric, value in node.items():
                if metric not in metric_columns:
                    metric_columns.append(metric)
                row[metric] = value
            rows.append(row)
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + [key])
        else:
            row = {
                column: path[index]
                for index, column in enumerate(dimension_columns)
                if index < len(path)
            }
            if "value" not in metric_columns:
                metric_columns.append("value")
            row["value"] = node
            rows.append(row)

    walk(answer, [])
    columns = [*dimension_columns, *metric_columns]
    return pd.DataFrame(rows, columns=columns)


def _requested_dimensions(command):
    found = []
    for cue in _GROUP_CUE.finditer(command):
        segment = command[cue.end():]
        segment = re.split(r"[.;!?]", segment, maxsplit=1)[0]
        matches = []
        for dimension, pattern in _DIMENSION_PATTERNS.items():
            matches.extend(
                (match.start(), dimension)
                for match in pattern.finditer(segment)
            )
        for _, dimension in sorted(matches):
            if dimension not in found:
                found.append(dimension)
    return found


def _metric_choice(command, metric_columns):
    command_words = _metric_words(command) & _METRIC_WORDS
    if not command_words:
        return None

    best = None
    best_score = 0
    for metric in metric_columns:
        words = _metric_words(metric)
        score = len(command_words & words)
        if command_words & _PATIENT_COUNT_WORDS and words & _PATIENT_COUNT_WORDS:
            score += 1
        if score > best_score:
            best = metric
            best_score = score
    return best


def interpret_chart_command(text, current_spec, metric_columns, dimension_columns):
    command = str(text or "").strip()
    current = dict(current_spec or {})
    metrics = list(metric_columns or [])
    dimensions = list(dimension_columns or [])
    metric = current.get("metric")
    if metric not in metrics:
        metric = metrics[0] if metrics else None
    x = current.get("x")
    if x not in dimensions:
        x = dimensions[0] if dimensions else None
    color = current.get("color")
    if color not in dimensions or color == x:
        color = None
    chart_type = current.get("type") if current.get("type") in {"bar", "line"} else "bar"
    spec = {"type": chart_type, "metric": metric, "x": x, "color": color}

    extras = {"table": False, "filter": None, "prefer": None, "refused_type": None}
    if _TABLE_CUE.search(command):
        return {"ok": True, "message": "", "spec": spec, "regroup": None, **extras, "table": True}
    age = _AGE_FILTER.search(command)
    if age and re.search(r"\b(?:only|restrict|limit|just)\b", command, re.I):
        low, high, single = age.group(1), age.group(2), age.group(3)
        item = ({"type": "age", "label": "age", "min": int(low), "max": int(high)} if low
                else {"type": "age", "label": "age", "min": int(single)})
        return {"ok": True, "message": "Re-running the request with the new age filter through the approval process.",
                "spec": spec, "regroup": None, **extras, "filter": item}
    if _PERCENT_CUE.search(command) and "percentage" not in command.casefold().split("chart")[0] or re.search(
            r"\b(?:percentages?|proportions?)\s+instead\b", command, re.I):
        percentage = next((m for m in metrics if re.search(r"%|prevalence|percentage|share", str(m), re.I)), None)
        if percentage:
            spec["metric"] = percentage
            return {"ok": True, "message": "", "spec": spec, "regroup": None, **extras, "prefer": "percentage"}
        return {"ok": False, "message": ("Percentages are not part of this result. Ask a question that names a "
                                         "prevalence or share (e.g. 'prevalence of type 2 diabetes by region') so the "
                                         "percentage is calculated and privacy-checked by the engine."),
                "spec": spec, "regroup": None, **extras}
    refused_type = _REFUSED_TYPE.search(command)
    if refused_type:
        return {"ok": True, "message": "", "spec": spec, "regroup": None, **extras,
                "refused_type": refused_type.group(1).casefold()}
    if _REFUSED_CHART.search(command):
        return {
            "ok": False,
            "message": (
                "Not available: this chart needs individual patients' values, "
                "which only approved users may analyse inside the secure "
                "processing environment. Ask for the 'distribution of BMI' to get counts per clinical category, "
                "or the 'relationship between BMI and HbA1c' for mean values per category."
            ),
            "spec": spec,
            "regroup": None,
            **extras,
        }
    if _REFUSED_MONTHLY.search(command):
        return {
            "ok": False,
            "message": "Trends over time are available per calendar year only; monthly results are not "
                       "available in this prototype. Try 'show this by year as a line graph'.",
            "spec": spec,
            "regroup": None,
            **extras,
        }

    recognized = False
    time_cue = bool(_TIME_CUE.search(command))
    if re.search(r"\bline\b", command, re.I):
        spec["type"] = "line"
        recognized = True
    elif re.search(r"\b(?:bar|columns?)\b", command, re.I):
        spec["type"] = "bar"
        recognized = True
    elif time_cue:
        spec["type"] = "line"

    chosen_metric = _metric_choice(command, metrics)
    if chosen_metric is not None:
        spec["metric"] = chosen_metric
        recognized = True

    requested = _requested_dimensions(command)
    if time_cue and "year" not in requested:
        requested.insert(0, "year")
    if requested:
        recognized = True
        requested_columns = [
            _dimension_column(dimension, dimensions)
            or DIMENSION_LABELS[dimension]
            for dimension in requested
        ]
        spec["x"] = requested_columns[0]
        spec["color"] = requested_columns[1] if len(requested_columns) > 1 else None
        missing = [
            dimension for dimension in requested
            if _dimension_column(dimension, dimensions) is None
        ]
        if missing:
            return {
                "ok": True,
                "message": (
                    "Re-running the request with the new grouping through the "
                    "approval process."
                ),
                "spec": spec,
                "regroup": requested,
                **extras,
            }

    if not recognized:
        return {
            "ok": False,
            "message": _HELP_MESSAGE,
            "spec": spec,
            "regroup": None,
            **extras,
        }
    return {"ok": True, "message": "", "spec": spec, "regroup": None, **extras}


def plotted_data(table, spec):
    result = table.copy()
    metric = spec.get("metric")
    if metric in result.columns:
        result[metric] = pd.to_numeric(result[metric], errors="coerce")
    return result


def _display_value(value):
    if pd.isna(value):
        return ""
    return str(value)


def render_chart(table, spec, title, population_text, period_text, unit):
    metric = spec.get("metric")
    if metric not in table.columns:
        raise ValueError(f"Metric column {metric!r} is not present in the answer table.")

    x_column = spec.get("x")
    color_column = spec.get("color")
    data = plotted_data(table, spec)
    if x_column in table.columns:
        categories = list(dict.fromkeys(table[x_column].map(_display_value)))
        x_label = str(x_column)
    else:
        categories = ["All"] if len(table) == 1 else [
            str(index + 1) for index in range(len(table))
        ]
        x_label = "Result"
    positions = list(range(len(categories)))
    category_index = {category: index for index, category in enumerate(categories)}

    labels = {}
    values_by_group = {}
    for row_index, row in table.iterrows():
        category = (
            _display_value(row[x_column])
            if x_column in table.columns
            else categories[list(table.index).index(row_index)]
        )
        raw_value = row[metric]
        group = _display_value(row[color_column]) if color_column in table.columns else None
        values_by_group[(category, group)] = data.loc[row_index, metric]
        if isinstance(raw_value, str) and raw_value.casefold() in {"<10", "suppressed"}:
            labels[(category, group)] = raw_value

    fig = Figure(figsize=(9, 5))
    axis = fig.add_subplot(111)
    colors = (
        list(dict.fromkeys(table[color_column].map(_display_value)))
        if color_column in table.columns
        else []
    )

    def value_for(category, color_value=None):
        return values_by_group.get((category, color_value), float("nan"))

    def annotate_missing(x_position, category, color_value=None):
        annotation = labels.get((category, color_value))
        if annotation:
            axis.annotate(
                annotation,
                (x_position, 0),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    chart_type = spec.get("type", "bar")
    if color_column in table.columns:
        width = 0.8 / max(len(colors), 1)
        for color_index, color_value in enumerate(colors):
            values = [value_for(category, color_value) for category in categories]
            if chart_type == "line":
                axis.plot(
                    positions,
                    values,
                    marker="o",
                    label=color_value,
                )
                for position, category in zip(positions, categories):
                    annotate_missing(position, category, color_value)
            else:
                offset = (color_index - (len(colors) - 1) / 2) * width
                plotted_positions = [position + offset for position in positions]
                axis.bar(
                    plotted_positions,
                    values,
                    width=width,
                    label=color_value,
                )
                for position, category in zip(plotted_positions, categories):
                    annotate_missing(position, category, color_value)
        axis.legend(title=str(color_column))
    else:
        values = [value_for(category) for category in categories]
        if chart_type == "line":
            axis.plot(positions, values, marker="o")
        else:
            axis.bar(positions, values)
        for position, category in zip(positions, categories):
            annotate_missing(position, category)

    if len(categories) > 6 or max((len(c) for c in categories), default=0) > 10:
        axis.set_xticks(positions, categories, rotation=30, ha="right")
    else:
        axis.set_xticks(positions, categories)
    axis.set_xlabel(x_label)
    metric_label = str(metric)
    if unit and not re.search(r"\([^()]+\)\s*$", metric_label):
        metric_label += f" ({unit})"
    axis.set_ylabel(metric_label)
    axis.set_title(title)
    axis.axhline(0, color="#666666", linewidth=0.6)
    footnote = (
        f"Population: {population_text or 'Not specified'} | "
        f"Time period: {period_text or 'Not specified'}. "
        "Cells marked <10 / suppressed are hidden by the small-cell rule "
        "(fewer than 10 patients). Opted-out patients are excluded."
    )
    fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.1, 1, 1))

    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return fig, output.getvalue()


def methodology_text(spec, response):
    return (
        "Counts are distinct patients. Means are the mean of each patient's latest "
        "value (within the period if given). Numeric results are rounded half-up "
        "to 1 decimal place. The small-cell rule shows counts below 10 as <10 "
        "and suppresses percentages and means based on fewer than 10 patients."
    )
