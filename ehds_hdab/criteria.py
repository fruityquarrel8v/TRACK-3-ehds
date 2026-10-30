from datetime import date
import math
import re

from .plan import QueryPlan


_DIMENSIONS = {"sex", "age_band", "region", "data_holder", "year"}
DEFAULT_YEARS = (2023, 2025)
MAX_YEAR_SPAN = 30
_RECORD_TYPES = {"conditions", "medications", "observations", "encounters"}
_LAB_OPERATORS = {">", ">=", "<", "<=", "present"}


def _slug(value):
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    return slug or "criterion"


def _criterion_name(item):
    return str(item.get("label") or item.get("type") or "unknown")


def _date(value):
    if value in (None, ""):
        return None, True
    if not isinstance(value, str):
        return None, False
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None, False
    return parsed.isoformat(), parsed.isoformat() == value


def _date_range(item, start_field, end_field):
    start, start_valid = _date(item.get(start_field))
    end, end_valid = _date(item.get(end_field))
    if not start_valid or not end_valid:
        return None, None
    if start and end and start > end:
        return None, None
    return start, end


def _codes(value):
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return []
    if any(not isinstance(code, str) for code in values):
        return []
    return list(dict.fromkeys(str(code).strip().upper() for code in values if str(code).strip()))


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _valid_values(value):
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _default_years(criteria):
    """Years for a 'year' grouping: the request's own period if it names one, else DEFAULT_YEARS."""
    years = []

    def visit(item):
        if not isinstance(item, dict):
            return
        for child in item.get("items", []) or []:
            visit(child)
        for key in ("period_start", "onset_start", "period_end", "onset_end"):
            value = item.get(key)
            if isinstance(value, str) and re.match(r"\d{4}", value):
                years.append(int(value[:4]))

    population = criteria.get("population")
    if isinstance(population, dict):
        for item in population.get("items", []) or []:
            visit(item)
    for metric in criteria.get("metrics", []) or []:
        visit(metric)
    if years:
        return min(years), max(years)
    return DEFAULT_YEARS


def _valid_leaf(item):
    kind = item.get("type")
    label = _criterion_name(item)
    spec = {"label": str(item.get("label") or label)}

    if kind == "condition":
        codes = _codes(item.get("icd10"))
        start, end = _date_range(item, "onset_start", "onset_end")
        if not codes or start is None and item.get("onset_start") not in (None, ""):
            return None
        if end is None and item.get("onset_end") not in (None, ""):
            return None
        spec.update({"kind": "condition", "icd10_prefixes": codes})
        if start:
            spec["onset_start"] = start
        if end:
            spec["onset_end"] = end
    elif kind == "medication":
        codes = _codes(item.get("atc"))
        if not codes:
            return None
        status = str(item.get("status") or "").casefold()
        if status == "ever":
            spec.update({"kind": "medication", "atc_prefixes": codes, "active_only": False})
        elif status == "during_period":
            start, end = _date_range(item, "period_start", "period_end")
            if not start or not end:
                return None
            spec.update({
                "kind": "medication",
                "atc_prefixes": codes,
                "period_start": start,
                "period_end": end,
            })
        else:
            spec.update({"kind": "medication", "atc_prefixes": codes, "active_only": True})
    elif kind == "lab":
        raw_loinc = item.get("loinc")
        loinc = raw_loinc.strip() if isinstance(raw_loinc, str) else ""
        operator = str(item.get("op") or "").strip()
        start, end = _date_range(item, "period_start", "period_end")
        if not loinc or operator not in _LAB_OPERATORS:
            return None
        if start is None and item.get("period_start") not in (None, ""):
            return None
        if end is None and item.get("period_end") not in (None, ""):
            return None
        spec.update({
            "kind": "observation",
            "loinc": loinc,
            "pick": "latest",
            "op": operator,
        })
        if operator != "present":
            value = item.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            if not math.isfinite(float(value)):
                return None
            spec["value"] = float(value)
        if start:
            spec["date_start"] = start
        if end:
            spec["date_end"] = end
    elif kind == "age":
        low = item.get("min")
        high = item.get("max")
        if low is not None:
            low = _integer(low)
        if high is not None:
            high = _integer(high)
        if (item.get("min") is not None and low is None
                or item.get("max") is not None and high is None
                or low is None and high is None
                or low is not None and high is not None and low > high):
            return None
        spec.update({"kind": "age", "min": low, "max": high})
    elif kind == "sex":
        value = str(item.get("value") or "").casefold()
        if value not in {"female", "male"}:
            return None
        spec.update({"kind": "sex", "value": value})
    elif kind in {"region", "data_holder"}:
        values = _valid_values(item.get("values"))
        if not values:
            return None
        spec.update({"kind": kind, "values": values})
    elif kind == "hospital_stay":
        codes = _codes(item.get("icd10"))
        start, end = _date_range(item, "period_start", "period_end")
        if start is None and item.get("period_start") not in (None, ""):
            return None
        if end is None and item.get("period_end") not in (None, ""):
            return None
        spec.update({"kind": "encounter", "types": ["inpatient"]})
        if codes:
            spec["primary_icd10_prefixes"] = codes
        if start:
            spec["start_start"] = start
        if end:
            spec["start_end"] = end
    else:
        return None
    return spec


def plan_from_criteria(criteria) -> QueryPlan:
    plan = QueryPlan()
    counter = 0

    def warning(item):
        plan.warnings.append(
            f"Could not apply criterion '{_criterion_name(item)}' — staff review required."
        )

    def add_item(item):
        nonlocal counter
        if not isinstance(item, dict):
            warning({})
            return None
        if "items" in item:
            match = str(item.get("match") or "all").casefold()
            if match not in {"all", "any"} or not isinstance(item.get("items"), list):
                warning(item)
                return None
            group_items = []
            for child in item["items"]:
                concept_key = add_item(child)
                if concept_key:
                    group_items.append({
                        "concept": concept_key,
                        "negate": bool(child.get("exclude")) if isinstance(child, dict) else False,
                    })
            if not group_items:
                return None
            counter += 1
            key = f"c{counter}_{_slug(item.get('label') or f'{match}_group')}"
            plan.concepts[key] = {
                "kind": "group",
                "match": match,
                "items": group_items,
            }
            return key

        spec = _valid_leaf(item)
        if spec is None:
            warning(item)
            return None
        counter += 1
        key = f"c{counter}_{_slug(item.get('label') or item.get('type'))}"
        plan.concepts[key] = spec
        if item.get("type") == "medication" and str(item.get("status") or "").casefold() not in {
            "active", "ever", "during_period"
        }:
            medication_warning = (
                f"Medication status for '{spec['label']}' was not defined; counted as active "
                "on 31 December 2025 — staff review required."
            )
            if medication_warning not in plan.warnings:
                plan.warnings.append(medication_warning)
        return key

    population = criteria.get("population")
    if not isinstance(population, dict) or not isinstance(population.get("items", []), list):
        if population is not None:
            warning({"label": "population"})
        population = {"match": "all", "items": []}
    match = str(population.get("match") or "all").casefold()
    population_items = []
    for item in population.get("items", []):
        key = add_item(item)
        if key:
            population_items.append({
                "concept": key,
                "negate": bool(item.get("exclude")) if isinstance(item, dict) else False,
            })
    if match not in {"all", "any"}:
        plan.warnings.append("Could not apply criterion 'population' — staff review required.")
        population_items = []
    if match == "any" and population_items:
        counter += 1
        key = f"c{counter}_any_population"
        plan.concepts[key] = {
            "kind": "group",
            "match": "any",
            "items": population_items,
        }
        plan.cohort = [{"concept": key, "negate": False}]
    else:
        plan.cohort = population_items

    if not plan.cohort:
        if not plan.warnings:
            plan.warnings.append("No applicable population criteria — staff review required.")
        return plan

    dimensions = []
    requested_dimensions = criteria.get("group_by", []) or []
    if not isinstance(requested_dimensions, list):
        requested_dimensions = []
        plan.warnings.append("Could not apply grouping 'unknown' — staff review required.")
    for dimension in requested_dimensions:
        if not isinstance(dimension, dict):
            plan.warnings.append("Could not apply grouping 'unknown' — staff review required.")
            continue
        name = dimension.get("dimension")
        if not isinstance(name, str) or name not in _DIMENSIONS:
            plan.warnings.append(
                f"Could not apply grouping '{name or 'unknown'}' — staff review required."
            )
            continue
        if name == "age_band":
            bands = dimension.get("bands")
            if not isinstance(bands, list) or not bands:
                plan.warnings.append("Could not apply grouping 'age_band' — staff review required.")
                continue
            valid_bands = []
            for band in bands:
                if not isinstance(band, dict) or not isinstance(band.get("label"), str):
                    valid_bands = []
                    break
                low, high = band.get("lo"), band.get("hi")
                if (low is not None and _integer(low) is None
                        or high is not None and _integer(high) is None
                        or low is None and high is None
                        or low is not None and high is not None and low > high):
                    valid_bands = []
                    break
                valid_bands.append({"lo": low, "hi": high, "label": band["label"]})
            if not valid_bands:
                plan.warnings.append("Could not apply grouping 'age_band' — staff review required.")
                continue
            dimensions.append({"dimension": name, "bands": valid_bands})
        elif name == "year":
            default_from, default_to = _default_years(criteria)
            start = _integer(dimension.get("from")) if dimension.get("from") is not None else default_from
            end = _integer(dimension.get("to")) if dimension.get("to") is not None else default_to
            if start is None or end is None or start > end or end - start > MAX_YEAR_SPAN:
                plan.warnings.append("Could not apply grouping 'year' — staff review required.")
                continue
            if any(d["dimension"] == "year" for d in dimensions):
                continue
            dimensions.append({"dimension": "year", "from": start, "to": end})
        else:
            dimensions.append({"dimension": name})
    dimensions.sort(key=lambda d: d["dimension"] != "year")
    plan.group_by = dimensions

    record_types = criteria.get("record_types", []) or []
    if isinstance(record_types, list):
        plan.record_types = [
            value for value in record_types
            if isinstance(value, str) and value in _RECORD_TYPES
        ]
        if len(plan.record_types) != len(record_types):
            plan.warnings.append("Could not apply criterion 'record_types' — staff review required.")
    else:
        plan.warnings.append("Could not apply criterion 'record_types' — staff review required.")

    if plan.cohort:
        metrics = criteria.get("metrics", []) or []
        if not isinstance(metrics, list):
            plan.warnings.append("Could not apply criterion 'metrics' — staff review required.")
            metrics = []
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                plan.warnings.append("Could not apply criterion 'metric' — staff review required.")
                continue
            kind = metric.get("type")
            label = str(metric.get("label") or ("patients" if kind == "count_patients" else kind or "metric"))
            metric_id = f"m{index + 1}_{_slug(label)}"
            if kind == "count_patients":
                plan.metrics.append({
                    "id": metric_id,
                    "label": label,
                    "type": "count_patients",
                    "population": list(plan.cohort),
                })
            elif kind == "mean_latest_lab":
                raw_loinc = metric.get("loinc")
                loinc = raw_loinc.strip() if isinstance(raw_loinc, str) else ""
                start, end = _date_range(metric, "period_start", "period_end")
                if (not loinc
                        or start is None and metric.get("period_start") not in (None, "")
                        or end is None and metric.get("period_end") not in (None, "")):
                    warning({"label": label})
                    continue
                counter += 1
                observation_key = f"c{counter}_{_slug(label)}_latest"
                observation_spec = {
                    "kind": "observation",
                    "loinc": loinc,
                    "pick": "latest",
                }
                if start:
                    observation_spec["date_start"] = start
                if end:
                    observation_spec["date_end"] = end
                plan.concepts[observation_key] = observation_spec
                plan.metrics.append({
                    "id": metric_id,
                    "label": label,
                    "type": "mean",
                    "observation": observation_key,
                    "population": [*plan.cohort, {"concept": observation_key, "negate": False}],
                })
            else:
                warning({"label": label})
        if not plan.metrics:
            plan.metrics.append({
                "id": "m1_patients",
                "label": "patients",
                "type": "count_patients",
                "population": list(plan.cohort),
            })
    return plan


def _codes_text(codes):
    return ", ".join(str(code) for code in (codes or []))


def _format_date(value):
    try:
        parsed = date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return str(value)
    return f"{parsed.day} {parsed.strftime('%B %Y')}"


def _period_text(start, end):
    if not start and not end:
        return ""
    if start and end and start[:4] == end[:4] and start.endswith("-01-01") and end.endswith("-12-31"):
        return f" in {start[:4]}"
    if start and end:
        return f" from {start} to {end}"
    if start:
        return f" since {start}"
    return f" through {end}"


def _describe_leaf(item, excluded=False):
    kind = item.get("type")
    label = str(item.get("label") or kind or "criterion")
    if kind == "condition":
        codes = _codes_text(item.get("icd10"))
        result = "with " + label + (f" (ICD-10 {codes})" if codes else "")
        onset_start = item.get("onset_start")
        onset_end = item.get("onset_end")
        onset_start_text = str(onset_start) if onset_start else None
        onset_end_text = str(onset_end) if onset_end else None
        if onset_start_text and onset_end_text:
            if (
                onset_start_text == f"{onset_start_text[:4]}-01-01"
                and onset_end_text == f"{onset_start_text[:4]}-12-31"
            ):
                result += f" newly diagnosed in {onset_start_text[:4]}"
            else:
                result += (
                    f" newly diagnosed from {_format_date(onset_start_text)}"
                    f" to {_format_date(onset_end_text)}"
                )
        elif onset_end_text:
            result += f" diagnosed on or before {_format_date(onset_end_text)}"
        elif onset_start_text:
            result += f" diagnosed since {_format_date(onset_start_text)}"
        return result
    if kind == "medication":
        codes = _codes_text(item.get("atc"))
        result = label + (f" (ATC {codes})" if codes else "")
        status = item.get("status")
        excluded = excluded or bool(item.get("exclude"))
        if status == "ever":
            action = "never prescribed" if excluded else "ever prescribed"
            return f"{action} {result}"
        if status == "during_period":
            action = "not prescribed" if excluded else "prescribed"
            return f"{action} {result} during the time period"
        if excluded:
            return f"not taking {result} on 31 December 2025"
        return f"taking {result} (active on 31 December 2025)"
    if kind == "lab":
        loinc = item.get("loinc")
        op = item.get("op")
        operator_words = {">": "above", ">=": "at least", "<": "below", "<=": "at most"}
        result = f"with a latest {label} (LOINC {loinc})"
        if op == "present":
            result = f"with a measurement of {label} (LOINC {loinc})"
        elif op in operator_words:
            unit = item.get("unit")
            result += f" {operator_words[op]} {item.get('value')}"
            if unit:
                result += f" {unit}"
        return result + _period_text(item.get("period_start"), item.get("period_end"))
    if kind == "age":
        low, high = item.get("min"), item.get("max")
        if low is not None and high is not None:
            return f"aged {low}–{high}"
        if low is not None:
            return f"aged {low} or older"
        return f"aged {high} or younger"
    if kind == "sex":
        return f"who are {item.get('value') or label}"
    if kind in {"region", "data_holder"}:
        name = "region" if kind == "region" else "hospital"
        return f"in {name} {', '.join(_valid_values(item.get('values')))}"
    if kind == "hospital_stay":
        codes = _codes_text(item.get("icd10"))
        result = "with an inpatient stay"
        if codes:
            result += f" for {label} with main diagnosis ICD-10 {codes}"
        return result + _period_text(item.get("period_start"), item.get("period_end"))
    return label


def _describe_item(item, negated=False):
    if not isinstance(item, dict):
        return "an unspecified criterion"
    if "items" in item:
        match = str(item.get("match") or "all").casefold()
        parts = [
            _describe_item(child, bool(child.get("exclude")) if isinstance(child, dict) else False)
            for child in item.get("items", [])
        ]
        if match == "any":
            description = "who meet at least one of: " + " or ".join(parts)
        else:
            description = "who meet all of: " + " and ".join(parts)
    else:
        excluded = negated or bool(item.get("exclude"))
        description = _describe_leaf(
            item,
            excluded=excluded and item.get("type") == "medication",
        )
    if (negated or item.get("exclude")) and item.get("type") != "medication":
        for prefix, replacement in (("with ", "without "), ("who are ", "who are not "), ("who meet ", "who do not meet ")):
            if description.startswith(prefix):
                return replacement + description[len(prefix):]
        return f"not {description}"
    return description


def describe_criteria(criteria) -> str:
    population = criteria.get("population") if isinstance(criteria, dict) else None
    items = population.get("items", []) if isinstance(population, dict) else []
    match = str(population.get("match") or "all").casefold() if isinstance(population, dict) else "all"
    parts = [_describe_item(item) for item in items if isinstance(item, dict)]
    if not parts:
        return "Find patients matching the selected criteria."
    if match == "any":
        return "Find patients who meet at least one of: " + " or ".join(parts) + "."
    return "Find patients " + " and ".join(parts) + "."


def criteria_to_definitions(criteria) -> list[str]:
    if not isinstance(criteria, dict):
        return []
    population = criteria.get("population")
    if not isinstance(population, dict):
        return []
    match = str(population.get("match") or "all").casefold()
    lines = [f"Patients must meet {match.upper()} of the listed criteria."]

    def collect(item, inherited_exclude=False):
        if not isinstance(item, dict):
            return
        if "items" in item:
            group_match = str(item.get("match") or "all").casefold()
            lines.append(f"Nested criteria are matched using {group_match.upper()}.")
            for child in item.get("items", []):
                collect(child, inherited_exclude or bool(child.get("exclude")) if isinstance(child, dict) else inherited_exclude)
            return
        description = _describe_leaf(item)
        excluded = inherited_exclude or bool(item.get("exclude"))
        lines.append(("Exclude: " if excluded else "Include: ") + description + ".")

    for item in population.get("items", []):
        collect(item, bool(item.get("exclude")) if isinstance(item, dict) else False)
    return lines
