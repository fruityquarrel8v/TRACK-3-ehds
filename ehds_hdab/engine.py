from decimal import Decimal, InvalidOperation
from itertools import product

import pandas as pd

import copy

from .config import REFERENCE_DATE


def _prefix_match(series, prefixes):
    if not prefixes:
        return pd.Series(False, index=series.index)
    values = series.astype(str)
    return pd.concat([values.str.startswith(str(prefix)) for prefix in prefixes], axis=1).any(axis=1)


def _concept_patients(key, plan, data):
    patients = data["patients"]
    spec = plan.concepts.get(key, {})
    kind = spec.get("kind")
    if kind == "group":
        all_ids = set(patients["patient_id"])
        match = spec.get("match")
        selected = set(all_ids) if match == "all" else set()
        for item in spec.get("items", []):
            matching = _concept_patients(item.get("concept"), plan, data)
            if item.get("negate"):
                matching = all_ids - matching
            if match == "any":
                selected.update(matching)
            else:
                selected.intersection_update(matching)
        return selected
    if kind == "age":
        ages = patients["age"]
        mask = pd.Series(True, index=patients.index)
        if spec.get("min") is not None:
            mask &= ages.ge(spec["min"])
        if spec.get("max") is not None:
            mask &= ages.le(spec["max"])
        return set(patients.loc[mask, "patient_id"])
    if kind == "sex":
        return set(patients.loc[patients["sex"].str.casefold().eq(str(spec.get("value", "")).casefold()), "patient_id"])
    if kind in ("region", "data_holder"):
        column = "region" if kind == "region" else "data_holder"
        values = spec.get("values", [])
        return set(patients.loc[patients[column].isin(values), "patient_id"])
    if kind == "condition":
        conditions = data["conditions"]
        mask = _prefix_match(conditions["icd10"], spec.get("icd10_prefixes", []))
        if spec.get("onset_year") is not None:
            mask &= conditions["onset_date"].dt.year.eq(spec["onset_year"])
        if spec.get("onset_start"):
            mask &= conditions["onset_date"].ge(pd.Timestamp(spec["onset_start"]))
        if spec.get("onset_end"):
            mask &= conditions["onset_date"].le(pd.Timestamp(spec["onset_end"]))
        return set(conditions.loc[mask, "patient_id"])
    if kind == "medication":
        medications = data["medications"]
        mask = _prefix_match(medications["atc"], spec.get("atc_prefixes", []))
        if spec.get("active_only"):
            reference = pd.Timestamp(spec.get("as_of") or REFERENCE_DATE)
            mask &= medications["start_date"].le(reference)
            mask &= medications["end_date"].isna() | medications["end_date"].ge(reference)
        if spec.get("period_start"):
            mask &= medications["end_date"].isna() | medications["end_date"].ge(
                pd.Timestamp(spec["period_start"])
            )
        if spec.get("period_end"):
            mask &= medications["start_date"].le(pd.Timestamp(spec["period_end"]))
        return set(medications.loc[mask, "patient_id"])
    if kind == "observation":
        observations = data["observations"]
        mask = observations["loinc"].eq(str(spec.get("loinc", "")))
        if spec.get("date_start"):
            mask &= observations["date"].ge(pd.Timestamp(spec["date_start"]))
        if spec.get("date_end"):
            mask &= observations["date"].le(pd.Timestamp(spec["date_end"]))
        matching = observations.loc[mask]
        operator = spec.get("op")
        if operator == "present":
            return set(matching["patient_id"])
        if operator in {">", ">=", "<", "<="}:
            latest = matching.sort_values("date").drop_duplicates("patient_id", keep="last")
            values = latest["value"].map(_to_decimal)
            threshold = Decimal(str(spec["value"]))
            comparisons = {
                ">": lambda value: value is not None and value > threshold,
                ">=": lambda value: value is not None and value >= threshold,
                "<": lambda value: value is not None and value < threshold,
                "<=": lambda value: value is not None and value <= threshold,
            }
            selected = values.map(comparisons[operator])
            return set(latest.loc[selected, "patient_id"])
        return set(matching["patient_id"])
    if kind == "encounter":
        return set(_encounters_for(spec, data).loc[:, "patient_id"])
    return set()


def _encounters_for(spec, data):
    encounters = data["encounters"]
    mask = pd.Series(True, index=encounters.index)
    if spec.get("types"):
        mask &= encounters["type"].str.casefold().isin([str(x).casefold() for x in spec["types"]])
    prefixes = spec.get("primary_icd10_prefixes", [])
    if prefixes:
        mask &= _prefix_match(encounters["primary_icd10"], prefixes)
    if spec.get("start_year") is not None:
        mask &= encounters["start_date"].dt.year.eq(spec["start_year"])
    if spec.get("start_start"):
        mask &= encounters["start_date"].ge(pd.Timestamp(spec["start_start"]))
    if spec.get("start_end"):
        mask &= encounters["start_date"].le(pd.Timestamp(spec["start_end"]))
    return encounters.loc[mask]


def population_patient_ids(population, plan, data):
    all_ids = set(data["patients"]["patient_id"])
    result = set(all_ids)
    for item in population or []:
        selected = _concept_patients(item.get("concept"), plan, data)
        result.intersection_update(all_ids - selected if item.get("negate") else selected)
    return result


def _dimension_values(dim, data):
    name = dim["dimension"]
    if name == "age_band":
        bands = dim.get("bands") or []
        return [band["label"] for band in bands], bands
    if name in ("region", "data_holder", "sex"):
        return data["categories"].get(name, []), None
    if name == "year":
        return [str(year) for year in range(int(dim["from"]), int(dim["to"]) + 1)], None
    return [], None


def _plan_for_year(plan, year):
    """The same plan evaluated for one calendar year: diagnoses recorded by 31 Dec of that year (or within
    it when the request asks for new cases), medication status as of 31 Dec, latest lab value within the
    year, hospital stays within the year. Age stays as of the reference date."""
    start, end = f"{year}-01-01", f"{year}-12-31"
    yearly = copy.deepcopy(plan)
    for spec in yearly.concepts.values():
        kind = spec.get("kind")
        if kind == "condition":
            if spec.get("onset_year") is not None:
                spec["onset_year"] = year
            elif spec.get("onset_start") and spec.get("onset_end"):
                spec["onset_start"], spec["onset_end"] = start, end
            else:
                spec["onset_end"] = end
        elif kind == "medication":
            if spec.get("active_only"):
                spec["as_of"] = end
            elif spec.get("period_start") or spec.get("period_end"):
                spec["period_start"], spec["period_end"] = start, end
            else:
                spec["period_end"] = end
        elif kind == "observation":
            spec["date_start"], spec["date_end"] = start, end
        elif kind == "encounter":
            spec.pop("start_year", None)
            spec["start_start"], spec["start_end"] = start, end
    return yearly


def _selected_observations(metric, population, plan, data):
    key = metric.get("observation")
    spec = plan.concepts.get(key, {})
    frame = data["observations"]
    frame = frame.loc[
        frame["patient_id"].isin(population)
        & frame["loinc"].eq(str(spec.get("loinc", "")))
    ].copy()
    if spec.get("date_start"):
        frame = frame.loc[frame["date"].ge(pd.Timestamp(spec["date_start"]))]
    if spec.get("date_end"):
        frame = frame.loc[frame["date"].le(pd.Timestamp(spec["date_end"]))]
    if frame.empty:
        frame["_numeric"] = pd.Series(dtype=object)
        return frame
    pick = spec.get("pick", "latest")
    frame["_numeric"] = frame["value"].map(_to_decimal)
    if pick in ("max", "min"):
        frame = frame.loc[frame["_numeric"].notna()]
        return frame.sort_values(
            ["_numeric", "date"],
            ascending=[pick == "min", True],
        ).drop_duplicates("patient_id", keep="first")
    ascending = pick == "earliest"
    return frame.sort_values(["date"], ascending=ascending).drop_duplicates("patient_id", keep="first")


def _to_decimal(value):
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _population_ids_cached(population, plan, data, all_ids, concept_cache):
    selected = set(all_ids)
    for item in population or []:
        key = item.get("concept")
        if key not in concept_cache:
            concept_cache[key] = _concept_patients(key, plan, data)
        matching = concept_cache[key]
        selected.intersection_update(all_ids - matching if item.get("negate") else matching)
    return selected


def _group_key(key, dimension_count):
    if dimension_count == 1:
        return key if isinstance(key, tuple) else (key,)
    return tuple(key)


def _age_band_series(ages, bands):
    result = pd.Series(None, index=ages.index, dtype=object)
    for band in bands:
        low, high = band.get("lo"), band.get("hi")
        mask = ages.notna()
        if low is not None:
            mask &= ages.ge(low)
        if high is not None:
            mask &= ages.le(high)
        result.loc[mask] = band["label"]
    return result


def execute(plan, data):
    year_dims = [dim for dim in plan.group_by if dim.get("dimension") == "year"]
    if not year_dims:
        return _execute(plan, data)
    year_dim = year_dims[0]
    inner = copy.copy(plan)
    inner.group_by = [dim for dim in plan.group_by if dim.get("dimension") != "year"]
    groups = []
    for year in range(int(year_dim["from"]), int(year_dim["to"]) + 1):
        result = _execute(_plan_for_year(inner, year), data)
        for group in result["groups"]:
            sub = list(group["path"]) if inner.group_by else []
            groups.append({"path": [str(year), *sub], "values": group["values"]})
    return {
        "groups": groups,
        "dimensions": ["year", *[dim["dimension"] for dim in inner.group_by]],
        "metrics": [{"id": item["id"], "label": item["label"], "type": item["type"]} for item in plan.metrics],
    }


def _execute(plan, data):
    dims = plan.group_by
    choices = [_dimension_values(dim, data)[0] for dim in dims]
    paths = list(product(*choices)) if choices else [("All",)]
    patients = data["patients"]
    all_ids = set(patients["patient_id"])
    concept_cache = {}
    group_columns = [f"_group_{index}" for index in range(len(dims))]
    patient_groups = pd.DataFrame({"patient_id": patients["patient_id"]}, index=patients.index)
    for column, dim in zip(group_columns, dims):
        if dim["dimension"] == "age_band":
            patient_groups[column] = _age_band_series(patients["age"], dim.get("bands", []))
        else:
            patient_groups[column] = patients[dim["dimension"]].to_numpy()

    group_patient_ids = {}
    if group_columns:
        for key, frame in patient_groups.groupby(group_columns, dropna=False, sort=False):
            group_patient_ids[_group_key(key, len(group_columns))] = set(frame["patient_id"])
    else:
        group_patient_ids[("All",)] = all_ids

    encounter_group_columns = []
    encounter_groups = None
    if dims:
        encounters = data["encounters"]
        encounter_groups = encounters[["patient_id", "data_holder"]].copy()
        encounter_groups["_encounter_index"] = encounters.index
        encounter_groups = encounter_groups.merge(patient_groups, on="patient_id", how="left", sort=False)
        for index, dim in enumerate(dims):
            name = f"_encounter_group_{index}"
            encounter_group_columns.append(name)
            if dim["dimension"] == "data_holder":
                encounter_groups[name] = encounter_groups["data_holder"]
            else:
                encounter_groups[name] = encounter_groups[group_columns[index]]

    mean_group_values = {}
    for metric in plan.metrics:
        if metric.get("type") != "mean":
            continue
        selected_ids = _population_ids_cached(
            metric.get("population", []), plan, data, all_ids, concept_cache
        )
        observations = _selected_observations(metric, selected_ids, plan, data)
        observations = observations.loc[observations["_numeric"].notna()]
        if not group_columns:
            values = observations["_numeric"].tolist()
            mean_group_values[metric["id"]] = {
                ("All",): (
                    sum(values, Decimal(0)) / Decimal(len(values)) if values else None,
                    len(values),
                )
            }
            continue
        grouped = observations[["patient_id", "_numeric"]].merge(
            patient_groups, on="patient_id", how="left", sort=False
        )
        values = {}
        for key, frame in grouped.groupby(group_columns, dropna=False, sort=False):
            decimals = frame["_numeric"].tolist()
            values[_group_key(key, len(group_columns))] = (
                sum(decimals, Decimal(0)) / Decimal(len(decimals)) if decimals else None,
                len(decimals),
            )
        mean_group_values[metric["id"]] = values

    encounter_group_values = {}
    for metric in plan.metrics:
        if metric.get("type") != "count_encounters":
            continue
        spec = plan.concepts.get(metric.get("encounter"), {})
        records = _encounters_for(spec, data)
        allowed_ids = _population_ids_cached(
            metric.get("population", []), plan, data, all_ids, concept_cache
        )
        records = records.loc[records["patient_id"].isin(allowed_ids)]
        if encounter_groups is not None:
            selected = encounter_groups.loc[
                encounter_groups["_encounter_index"].isin(records.index)
                & encounter_groups["patient_id"].isin(allowed_ids)
            ]
            counts = selected.groupby(
                encounter_group_columns, dropna=False, sort=False
            ).size()
            encounter_group_values[metric["id"]] = {
                _group_key(group_key, len(encounter_group_columns)): int(count)
                for group_key, count in counts.items()
            }
        else:
            encounter_group_values[metric["id"]] = {("All",): len(records)}

    groups = []
    for path in paths:
        metrics = {}
        group_ids = group_patient_ids.get(tuple(path), set())
        for metric in plan.metrics:
            kind = metric.get("type")
            if kind == "count_patients":
                ids = _population_ids_cached(
                    metric.get("population", []), plan, data, all_ids, concept_cache
                ).intersection(group_ids)
                value = len(ids)
                metrics[metric["id"]] = {"type": kind, "value": value, "n": value, "ids": ids}
            elif kind == "count_encounters":
                value = encounter_group_values.get(metric["id"], {}).get(tuple(path), 0)
                metrics[metric["id"]] = {"type": kind, "value": value, "n": value}
            elif kind == "percentage":
                numerator = _population_ids_cached(
                    metric.get("numerator_population", []), plan, data, all_ids, concept_cache
                ).intersection(group_ids)
                denominator = _population_ids_cached(
                    metric.get("denominator_population", []), plan, data, all_ids, concept_cache
                ).intersection(group_ids)
                n, d = len(numerator), len(denominator)
                value = Decimal(n) * Decimal(100) / Decimal(d) if d else None
                metrics[metric["id"]] = {"type": kind, "value": value, "numerator": n, "denominator": d,
                                         "numerator_ids": numerator, "denominator_ids": denominator}
            elif kind == "mean":
                value, count = mean_group_values.get(metric["id"], {}).get(tuple(path), (None, 0))
                metrics[metric["id"]] = {"type": kind, "value": value, "n": count}
        groups.append({"path": list(path), "values": metrics})
    return {
        "groups": groups,
        "dimensions": [dim["dimension"] for dim in dims],
        "metrics": [{"id": item["id"], "label": item["label"], "type": item["type"]} for item in plan.metrics],
    }
