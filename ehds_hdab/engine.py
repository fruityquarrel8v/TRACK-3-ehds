from decimal import Decimal, InvalidOperation
from itertools import product

import pandas as pd

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
            reference = pd.Timestamp(REFERENCE_DATE)
            mask &= medications["start_date"].le(reference)
            mask &= medications["end_date"].isna() | medications["end_date"].ge(reference)
        return set(medications.loc[mask, "patient_id"])
    if kind == "observation":
        return set(data["observations"].loc[data["observations"]["loinc"].eq(str(spec.get("loinc", ""))), "patient_id"])
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
    return [], None


def _selected_observations(metric, population, plan, data):
    key = metric.get("observation")
    spec = plan.concepts.get(key, {})
    frame = data["observations"]
    frame = frame.loc[
        frame["patient_id"].isin(population)
        & frame["loinc"].eq(str(spec.get("loinc", "")))
    ].copy()
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
                metrics[metric["id"]] = {"type": kind, "value": value, "n": value}
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
                metrics[metric["id"]] = {"type": kind, "value": value, "numerator": n, "denominator": d}
            elif kind == "mean":
                value, count = mean_group_values.get(metric["id"], {}).get(tuple(path), (None, 0))
                metrics[metric["id"]] = {"type": kind, "value": value, "n": count}
        groups.append({"path": list(path), "values": metrics})
    return {
        "groups": groups,
        "dimensions": [dim["dimension"] for dim in dims],
        "metrics": [{"id": item["id"], "label": item["label"], "type": item["type"]} for item in plan.metrics],
    }
