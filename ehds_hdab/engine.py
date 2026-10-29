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


def _age_band(age, bands):
    if pd.isna(age):
        return None
    for band in bands:
        low, high = band.get("lo"), band.get("hi")
        if (low is None or age >= low) and (high is None or age <= high):
            return band["label"]
    return None


def _dimension_values(dim, data):
    name = dim["dimension"]
    if name == "age_band":
        bands = dim.get("bands") or []
        return [band["label"] for band in bands], bands
    if name in ("region", "data_holder", "sex"):
        return data["categories"].get(name, []), None
    return [], None


def _path_for_patient(patient, dims):
    path = []
    for dim in dims:
        name = dim["dimension"]
        if name == "age_band":
            path.append(_age_band(patient.get("age"), dim.get("bands", [])))
        else:
            path.append(patient.get(name))
    return path


def _group_patient_ids(path, dims, data):
    selected = set()
    for row in data["patients"].to_dict("records"):
        if _path_for_patient(row, dims) == list(path):
            selected.add(row["patient_id"])
    return selected


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
        ascending = pick == "min"
        return frame.sort_values(["_numeric", "date"]).drop_duplicates("patient_id", keep="first")
    ascending = pick == "earliest"
    return frame.sort_values(["date"], ascending=ascending).drop_duplicates("patient_id", keep="first")


def _to_decimal(value):
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _group_metric_population(population, path, dims, plan, data):
    selected = population_patient_ids(population, plan, data)
    patient_group = _group_patient_ids(path, dims, data) if dims else set(data["patients"]["patient_id"])
    return selected.intersection(patient_group)


def _path_for_encounter(record, patient, dims):
    path = []
    for dim in dims:
        if dim["dimension"] == "data_holder":
            path.append(record.data_holder)
        elif dim["dimension"] == "age_band":
            path.append(_age_band(patient.get("age"), dim.get("bands", [])))
        else:
            path.append(patient.get(dim["dimension"]))
    return path


def execute(plan, data):
    dims = plan.group_by
    choices = [_dimension_values(dim, data)[0] for dim in dims]
    paths = list(product(*choices)) if choices else [("All",)]
    groups = []
    patient_lookup = data["patients"].set_index("patient_id", drop=False)
    for path in paths:
        metrics = {}
        group_ids = _group_patient_ids(path, dims, data) if dims else set(data["patients"]["patient_id"])
        for metric in plan.metrics:
            kind = metric.get("type")
            if kind == "count_patients":
                ids = population_patient_ids(metric.get("population", []), plan, data).intersection(group_ids)
                value = len(ids)
                metrics[metric["id"]] = {"type": kind, "value": value, "n": value}
            elif kind == "count_encounters":
                key = metric.get("encounter")
                spec = plan.concepts.get(key, {})
                records = _encounters_for(spec, data)
                allowed_ids = population_patient_ids(metric.get("population", []), plan, data)
                records = records.loc[records["patient_id"].isin(allowed_ids)]
                if not records.empty and dims:
                    patient_attrs = data["patients"].set_index("patient_id")
                    keep = []
                    for row in records.itertuples():
                        patient = patient_attrs.loc[row.patient_id].to_dict() if row.patient_id in patient_attrs.index else {}
                        if _path_for_encounter(row, patient, dims) == list(path):
                            keep.append(row.Index)
                    records = records.loc[keep]
                value = len(records)
                metrics[metric["id"]] = {"type": kind, "value": value, "n": value}
            elif kind == "percentage":
                numerator = _group_metric_population(metric.get("numerator_population", []), path, dims, plan, data)
                denominator = _group_metric_population(metric.get("denominator_population", []), path, dims, plan, data)
                n, d = len(numerator), len(denominator)
                value = Decimal(n) * Decimal(100) / Decimal(d) if d else None
                metrics[metric["id"]] = {"type": kind, "value": value, "numerator": n, "denominator": d}
            elif kind == "mean":
                population = _group_metric_population(metric.get("population", []), path, dims, plan, data)
                observations = _selected_observations(metric, population, plan, data)
                if not observations.empty and dims:
                    ids = set()
                    for patient_id in observations["patient_id"]:
                        patient = patient_lookup.loc[patient_id].to_dict()
                        if _path_for_patient(patient, dims) == list(path):
                            ids.add(patient_id)
                    observations = observations.loc[observations["patient_id"].isin(ids)]
                values = [v for v in observations["_numeric"].tolist() if v is not None]
                count = len(values)
                value = sum(values, Decimal(0)) / Decimal(count) if count else None
                metrics[metric["id"]] = {"type": kind, "value": value, "n": count}
        groups.append({"path": list(path), "values": metrics})
    return {
        "groups": groups,
        "dimensions": [dim["dimension"] for dim in dims],
        "metrics": [{"id": item["id"], "label": item["label"], "type": item["type"]} for item in plan.metrics],
    }
