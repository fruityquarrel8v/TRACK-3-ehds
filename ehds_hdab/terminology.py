from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class Concept:
    term: str
    kind: str
    codes: tuple[str, ...]


_TERMS = {
    "condition": {
        "type 2 diabetes": ("E11", "t2dm", "diabetes mellitus type 2"),
        "hypertension": ("I10", "high blood pressure"),
        "atrial fibrillation": ("I48", "af", "afib"),
        "heart failure": ("I50",),
        "copd": ("J44", "chronic obstructive pulmonary disease"),
        "asthma": ("J45",),
        "depression": ("F32", "depressive episode"),
        "obesity": ("E66",),
        "breast cancer": ("C50",),
        "stroke": ("I63", "cerebral infarction"),
        "chronic kidney disease": ("N18", "ckd"),
        "pneumonia": ("J18",),
        "hip fracture": ("S72", "femur fracture"),
        "appendicitis": ("K35",),
        "urinary tract infection": ("N39", "uti"),
    },
    "medication": {
        "metformin": ("A10BA02",),
        "insulin": ("A10A",),
        "glp-1 receptor agonist": ("A10BJ", "glp-1", "semaglutide"),
        "sglt2 inhibitor": ("A10BK", "empagliflozin"),
        "oral anticoagulant": ("B01AA", "B01AF"),
        "doac": ("B01AF",),
        "vitamin k antagonist": ("B01AA", "phenprocoumon"),
        "loop diuretic": ("C03CA", "furosemide"),
        "beta blocker": ("C07", "bisoprolol"),
        "calcium channel blocker": ("C08CA", "amlodipine"),
        "ace inhibitor": ("C09AA", "ramipril"),
        "arb": ("C09CA", "angiotensin receptor blocker", "candesartan"),
        "statin": ("C10AA", "atorvastatin"),
        "antidepressant": ("N06A",),
        "ssri": ("N06AB", "sertraline"),
        "inhaled salmeterol fluticasone": ("R03AK", "salmeterol", "fluticasone"),
        "lama": ("R03BB", "tiotropium"),
    },
    "observation": {
        "hba1c": ("4548-4", "glycated haemoglobin", "glycated hemoglobin"),
        "bmi": ("39156-5", "body mass index"),
        "systolic blood pressure": ("8480-6",),
        "egfr": ("62238-1", "kidney function"),
    },
}

_CODES = {
    "oral anticoagulant": ("B01AA", "B01AF"),
}


def normalize(term):
    value = unicodedata.normalize("NFKC", str(term)).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    return re.sub(r"\s+", " ", value)


def _variants(term):
    value = normalize(term)
    candidates = [value]
    if value.endswith("ies") and len(value) > 4:
        candidates.append(value[:-3] + "y")
    if value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        candidates.append(value[:-1])
    return candidates


def entries():
    for kind, concepts in _TERMS.items():
        for canonical, values in concepts.items():
            codes = _CODES.get(canonical, values[:1])
            for alias in (canonical, *values[1:]):
                yield alias, Concept(canonical, kind, tuple(codes))


def lookup(term, data=None):
    variants = set(_variants(term))
    for alias, concept in entries():
        if variants.intersection(_variants(alias)):
            return concept
    if data is None:
        try:
            from .data import load_data

            data = load_data()
        except (FileNotFoundError, KeyError):
            data = None
    if data:
        for table, column, kind, code_col in (
            ("conditions", "description", "condition", "icd10"),
            ("medications", "name", "medication", "atc"),
            ("observations", "name", "observation", "loinc"),
        ):
            frame = data.get(table)
            if frame is None or column not in frame:
                continue
            labels = frame[column].astype(str)
            matches = labels.str.casefold().eq(str(term).casefold())
            if not matches.any():
                matches = labels.str.casefold().str.contains(
                    re.escape(str(term).casefold()), regex=True, na=False
                )
            if matches.any():
                codes = tuple(
                    frame.loc[matches, code_col].dropna().astype(str).unique().tolist()
                )
                return Concept(str(term), kind, codes)
    return None
