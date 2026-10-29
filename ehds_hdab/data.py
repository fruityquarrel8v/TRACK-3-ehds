from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, REFERENCE_DATE

TABLES = ("patients", "conditions", "medications", "observations", "encounters")


@lru_cache(maxsize=1)
def load_data():
    root = Path(DATA_DIR)
    frames = {
        name: pd.read_csv(root / f"{name}.csv", dtype=str, keep_default_na=False)
        for name in TABLES
    }
    category_lists = {
        column: sorted(frames["patients"][column].dropna().astype(str).unique().tolist())
        for column in ("region", "data_holder", "sex")
    }
    opted_out = frames["patients"]["opted_out"].astype(str).str.casefold().eq("true")
    opted_out_ids = set(frames["patients"].loc[opted_out, "patient_id"])
    frames["patients"] = frames["patients"].loc[~opted_out].copy()
    for name in TABLES[1:]:
        frames[name] = frames[name].loc[~frames[name]["patient_id"].isin(opted_out_ids)].copy()
    frames["patients"]["birth_date"] = pd.to_datetime(
        frames["patients"]["birth_date"], errors="coerce"
    )
    ref_year = pd.Timestamp(REFERENCE_DATE).year
    frames["patients"]["age"] = ref_year - pd.to_numeric(
        frames["patients"]["birth_date"].dt.year, errors="coerce"
    )
    for name, column in (
        ("conditions", "onset_date"),
        ("medications", "start_date"),
        ("medications", "end_date"),
        ("observations", "date"),
        ("encounters", "start_date"),
        ("encounters", "end_date"),
    ):
        frames[name][column] = pd.to_datetime(frames[name][column], errors="coerce")
    frames["categories"] = category_lists
    return frames
