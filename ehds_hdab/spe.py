from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from .audit import audit_event
from .config import SPE_ROOT
from .engine import population_patient_ids


class DownloadBlockedError(RuntimeError):
    pass


@dataclass
class SPEInfo:
    path: str
    cohort_size: int
    files: dict[str, int]
    manifest_path: str

    def to_dict(self):
        return asdict(self)


def _safe_request_id(request_id):
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", str(request_id or "unknown"))
    if value in ("", ".", ".."):
        return "unknown"
    return value


def create_workspace(request_id, plan, data, articles=None, created_at=None):
    safe_id = _safe_request_id(request_id)
    root = Path(SPE_ROOT).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    workspace = (root / safe_id).resolve()
    if root not in workspace.parents:
        raise ValueError("Invalid request id for workspace")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    cohort_ids = population_patient_ids(plan.cohort, plan, data)
    cohort = data["patients"].loc[data["patients"]["patient_id"].isin(cohort_ids), ["patient_id"]].copy()
    row_counts = {}
    cohort_file = workspace / "cohort.csv"
    cohort.to_csv(cohort_file, index=False)
    row_counts[cohort_file.name] = len(cohort)
    observation_codes = [
        spec.get("loinc") for spec in plan.concepts.values()
        if spec.get("kind") == "observation" and spec.get("loinc")
    ]
    table_map = {
        "conditions": "conditions",
        "medications": "medications",
        "observations": "observations",
        "encounters": "encounters",
    }
    for record_type in plan.record_types:
        table = table_map.get(record_type)
        if not table:
            continue
        rows = data[table].loc[data[table]["patient_id"].isin(cohort_ids)].copy()
        if table == "observations" and observation_codes:
            rows = rows.loc[rows["loinc"].isin(observation_codes)]
        file_path = workspace / f"{table}.csv"
        rows.to_csv(file_path, index=False)
        row_counts[file_path.name] = len(rows)
    manifest = {
        "request_id": str(request_id),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "permit": {"articles": list(articles or [])},
        "articles": list(articles or []),
        "cohort_size": len(cohort_ids),
        "files": row_counts,
        "download_allowed": False,
    }
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_event("spe_workspace_created", request_id, {"cohort_size": len(cohort_ids)})
    return SPEInfo(str(workspace), len(cohort_ids), row_counts, str(manifest_path))


class SecureProcessingEnvironment:
    def __init__(self, workspace, request_id=None):
        self.workspace = Path(workspace).expanduser().resolve()
        self.request_id = request_id or self.workspace.name

    def list_tables(self):
        return sorted(path.stem for path in self.workspace.glob("*.csv"))

    def read_table(self, table):
        name = str(table)
        if not name.endswith(".csv"):
            name += ".csv"
        path = (self.workspace / name).resolve()
        if self.workspace not in path.parents or path.suffix != ".csv" or not path.exists():
            raise FileNotFoundError(name)
        import pandas as pd

        return pd.read_csv(path, dtype=str, keep_default_na=False)

    def request_download(self, *args, **kwargs):
        audit_event("download_blocked", self.request_id, {"reason": "downloads are not permitted"})
        raise DownloadBlockedError("Downloads from the secure processing environment are blocked")
