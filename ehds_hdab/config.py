import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DATE = "2025-12-31"
SMALL_CELL_THRESHOLD = 10
DATA_DIR = Path(os.environ.get("EHDS_DATA_DIR", ROOT / "data")).expanduser()
SPE_ROOT = Path(os.environ.get("EHDS_SPE_ROOT", ROOT / "spe_workspaces")).expanduser()
AUDIT_LOG = Path(os.environ.get("EHDS_AUDIT_LOG", ROOT / "audit_log.jsonl")).expanduser()
LEDGER_PATH = Path(os.environ.get("EHDS_LEDGER_PATH", ROOT / "disclosure_ledger.jsonl")).expanduser()
DISCLOSURE_AUDIENCE = os.environ.get("EHDS_DISCLOSURE_AUDIENCE", "shared")
