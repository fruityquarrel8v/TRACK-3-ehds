from datetime import datetime, timezone
import json
from pathlib import Path

from .config import AUDIT_LOG


def audit_event(event, request_id, details=None):
    path = Path(AUDIT_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": str(event),
        "request_id": str(request_id),
        "details": details or {},
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
