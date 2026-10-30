"""Disclosure ledger: remembers which statistics were released and blocks simple differencing attacks.

Every released count cell is stored with a one-way hash of its patient IDs (never the IDs themselves).
Before a new table is released, each of its cells is compared with every earlier released cell: if one
population is a subset of the other and they differ by fewer than SMALL_CELL_THRESHOLD patients, the two
counts together would reveal a protected small group, so the new table is withheld.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from . import config
from .config import SMALL_CELL_THRESHOLD

EXPLANATION = (
    "This result cannot be released safely. Although the requested group contains enough patients, "
    "combining this result with previously released statistics could reveal information about a "
    "smaller protected group."
)
ALTERNATIVES = [
    "Broaden the population (fewer or wider criteria).",
    "Remove a narrow geographic filter such as a single region or hospital.",
    "Use a less detailed grouping.",
    "Request an alternative statistical summary.",
]


def _ledger_path():
    return Path(config.LEDGER_PATH)


def _hash_id(patient_id):
    return hashlib.sha256(f"ehds-ledger:{patient_id}".encode("utf-8")).hexdigest()[:16]


def _digest(hashes):
    return hashlib.sha256("".join(sorted(hashes)).encode("utf-8")).hexdigest()


def reset_ledger():
    path = _ledger_path()
    if path.exists():
        path.unlink()


def load_ledger():
    path = _ledger_path()
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def ledger_summary():
    """Ledger rows without patient hashes, for staff display."""
    return [{k: v for k, v in entry.items() if k != "hashes"} for entry in load_ledger()]


def count_cells(raw):
    """(metric_id, label, path, patient_ids) for every count cell with an exact patient set."""
    labels = {item["id"]: item.get("label", item["id"]) for item in raw.get("metrics", [])}
    for group in raw.get("groups", []):
        path = list(group.get("path") or ["All"])
        for metric_id, result in group.get("values", {}).items():
            if result.get("type") == "count_patients" and "ids" in result:
                yield metric_id, labels.get(metric_id, metric_id), path, result["ids"]
            elif result.get("type") == "percentage":
                for part in ("numerator", "denominator"):
                    if f"{part}_ids" in result:
                        yield f"{metric_id}:{part}", f"{labels.get(metric_id, metric_id)} ({part})", path, result[f"{part}_ids"]


def _population_text(plan):
    return json.dumps({"cohort": plan.cohort, "concepts": plan.concepts, "group_by": plan.group_by},
                      sort_keys=True, ensure_ascii=False, default=str)


def assess(request_id, applicant_key, raw, plan):
    """Compare the new count cells with released cells in the ledger. Returns a dict with 'safe'."""
    audience = getattr(config, "DISCLOSURE_AUDIENCE", "shared")
    released = [
        e for e in load_ledger()
        if e.get("status") == "released" and e.get("hashes") is not None
        and (audience != "applicant" or e.get("applicant_key") == applicant_key)
    ]
    conflicts = []
    for metric_id, label, path, ids in count_cells(raw):
        if len(ids) < SMALL_CELL_THRESHOLD:
            continue  # shown as <10, releases no exact count
        hashes = {_hash_id(pid) for pid in ids}
        digest = _digest(hashes)
        for entry in released:
            if entry.get("request_id") == request_id and entry.get("digest") == digest:
                continue
            other_size = entry.get("size", 0)
            gap = abs(other_size - len(hashes))
            if gap == 0 or gap >= SMALL_CELL_THRESHOLD or entry.get("digest") == digest:
                continue
            other = set(entry["hashes"])
            if hashes < other or other < hashes:
                conflicts.append({
                    "earlier_request": entry.get("request_id"),
                    "earlier_cell": entry.get("path"),
                    "earlier_metric": entry.get("metric"),
                    "cell": path,
                    "metric": label,
                    "relation": "subset" if hashes < other else "superset",
                })
                break
    return {
        "safe": not conflicts,
        "conflicts": conflicts,
        "explanation": EXPLANATION if conflicts else "",
        "alternatives": list(ALTERNATIVES) if conflicts else [],
    }


def record(request_id, applicant_key, raw, answer_shown, plan, status):
    """Append one ledger row per count cell. Released numeric cells carry their patient hashes."""
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    population = _population_text(plan)
    existing = {(e.get("request_id"), tuple(e.get("path") or []), e.get("metric"), e.get("digest"))
                for e in load_ledger()}
    with path.open("a", encoding="utf-8") as stream:
        for metric_id, label, cell_path, ids in count_cells(raw):
            hashes = sorted(_hash_id(pid) for pid in ids)
            digest = _digest(hashes)
            if (request_id, tuple(cell_path), label, digest) in existing:
                continue
            shown = None
            node = answer_shown or {}
            for key in cell_path:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            if isinstance(node, dict):
                shown = node.get(label.split(" (")[0])
            released_exact = status == "released" and len(ids) >= SMALL_CELL_THRESHOLD
            stream.write(json.dumps({
                "request_id": request_id,
                "applicant_key": applicant_key,
                "timestamp": timestamp,
                "population": population,
                "path": cell_path,
                "metric": label,
                "size": len(ids),
                "released": shown if status == "released" else None,
                "status": status if released_exact or status != "released" else "suppressed",
                "suppression": f"counts below {SMALL_CELL_THRESHOLD} shown as <{SMALL_CELL_THRESHOLD}",
                "digest": digest,
                "hashes": hashes if released_exact else None,
            }, ensure_ascii=False, sort_keys=True) + "\n")
