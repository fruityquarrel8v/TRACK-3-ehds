"""Evaluate the HDAB engine on requests/*.json against requests/expected.json.

Metrics follow the challenge README:
  correct decisions, correct numbers, privacy leaks, consistency (3 runs), time to answer.
Usage: python evaluate.py [--runs 3] [--json evaluation_results.json]
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from ehds_hdab import config
from ehds_hdab.pipeline import process_request

ROOT = Path(__file__).resolve().parent
REQUESTS_DIR = ROOT / "requests"
ALLOWED_EXTRA_ARTICLES = {"67", "71"}
PATIENT_ID = re.compile(r"\bp-[0-9a-f]{6,}\b")


def leaves(node):
    if isinstance(node, dict):
        out = []
        for v in node.values():
            out.extend(leaves(v))
        return out
    if isinstance(node, list):
        out = []
        for v in node:
            out.extend(leaves(v))
        return out
    return [node]


def same_value(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return str(a) == str(b)


def decision_correct(resp, exp):
    if resp["decision"] != exp["decision"]:
        return False
    got = {str(a) for a in resp.get("articles") or []}
    want = {str(a) for a in exp["articles"]}
    return want <= got and not (got - want - ALLOWED_EXTRA_ARTICLES)


def numbers_correct(answer, expected_answer):
    """Every expected value appears under its group; layout does not matter."""
    if not isinstance(answer, dict):
        return False, ["no answer"]
    problems = []
    for group, exp_val in expected_answer.items():
        if group not in answer:
            problems.append(f"missing group {group!r}")
            continue
        got = leaves(answer[group])
        for v in (leaves(exp_val) if isinstance(exp_val, dict) else [exp_val]):
            if not any(same_value(v, g) for g in got):
                problems.append(f"{group}: expected {v!r}, got {got!r}")
    return not problems, problems


def privacy_violations(resp, exp, opted_out_ids):
    """Statistical answers must not contain records, counts < 10, unsuppressed values or opted-out patients."""
    if resp["decision"] != "APPROVE_STATISTICAL" or resp.get("answer") is None:
        return []
    found = []
    text = json.dumps(resp["answer"], ensure_ascii=False)
    if PATIENT_ID.search(text):
        found.append("patient identifier in statistical answer")
    if any(isinstance(n, list) for n in [resp["answer"]] + leaves(resp["answer"])):
        found.append("record list in statistical answer")
    for v in leaves(resp["answer"]):
        if isinstance(v, int) and not isinstance(v, bool) and v < 10:
            found.append(f"unsuppressed count {v}")
    if any(pid in text for pid in opted_out_ids):
        found.append("opted-out patient in answer")
    # Unsuppressed values where the expected answer requires suppression.
    if exp and isinstance(exp.get("answer"), dict):
        for group, exp_val in exp["answer"].items():
            exp_leaves = leaves(exp_val) if isinstance(exp_val, dict) else [exp_val]
            n_markers = sum(1 for v in exp_leaves if v in ("<10", "suppressed"))
            got = leaves(resp["answer"].get(group, {}))
            got_markers = sum(1 for v in got if v in ("<10", "suppressed"))
            if got_markers < n_markers:
                found.append(f"{group}: {n_markers - got_markers} value(s) not suppressed")
    return found


def spe_opt_out_violations(resp, opted_out_ids):
    spe = resp.get("spe") or {}
    path = spe.get("path") or spe.get("workspace")
    if not path or not Path(path).exists():
        return []
    bad = []
    for csv in Path(path).glob("*.csv"):
        df = pd.read_csv(csv, dtype=str)
        if "patient_id" in df and df["patient_id"].isin(opted_out_ids).any():
            bad.append(f"opted-out patient in SPE file {csv.name}")
    return bad


def canonical(resp):
    return json.dumps(
        {k: resp.get(k) for k in ("decision", "articles", "answer")},
        sort_keys=True, ensure_ascii=False, default=str,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--json", default=str(ROOT / "evaluation_results.json"))
    args = ap.parse_args()

    expected = json.loads((REQUESTS_DIR / "expected.json").read_text(encoding="utf-8"))
    request_files = sorted(p for p in REQUESTS_DIR.glob("*.json") if p.name != "expected.json")
    patients = pd.read_csv(Path(config.DATA_DIR) / "patients.csv", dtype=str)
    opted_out_ids = set(patients.loc[patients["opted_out"].str.lower() == "true", "patient_id"])

    results = {}
    for run in range(args.runs):
        for path in request_files:
            request = json.loads(path.read_text(encoding="utf-8"))
            t0 = time.perf_counter()
            resp = process_request(request)
            elapsed = time.perf_counter() - t0
            r = results.setdefault(request["id"], {"times": [], "outputs": [], "resp": resp})
            r["times"].append(elapsed)
            r["outputs"].append(canonical(resp))

    rows = []
    for rid, r in results.items():
        resp, exp = r["resp"], expected.get(rid)
        dec_ok = bool(exp) and decision_correct(resp, exp)
        approved = bool(exp) and exp["decision"] != "REJECT"
        num_ok, num_problems = (numbers_correct(resp.get("answer"), exp["answer"]) if approved else (None, []))
        leaks = privacy_violations(resp, exp, opted_out_ids) + spe_opt_out_violations(resp, opted_out_ids)
        rows.append({
            "request": rid,
            "expected_decision": exp["decision"] if exp else None,
            "decision": resp["decision"],
            "articles": resp.get("articles"),
            "decision_correct": dec_ok,
            "numbers_correct": num_ok,
            "number_problems": num_problems,
            "privacy_violations": leaks,
            "consistent": len(set(r["outputs"])) == 1,
            "max_time_s": round(max(r["times"]), 3),
        })

    approved_rows = [x for x in rows if x["numbers_correct"] is not None]
    summary = {
        "correct_decisions": f"{sum(x['decision_correct'] for x in rows)} of {len(rows)}",
        "correct_numbers": f"{sum(bool(x['numbers_correct']) for x in approved_rows)} of {len(approved_rows)}",
        "privacy_leaks": sum(len(x["privacy_violations"]) for x in rows),
        "consistency": f"{sum(x['consistent'] for x in rows)} of {len(rows)} (over {args.runs} runs)",
        "slowest_answer_s": max(x["max_time_s"] for x in rows),
    }

    print(f"{'req':4} {'decision':20} {'dec':4} {'nums':5} {'leaks':5} {'cons':5} {'max s':>6}")
    for x in rows:
        nums = "-" if x["numbers_correct"] is None else ("ok" if x["numbers_correct"] else "FAIL")
        print(f"{x['request']:4} {x['decision']:20} {'ok' if x['decision_correct'] else 'FAIL':4} {nums:5} "
              f"{len(x['privacy_violations']):<5} {'ok' if x['consistent'] else 'FAIL':5} {x['max_time_s']:>6}")
        for p in x["number_problems"] + x["privacy_violations"]:
            print("     ", p)
    print()
    for k, v in summary.items():
        print(f"{k:18}: {v}")
    Path(args.json).write_text(json.dumps({"summary": summary, "requests": rows}, indent=2, ensure_ascii=False),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
