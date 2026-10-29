from dataclasses import asdict
import time

from .audit import audit_event
from .data import load_data
from .decision import decide
from .engine import execute
from .parser import parse_request
from .plan import QueryPlan
from .privacy import apply_small_cell, check_statistical_output
from .spe import create_workspace


def process_request(request):
    started = time.perf_counter()
    decision = decide(request)
    data = load_data()
    if decision.decision == "REJECT":
        plan = QueryPlan()
        answer = None
        spe_info = None
        warnings = []
        privacy_check = None
    else:
        plan = parse_request(request)
        warnings = list(plan.warnings)
        if not decision.purpose_category:
            warnings.append("Purpose was not matched to a specific Art. 53 category.")
        if decision.decision == "APPROVE_SPE":
            spe_info = create_workspace(
                request.get("id", "unknown"),
                plan,
                data,
                articles=decision.articles,
                created_at=request.get("submitted"),
            ).to_dict()
            answer = {"cohort_size": spe_info["cohort_size"]}
            privacy_check = None
        else:
            raw = execute(plan, data)
            answer = apply_small_cell(raw)
            privacy_check = {"passed": not check_statistical_output(answer), "violations": check_statistical_output(answer)}
            if plan.patient_level_listing:
                warnings.append("Patient-level listings are not provided in statistical answers.")
            spe_info = None
    needs_staff_review = (
        decision.decision != "REJECT"
        and bool(warnings or plan.empty or not decision.purpose_category)
    )
    response = {
        "request_id": request.get("id"),
        "decision": decision.decision,
        "articles": sorted(decision.articles, key=str),
        "reason": decision.reason,
        "purpose_category": decision.purpose_category,
        "answer": answer,
        "query_plan": plan.to_dict(),
        "spe": spe_info,
        "warnings": warnings,
        "needs_staff_review": needs_staff_review,
        "processing_time_s": time.perf_counter() - started,
        **({"privacy_check": privacy_check} if privacy_check is not None else {}),
    }
    audit_event(
        "request_processed",
        request.get("id"),
        {
            "decision": response["decision"],
            "articles": response["articles"],
            "purpose_category": response["purpose_category"],
            "needs_staff_review": response["needs_staff_review"],
            "processing_time_s": response["processing_time_s"],
        },
    )
    return response


def process_file(path):
    import json

    with open(path, encoding="utf-8") as stream:
        return process_request(json.load(stream))
