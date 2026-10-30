from dataclasses import asdict
import time

from .audit import audit_event
from .data import load_data
from .decision import decide
from .engine import execute
from . import disclosure
from .parser import parse_request
from .plan import QueryPlan
from .privacy import apply_small_cell, check_statistical_output
from .spe import create_workspace


def reset_query_history():
    disclosure.reset_ledger()


def _applicant_key(request):
    applicant = request.get("applicant")
    if not isinstance(applicant, dict):
        return None
    name = str(applicant.get("name") or "").strip()
    organisation = str(applicant.get("organisation") or "").strip()
    if not name and not organisation:
        return None
    return f"{name}|{organisation}".casefold()


def _condition_source_warnings(plan, data):
    referenced = set()

    def collect(items):
        for item in items or []:
            key = item.get("concept")
            spec = plan.concepts.get(key, {})
            if spec.get("kind") == "condition":
                referenced.add(key)
            elif spec.get("kind") == "group":
                collect(spec.get("items"))

    for metric in plan.metrics:
        for field in ("population", "numerator_population", "denominator_population"):
            collect(metric.get(field))
    collect(plan.cohort)

    condition_data = data.get("conditions")
    conditions = (
        condition_data["icd10"].astype(str)
        if condition_data is not None and "icd10" in condition_data
        else None
    )
    encounter_data = data.get("encounters")
    encounter_diagnoses = (
        encounter_data["primary_icd10"].astype(str)
        if encounter_data is not None and "primary_icd10" in encounter_data
        else None
    )
    warnings = []
    for key in sorted(referenced):
        prefixes = plan.concepts[key].get("icd10_prefixes", [])
        if not prefixes:
            continue
        if conditions is not None and conditions.str.startswith(tuple(prefixes)).any():
            continue
        if (
            encounter_diagnoses is not None
            and encounter_diagnoses.str.startswith(tuple(prefixes)).any()
        ):
            warnings.append(
                f"Condition '{key}' appears only as an encounter main diagnosis "
                "(primary_icd10), not in the diagnosis list; counted from the "
                "diagnosis list per the data rules — the request may mean "
                "hospital stays. Staff review required."
            )
    return warnings


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
        withheld = False
    else:
        plan = parse_request(request)
        for warning in _condition_source_warnings(plan, data):
            if warning not in plan.warnings:
                plan.warnings.append(warning)
        warnings = list(plan.warnings)
        if not decision.purpose_category:
            warnings.append("Purpose was not matched to a specific Art. 53 category.")
        withheld = False
        withheld_info = None
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
            applicant_key = _applicant_key(request)
            assessment = disclosure.assess(request.get("id"), applicant_key, raw, plan)
            if not assessment["safe"]:
                withheld = True
                withheld_info = assessment
                disclosure.record(request.get("id"), applicant_key, raw, answer, plan, "withheld")
                answer = None
                privacy_check = {"passed": True, "violations": []}
                first = assessment["conflicts"][0]
                warnings.append("Results withheld: " + assessment["explanation"])
                audit_event("differencing_withheld", request.get("id"), {
                    "earlier_request": first["earlier_request"], "relation": first["relation"],
                    "cell": first["cell"], "metric": first["metric"]})
            else:
                violations = check_statistical_output(answer)
                privacy_check = {"passed": not violations, "violations": violations}
                disclosure.record(request.get("id"), applicant_key, raw, answer, plan, "released")
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
        **({"withheld": True, "withheld_reason": withheld_info["explanation"],
            "safe_alternatives": withheld_info["alternatives"],
            "conflicts": withheld_info["conflicts"]} if withheld else {}),
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
