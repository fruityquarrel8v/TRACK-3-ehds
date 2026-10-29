from decimal import Decimal, ROUND_HALF_UP
import re

from .config import SMALL_CELL_THRESHOLD


def round_half_up(value, places=1):
    quantum = Decimal("1").scaleb(-places)
    return Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)


def _rounded_number(value):
    return float(round_half_up(value, 1))


def apply_small_cell(raw):
    metrics = {item["id"]: item for item in raw.get("metrics", [])}
    answer = {}
    for group in raw.get("groups", []):
        path = group.get("path") or ["All"]
        values = {}
        for metric_id, result in group.get("values", {}).items():
            metric = metrics.get(metric_id, {})
            kind = result.get("type", metric.get("type"))
            value = result.get("value")
            if kind in ("count_patients", "count_encounters"):
                shown = f"<{SMALL_CELL_THRESHOLD}" if value < SMALL_CELL_THRESHOLD else int(value)
            elif kind == "percentage":
                numerator = result.get("numerator", 0)
                denominator = result.get("denominator", 0)
                if denominator == 0 or numerator < SMALL_CELL_THRESHOLD or denominator < SMALL_CELL_THRESHOLD:
                    shown = "suppressed"
                else:
                    shown = _rounded_number(value)
            elif kind == "mean":
                if result.get("n", 0) < SMALL_CELL_THRESHOLD or value is None:
                    shown = "suppressed"
                else:
                    shown = _rounded_number(value)
            else:
                shown = value
            values[metric.get("label", metric_id)] = shown
        node = answer
        for label in path[:-1]:
            node = node.setdefault(label, {})
        if path:
            node[path[-1]] = values
        else:
            answer["All"] = values
    return answer


def check_statistical_output(answer):
    violations = []

    def visit(value):
        if isinstance(value, list):
            violations.append("list of records in statistical output")
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, int) and not isinstance(value, bool) and value < SMALL_CELL_THRESHOLD:
            violations.append(f"unsuppressed count below {SMALL_CELL_THRESHOLD}: {value}")
        elif isinstance(value, str) and re.search(r"\bp-[0-9a-f]+\b", value, re.I):
            violations.append("patient identifier in statistical output")

    visit(answer)
    return violations
