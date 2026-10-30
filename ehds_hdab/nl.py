from dataclasses import dataclass
import re

from .data import load_data
from .terminology import Concept, display_name, entries


_STOPWORDS = {
    "a", "about", "above", "across", "active", "admitted", "admission",
    "admissions", "adult", "adults", "after", "aged", "and", "any", "are",
    "as", "at", "average", "band", "bands", "below", "between", "by",
    "among", "cases", "children", "compare", "compared", "comparison",
    "condition", "conditions", "count", "distribution", "during", "each",
    "every", "ever", "exclude", "female", "females",
    "for", "from", "gender", "greater", "group", "grouping", "had", "has",
    "having", "have", "health", "how", "hospital", "hospitalised",
    "hospitalized", "hospitalisation", "hospitalisations", "hospitalization",
    "hospitalizations", "in", "include", "incident", "inpatient", "is",
    "individuals", "latest", "level", "levels", "less", "male", "males",
    "many", "mean",
    "measurement", "measurements", "men", "new", "newly", "no", "not",
    "number", "of", "on", "or", "overall", "older", "over", "patients",
    "per", "period", "people", "prescribed", "prescription", "prevalence",
    "region", "regions", "results", "sex", "show", "since", "stay", "stays",
    "plot", "receiving", "receive", "taking", "take", "takes", "took", "the",
    "there", "those", "threshold", "time", "to", "trend", "trends",
    "under", "use", "used", "uses", "using", "value", "values", "was",
    "were", "what", "when", "where", "which", "who", "with", "within",
    "women", "years", "diagnosed",
    "diagnosis", "diagnoses", "treated", "cohort", "population", "statistics",
}
_CLINICAL_TYPES = {"condition", "medication", "lab", "hospital_stay"}
_DEMOGRAPHIC_TYPES = {"age", "sex", "region", "data_holder"}
_AGE_BANDS = [
    {"lo": low, "hi": low + 9, "label": f"{low}-{low + 9}"}
    for low in range(0, 90, 10)
] + [{"lo": 90, "hi": None, "label": "90+"}]
_THRESHOLD_RE = re.compile(
    r"(?P<op>>=|≥|>|<=|≤|<|at\s+least|at\s+most|greater\s+than|"
    r"less\s+than|above|over|below|under)\s*"
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>%|mm\s*(?:\[?Hg\]?)|kg/m2|mL/min/\{?1\.73_m2\}?)?",
    re.I,
)
_PERIOD_PATTERNS = (
    (
        "range",
        re.compile(r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b", re.I),
    ),
    (
        "range",
        re.compile(r"\bfrom\s+(\d{4})\s+to\s+(\d{4})\b", re.I),
    ),
    ("since", re.compile(r"\bsince\s+(\d{4})\b", re.I)),
    ("year", re.compile(r"\b(?:in|during)\s+(\d{4})\b", re.I)),
)
_PERIOD_CUE_RE = re.compile(r"\b(?:in|during|between|from|since)\b", re.I)


@dataclass
class Interpretation:
    criteria: dict
    assumptions: list[str]
    unrecognised: list[str]

    def ok(self) -> bool:
        population = self.criteria.get("population", {})

        def has_supported_leaf(items):
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                if "items" in item:
                    if has_supported_leaf(item.get("items")):
                        return True
                elif item.get("type") in _CLINICAL_TYPES | _DEMOGRAPHIC_TYPES:
                    return True
            return False

        return has_supported_leaf(population.get("items", []))


def _label(term):
    return display_name(term)


def _phrase_pattern(value):
    phrase = str(value).strip()
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(phrase).replace(r"\ ", r"[\s-]+") + r"(?![A-Za-z0-9])",
        re.I,
    )


def _concept_matches(text):
    candidates = []
    seen = set()
    for alias, concept in entries():
        identity = (alias.casefold(), concept.kind, concept.term)
        if identity in seen:
            continue
        seen.add(identity)
        for match in _phrase_pattern(alias).finditer(text):
            candidates.append((match.start(), match.end(), alias, concept))
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2].casefold()))
    selected = []
    for candidate in candidates:
        start, end = candidate[:2]
        if any(start < other[1] and end > other[0] for other in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item[0])


def _category_matches(text, reserved_spans):
    data = load_data()
    candidates = []
    for kind in ("region", "data_holder"):
        for value in data["categories"].get(kind, []):
            if not value:
                continue
            for match in _phrase_pattern(value).finditer(text):
                candidates.append((match.start(), match.end(), kind, value))
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))
    selected = []
    for candidate in candidates:
        start, end = candidate[:2]
        if any(start < right and end > left for left, right in reserved_spans):
            continue
        if any(start < other[1] and end > other[0] for other in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item[0])


def _periods(text):
    periods = []
    for kind, pattern in _PERIOD_PATTERNS:
        for match in pattern.finditer(text):
            if kind == "range":
                start, end = int(match.group(1)), int(match.group(2))
                bounds = (f"{start}-01-01", f"{end}-12-31")
            elif kind == "since":
                year = int(match.group(1))
                bounds = (f"{year}-01-01", None)
            else:
                year = int(match.group(1))
                bounds = (f"{year}-01-01", f"{year}-12-31")
            periods.append((match.start(), match.end(), bounds))
    return sorted(periods, key=lambda item: item[0])


def _period_for(span, periods):
    start, end = span
    if not periods:
        return None
    candidate = min(
        periods,
        key=lambda item: 0 if item[0] <= end and item[1] >= start else min(
            abs(item[1] - start), abs(item[0] - end)
        ),
    )
    distance = 0 if candidate[0] <= end and candidate[1] >= start else min(
        abs(candidate[1] - start), abs(candidate[0] - end)
    )
    return candidate[2] if distance <= 100 else None


def _period_fields(start, end, prefix="period_"):
    result = {}
    if start:
        result[f"{prefix}start"] = start
    if end:
        result[f"{prefix}end"] = end
    return result


def _is_diagnosis_timing(text, start, end):
    before = text[max(0, start - 90):start].casefold()
    after = text[end:min(len(text), end + 90)].casefold()
    diagnosed_before = re.search(
        r"(?:diagnos(?:ed|is)|new\s+(?:cases?|diagnoses?)|incident)"
        r"[\w\s-]{0,55}$",
        before,
    )
    diagnosed_after = re.match(r"\s*(?:was\s+)?diagnos(?:ed|is)\b", after)
    return bool(diagnosed_before or diagnosed_after)


def _is_hospital_stay(text, start):
    before = text[max(0, start - 100):start]
    return bool(
        re.search(
            r"\b(?:hospitali[sz](?:ed|ation)s?|"
            r"admitted(?:\s+to\s+(?:the\s+)?hospital)?|admissions?)"
            r"\s+(?:for|with)\s*$",
            before,
            re.I,
        )
    )


def _is_excluded(text, start):
    before = text[max(0, start - 75):start]
    return bool(
        re.search(
            r"\b(?:without|no|never\s+(?:been\s+)?prescribed|"
            r"(?:(?:who\s+)?do\s+not\s+take)|"
            r"not\s+(?:on|taking|prescribed))"
            r"\s+(?:(?:a|an|any)\s+)?$",
            before,
            re.I,
        )
    )


def _medication_status(text, start, end, period):
    before = text[max(0, start - 70):start].casefold()
    after = text[end:min(len(text), end + 70)].casefold()
    context = before + " " + after
    if re.search(r"\b(?:never\s+(?:been\s+)?prescribed|"
                 r"ever\s+(?:been\s+)?prescribed|history\s+of|"
                 r"at\s+any\s+time|ever\s+used)\b", context):
        return "ever", None
    if re.search(r"\bactive\b", before[-35:]) or re.search(r"\bactive\b", after[:35]):
        return "active", None
    if period and _PERIOD_CUE_RE.search(context) and re.search(
        r"\b(?:prescrib\w*|prescription|taking|take|treated)\b", context
    ):
        return "during_period", period
    return "active", "default"


def _data_unit(loinc):
    observations = load_data()["observations"]
    rows = observations.loc[observations["loinc"].eq(str(loinc))]
    units = rows["unit"].dropna().astype(str)
    return next((unit for unit in units if unit), "")


def _threshold(text, start, end):
    suffix = text[end:min(len(text), end + 75)]
    match = _THRESHOLD_RE.search(suffix)
    if not match:
        return None
    operator = match.group("op").casefold()
    op = {
        ">": ">", "above": ">", "over": ">", "greater than": ">",
        ">=": ">=", "≥": ">=", "at least": ">=",
        "<": "<", "below": "<", "under": "<", "less than": "<",
        "<=": "<=", "≤": "<=", "at most": "<=",
    }[re.sub(r"\s+", " ", operator)]
    raw_unit = match.group("unit")
    unit = None
    if raw_unit:
        unit = re.sub(r"\s+", "", raw_unit)
        if unit.casefold().startswith("mm"):
            unit = "mmHg"
    return {
        "op": op,
        "value": float(match.group("value").replace(",", ".")),
        "unit": unit,
    }


def _is_lab_outcome(text, start, end):
    before = text[max(0, start - 60):start].casefold()
    after = text[end:min(len(text), end + 60)].casefold()
    if re.search(r"\b(?:mean|average|compare|distribution)\b", before):
        return True
    return bool(re.match(r"\s*(?:levels?|distribution)\b", after))


def _is_lab_presence(text, start, end):
    before = text[max(0, start - 60):start].casefold()
    after = text[end:min(len(text), end + 60)].casefold()
    return bool(
        re.search(r"\bwith\s+(?:a|an)?\s*(?:measurement\s+of\s+)?$", before)
        and re.search(r"\bmeasurement\b", after)
    ) or bool(re.search(r"\bwith\s+(?:a|an)\s+measurement\s+of\s*$", before))


def _age_criterion(text, concept_spans):
    minimums = []
    maximums = []
    spans = []
    for match in re.finditer(r"\baged?\s+(\d+)\s*[-–]\s*(\d+)\b", text, re.I):
        minimums.append(int(match.group(1)))
        maximums.append(int(match.group(2)))
        spans.append(match.span())
    for match in re.finditer(r"\bbetween\s+(\d+)\s+and\s+(\d+)\s+years?\b", text, re.I):
        minimums.append(int(match.group(1)))
        maximums.append(int(match.group(2)))
        spans.append(match.span())
    for match in re.finditer(r"\badults?\b", text, re.I):
        minimums.append(18)
        spans.append(match.span())
    for match in re.finditer(r"\bchildren\b", text, re.I):
        maximums.append(17)
        spans.append(match.span())
    for match in re.finditer(r"\baged?\s+(\d+)\s*(?:or older|and over|and older)\b", text, re.I):
        minimums.append(int(match.group(1)))
        spans.append(match.span())
    for match in re.finditer(r"\b(\d+)\s*\+(?=\s|[^\w]|$)", text):
        minimums.append(int(match.group(1)))
        spans.append(match.span())
    for match in re.finditer(r"\bover\s+(\d+)\b", text, re.I):
        if any(
            concept.kind == "observation"
            and 0 <= match.start() - end <= 24
            for _, end, _, concept in concept_spans
        ):
            continue
        minimums.append(int(match.group(1)) + 1)
        spans.append(match.span())
    for match in re.finditer(r"\bunder\s+(\d+)\b", text, re.I):
        if any(
            concept.kind == "observation"
            and 0 <= match.start() - end <= 24
            for _, end, _, concept in concept_spans
        ):
            continue
        maximums.append(int(match.group(1)) - 1)
        spans.append(match.span())
    if not minimums and not maximums:
        return None
    item = {"type": "age", "label": "age"}
    if minimums:
        item["min"] = max(minimums)
    if maximums:
        item["max"] = min(maximums)
    return item, min((span[0] for span in spans), default=0), max(
        (span[1] for span in spans), default=0
    )


def _gender_info(text):
    male = list(re.finditer(r"\b(?:men|males?)\b", text, re.I))
    female = list(re.finditer(r"\b(?:women|females?)\b", text, re.I))
    group_cue = bool(re.search(r"\bby\s+(?:sex|gender)\b", text, re.I))
    if group_cue or male and female:
        return [], [{"dimension": "sex"}], [
            *(match.span() for match in male),
            *(match.span() for match in female),
        ]
    if male:
        match = male[0]
        return [({"type": "sex", "label": "sex", "value": "male"}, match.start(), match.end())], [], [
            item.span() for item in male
        ]
    if female:
        match = female[0]
        return [({"type": "sex", "label": "sex", "value": "female"}, match.start(), match.end())], [], [
            item.span() for item in female
        ]
    return [], [], []


def _grouping(text, assumptions):
    groups = []
    if re.search(r"\bby\s+(?:age(?:\s+(?:band|group))?|age\s*band)\b", text, re.I):
        groups.append({"dimension": "age_band", "bands": _AGE_BANDS})
        assumptions.append("Age-band results use 10-year bands.")
    if re.search(r"\b(?:by\s+sex|by\s+gender)\b", text, re.I):
        groups.append({"dimension": "sex"})
    if re.search(r"\b(?:by\s+region|per\s+region|in\s+each\s+region|across\s+regions)\b", text, re.I):
        groups.append({"dimension": "region"})
    if re.search(r"\b(?:by\s+hospital|per\s+hospital|in\s+each\s+hospital|across\s+hospitals)\b", text, re.I):
        groups.append({"dimension": "data_holder"})
    unique = []
    for group in groups:
        if group["dimension"] not in {item["dimension"] for item in unique}:
            unique.append(group)
    return unique


def _or_joined(text, left, right):
    return bool(re.fullmatch(r"[\s,]*or[\s,]*", text[left:right], re.I))


def _make_population_nodes(text, nodes):
    nodes = [node for node in nodes if node["item"] is not None]
    nodes.sort(key=lambda item: item["start"])
    clinical = [node for node in nodes if node["clinical"]]
    grouped_spans = set()
    or_groups = []
    index = 0
    while index < len(clinical):
        run = [clinical[index]]
        next_index = index + 1
        while (
            next_index < len(clinical)
            and _or_joined(text, run[-1]["end"], clinical[next_index]["start"])
        ):
            run.append(clinical[next_index])
            next_index += 1
        if len(run) > 1:
            or_groups.append({
                "start": run[0]["start"],
                "end": run[-1]["end"],
                "item": {
                    "match": "any",
                    "items": [node["item"] for node in run],
                },
            })
            grouped_spans.update((node["start"], node["end"]) for node in run)
        index = next_index

    result = []
    emitted_groups = set()
    for node in nodes:
        if (node["start"], node["end"]) in grouped_spans:
            group = next(
                group for group in or_groups
                if group["start"] <= node["start"] <= group["end"]
            )
            identity = (group["start"], group["end"])
            if identity not in emitted_groups:
                result.append((group["start"], group["item"]))
                emitted_groups.add(identity)
            continue
        result.append((node["start"], node["item"]))

    result.sort(key=lambda item: item[0])
    if len(result) == 1 and isinstance(result[0][1], dict) and result[0][1].get("match") == "any":
        return {"match": "any", "items": result[0][1]["items"]}
    return {"match": "all", "items": [item for _, item in result]}


def _unrecognised_tokens(text, spans):
    values = []
    seen = set()
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9'-]*", text):
        token = match.group()
        normalized = token.casefold().strip("'-")
        if len(normalized) < 4 or normalized in _STOPWORDS or normalized.isdigit():
            continue
        if any(match.start() < end and match.end() > start for start, end in spans):
            continue
        if normalized not in seen:
            values.append(normalized)
            seen.add(normalized)
    return values


def interpret_question(text: str) -> Interpretation:
    text = str(text or "")
    periods = _periods(text)
    assumptions = []
    nodes = []
    recognised_spans = []
    concept_spans = _concept_matches(text)

    for start, end, alias, concept in concept_spans:
        label = _label(concept.term)
        period = _period_for((start, end), periods)
        excluded = _is_excluded(text, start)
        common = {"label": label}
        if excluded:
            common["exclude"] = True

        if concept.kind == "condition":
            if _is_hospital_stay(text, start):
                item = {
                    "type": "hospital_stay",
                    **common,
                    "icd10": list(concept.codes),
                    **_period_fields(*(period or (None, None))),
                }
            else:
                item = {"type": "condition", **common, "icd10": list(concept.codes)}
                if period:
                    period_start, period_end = period
                    if _is_diagnosis_timing(text, start, end):
                        item.update(_period_fields(period_start, period_end, "onset_"))
                    elif period_end:
                        item["onset_end"] = period_end
            nodes.append({
                "item": item, "start": start, "end": end,
                "clinical": True, "kind": item["type"],
            })
        elif concept.kind == "medication":
            status, status_assumption = _medication_status(
                text, start, end, period
            )
            item = {
                "type": "medication",
                **common,
                "atc": list(concept.codes),
                "status": status,
            }
            if status == "during_period" and period:
                item.update(_period_fields(*period))
            if status_assumption == "default":
                assumptions.append(
                    f"Medication status for '{label}' not stated; counted as active "
                    "on 31 December 2025 (documented default)."
                )
            nodes.append({
                "item": item, "start": start, "end": end,
                "clinical": True, "kind": "medication",
            })
        elif concept.kind == "observation":
            threshold = _threshold(text, start, end)
            if threshold:
                unit = threshold["unit"] or _data_unit(concept.codes[0])
                item = {
                    "type": "lab",
                    **common,
                    "loinc": concept.codes[0],
                    "op": threshold["op"],
                    "value": threshold["value"],
                }
                if unit:
                    item["unit"] = unit
                if period:
                    item.update(_period_fields(*period))
                assumptions.append(
                    "Lab thresholds use each patient's latest value within the period "
                    "(latest overall if no period)."
                )
                nodes.append({
                    "item": item, "start": start, "end": end,
                    "clinical": True, "kind": "lab",
                })
            elif _is_lab_outcome(text, start, end):
                unit = _data_unit(concept.codes[0])
                metric_label = f"mean latest {label}" + (f" ({unit})" if unit else "")
                metric = {
                    "type": "mean_latest_lab",
                    "label": metric_label,
                    "loinc": concept.codes[0],
                }
                if period:
                    metric.update(_period_fields(*period))
                assumption_period = "within the period" if period else "overall"
                assumptions.append(
                    f"Compared as the mean of each patient's latest {label} "
                    f"{assumption_period}."
                )
                nodes.append({
                    "item": None, "start": start, "end": end,
                    "clinical": False, "kind": "lab_outcome",
                })
                nodes[-1]["metric"] = metric
            elif _is_lab_presence(text, start, end):
                item = {
                    "type": "lab",
                    **common,
                    "loinc": concept.codes[0],
                    "op": "present",
                }
                if period:
                    item.update(_period_fields(*period))
                nodes.append({
                    "item": item, "start": start, "end": end,
                    "clinical": True, "kind": "lab",
                })

        recognised_spans.append((start, end))

    for start, end, kind, value in _category_matches(
        text, [(item[0], item[1]) for item in concept_spans]
    ):
        criterion_type = "region" if kind == "region" else "data_holder"
        item = {
            "type": criterion_type,
            "label": value,
            "values": [value],
        }
        if _is_excluded(text, start):
            item["exclude"] = True
        nodes.append({
            "item": item, "start": start, "end": end,
            "clinical": False, "kind": criterion_type,
        })
        recognised_spans.append((start, end))

    age = _age_criterion(text, concept_spans)
    if age:
        item, start, end = age
        nodes.append({
            "item": item, "start": start, "end": end,
            "clinical": False, "kind": "age",
        })
        recognised_spans.extend(
            match.span() for match in re.finditer(
                r"\b(?:adults?|children|aged?\s+\d+(?:\s*[-–]\s*\d+)?|"
                r"between\s+\d+\s+and\s+\d+\s+years?|over\s+\d+|under\s+\d+|"
                r"\d+\s*\+(?=\s|[^\w]|$))",
                text,
                re.I,
            )
        )

    gender_nodes, gender_groups, gender_spans = _gender_info(text)
    for item, start, end in gender_nodes:
        nodes.append({
            "item": item, "start": start, "end": end,
            "clinical": False, "kind": "sex",
        })
    recognised_spans.extend(gender_spans)
    group_by = _grouping(text, assumptions)
    if any(item["dimension"] == "age_band" for item in group_by):
        recognised_spans.extend(
            match.span() for match in re.finditer(
                r"\bby\s+age(?:\s+(?:band|group))?\b", text, re.I
            )
        )
    if any(item["dimension"] == "sex" for item in group_by) or gender_groups:
        if not any(item["dimension"] == "sex" for item in group_by):
            group_by.append({"dimension": "sex"})
        recognised_spans.extend(
            match.span() for match in re.finditer(
                r"\bby\s+(?:sex|gender)\b", text, re.I
            )
        )

    metrics = [{"type": "count_patients", "label": "patients"}]
    for node in nodes:
        if node.get("kind") == "lab_outcome":
            metrics.append(node["metric"])

    population = _make_population_nodes(text, nodes)
    criteria = {
        "population": population,
        "group_by": group_by,
        "metrics": metrics,
        "record_types": [],
    }
    for period_start, period_end, _ in periods:
        recognised_spans.append((period_start, period_end))
    unrecognised = _unrecognised_tokens(text, recognised_spans)
    return Interpretation(
        criteria=criteria,
        assumptions=list(dict.fromkeys(assumptions)),
        unrecognised=unrecognised,
    )
