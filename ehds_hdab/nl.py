from dataclasses import dataclass, field
import re

from .criteria import DEFAULT_YEARS

from .data import load_data
from .terminology import Concept, display_name, entries


_STOPWORDS = {
    "a", "about", "above", "across", "active", "admitted", "admission",
    "admissions", "adult", "adults", "after", "aged", "and", "any", "are",
    "as", "at", "average", "band", "bands", "below", "between", "by", "changes", "changed", "occurred", "occur",
    "evolution", "development", "pattern", "patterns", "longitudinal",
    "among", "cases", "children", "compare", "compared", "comparison",
    "condition", "conditions", "count", "distribution", "during", "each",
    "every", "ever", "exclude", "female", "females",
    "for", "from", "gender", "greater", "group", "grouping", "had", "has",
    "having", "have", "health", "how", "hospital", "hospitalised",
    "hospitalized", "hospitalisation", "hospitalisations", "hospitalization",
    "hospitalizations", "in", "include", "incident", "inpatient", "is",
    "individuals", "groups", "group", "latest", "austria", "excluding", "level", "levels", "less", "male", "males",
    "many", "mean",
    "measurement", "measurements", "men", "never", "new", "newly", "no", "not",
    "number", "of", "on", "or", "overall", "older", "over", "patients",
    "per", "period", "people", "prescribed", "prescription", "prevalence",
    "region", "regions", "regional", "results", "sex", "show", "since", "stay", "stays",
    "highest", "lowest", "austrian", "austria", "identify", "which", "access", "require", "required", "requires",
    "hospital", "hospitals", "most", "fewest", "estimate", "researchers", "university", "studying", "analysis",
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
    problems: list[str] = field(default_factory=list)
    clarifications: list[dict] = field(default_factory=list)

    def ok(self) -> bool:
        if self.problems:
            return False
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


_INCIDENCE = re.compile(
    r"\b(?:newly\s+(?:diagnosed|recorded|documented)|new\s+(?:cases?|diagnos\w*)|incident\s+cases?|incidence|"
    r"first\s+(?:recorded|diagnosed)|diagnosed\s+(?:during|within))\b", re.I)
_CUMULATIVE = re.compile(
    r"\b(?:by\s+the\s+end\s+of|by\s+31\s+december|as\s+of\s+(?:the\s+end|31\s+december)|cumulative|documented|"
    r"prevalen\w+|ever\s+diagnosed|living\s+with|with\s+a\s+(?:recorded\s+|documented\s+)?diagnosis)\b", re.I)
DIAGNOSIS_TIMING_TEXT = {
    "cumulative": "Diagnosis timing: cumulative \u2014 each year counts patients with a diagnosis recorded on or before "
                  "31 December of that year (documented, not newly diagnosed).",
    "new": "Diagnosis timing: newly recorded \u2014 each year counts patients whose first recorded diagnosis falls "
           "within that calendar year.",
}
DIAGNOSIS_TIMING_LABEL = {"cumulative": "patients with documented diagnosis", "new": "newly diagnosed patients"}


def _is_diagnosis_timing(text, start, end):
    """True when a period next to a condition means 'first diagnosed within the period'. Explicit cumulative
    wording wins; explicit incidence wording wins next; a bare 'diagnosed in 2025' keeps that meaning only in
    non-trend questions (for trends the timing is decided separately, asking if ambiguous)."""
    if _CUMULATIVE.search(text):
        return False
    before = text[max(0, start - 90):start].casefold()
    after = text[end:min(len(text), end + 90)].casefold()
    if _INCIDENCE.search(before[-70:]) or _INCIDENCE.search(after[:70]):
        return True
    if TREND_WORDS.search(text):
        return False
    diagnosed_before = re.search(r"(?:diagnos(?:ed|is)|incident)[\w\s-]{0,55}$", before)
    diagnosed_after = re.match(r"\s*(?:was\s+)?diagnos(?:ed|is)\b", after)
    return bool(diagnosed_before or diagnosed_after)


def _trend_years(criteria):
    year_group = next((g for g in criteria.get("group_by", []) if g.get("dimension") == "year"), None)
    if year_group is None:
        return None
    return int(year_group.get("from") or DEFAULT_YEARS[0]), int(year_group.get("to") or DEFAULT_YEARS[1])


def apply_diagnosis_timing(criteria, mode):
    """Set the per-year meaning of every (non-excluded) condition in a yearly trend: 'cumulative' = diagnosis
    recorded on or before 31 December of each year; 'new' = first recorded within each year. Returns a copy."""
    import copy

    criteria = copy.deepcopy(criteria)
    years = _trend_years(criteria)
    if years is None or mode not in DIAGNOSIS_TIMING_TEXT:
        return criteria
    start, end = f"{years[0]}-01-01", f"{years[1]}-12-31"
    for item in _leaves(criteria.get("population", {}).get("items", [])):
        if item.get("type") != "condition" or item.get("exclude"):
            continue
        if mode == "new":
            item["onset_start"], item["onset_end"] = start, end
        else:
            # Cumulative: no onset window; the yearly engine applies "recorded by 31 December of each year" itself.
            # (An explicit onset_end would also narrow the trend range to that single year.)
            item.pop("onset_start", None)
    for metric in criteria.get("metrics", []):
        if metric.get("type") == "count_patients" and metric.get("label") in ("patients", *DIAGNOSIS_TIMING_LABEL.values()):
            metric["label"] = DIAGNOSIS_TIMING_LABEL[mode]
    criteria["diagnosis_timing"] = mode
    return criteria


def _diagnosis_timing(text, criteria, assumptions):
    """Decide cumulative vs newly recorded for yearly trends over a condition. Returns True when ambiguous."""
    if _trend_years(criteria) is None:
        return False
    if not any(item.get("type") == "condition" and not item.get("exclude")
               for item in _leaves(criteria.get("population", {}).get("items", []))):
        return False
    incidence, cumulative = bool(_INCIDENCE.search(text)), bool(_CUMULATIVE.search(text))
    mode = "new" if incidence and not cumulative else "cumulative"
    # Ask only when the wording is genuinely about timing: "diagnosed" or a diagnosis word next to a period. A plain
    # "diabetes diagnoses over time" keeps the default (cumulative), stated on the card.
    has_period = bool(_PERIOD_CUE_RE.search(text) and re.search(r"\b(?:19|20)\d{2}\b", text))
    ambiguous = (not incidence and not cumulative
                 and bool(re.search(r"\bdiagnosed\b", text, re.I) or (has_period and re.search(r"\bdiagnos", text, re.I))))
    criteria.update(apply_diagnosis_timing(criteria, mode))
    assumptions.append(DIAGNOSIS_TIMING_TEXT[mode] + (" Please confirm." if ambiguous else ""))
    return ambiguous


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
            r"(?:(?:who\s+)?do\s+not\s+take)|excluding|exclude|except(?:\s+for)?|"
            r"other\s+than|outside(?:\s+of)?|not\s+(?:in|from)|"
            r"not\s+(?:on|taking|prescribed))"
            r"\s+(?:(?:a|an|any|the)\s+)?$",
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
    if re.search(r"\b(?:mean|average|compare|distribution|trends?\s+(?:in|of))\b", before):
        return True
    if re.search(r"\b(?:relationship|association|correlation|relation)\s+between\b", before):
        return True
    return bool(re.match(r"\s*(?:levels?|values?|distribution|trends?)\b", after))


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


TREND_WORDS = re.compile(
    r"\b(?:trends?|over\s+time|time\s+series|yearly|annual(?:ly)?|per\s+year|by\s+year|each\s+year|"
    r"year\s+by\s+year|monthly|per\s+month|by\s+month)\b", re.I)
YEARLY_ASSUMPTION = (
    "Results are shown per calendar year: diagnoses recorded by 31 December of each year, medication "
    "status as of 31 December of each year, the latest laboratory value within each year, hospital stays "
    "within each year; age as of 31 December 2025."
)


_PREVALENCE = re.compile(r"\bprevalen(?:ce|t)\b|\bproportion\s+of\b|\bpercentage\s+of\b|\bshare\s+of\b|"
                         r"\brates?\s+of\b|\b(?:highest|lowest)\s+rates?\b", re.I)
_ACCESS_SPE = re.compile(r"\b(?:pseudonymi[sz]ed(?:\s+\w+)?\s+(?:records?|data)|individual[- ]level(?:\s+\w+)?\s*(?:records?|data)?|"
                         r"patient[- ]level(?:\s+\w+)?\s*(?:records?|data)?|record[- ]level\s+data|secure\s+processing\s+environment|"
                         r"\bspe\b|microdata|raw\s+data)\b", re.I)
_ACCESS_STATS = re.compile(r"\b(?:anonymi[sz]ed\s+(?:statistics|counts|data)|aggregated?\s+(?:statistics|data|results|counts)|"
                           r"summary\s+statistics|statistics\s+only)\b", re.I)
_PURPOSE_CUE = re.compile(r"\b(?:we\s+(?:are|want|would\s+like|plan|intend|aim|need|require)|so\s+that|in\s+order\s+to|"
                          r"our\s+(?:aim|goal|objective|company|team)|purpose|study(?:ing)?|research\w*|to\s+(?:inform|plan|improve|evaluate)|"
                          r"project|company|university|ministry|insur\w+|hospital\s+management)\b", re.I)
_UNSUPPORTED_ANALYSIS = re.compile(r"\b(?:odds\s+ratios?|hazard\s+ratios?|relative\s+risks?|regression|survival\s+analys\w+|"
                                   r"kaplan|cox\s+model|predict\w*|risk\s+(?:factors?|scores?|model\w*)|causal\w*|"
                                   r"incidence\s+rate\s+ratios?|correlation\s+coefficients?|propensity)\b", re.I)
_INCIDENCE = re.compile(r"\bincidence\b|\bnew\s+cases\b|\bnewly\s+(?:diagnosed|recorded)\b", re.I)
_STAYS_COUNTED = re.compile(r"\b(?:hospitali[sz]ations?|admissions?|inpatient\s+stays?|hospital\s+stays?)\b", re.I)
_DISTRIBUTION = re.compile(r"\bdistribution\s+of\b|\bhistogram\s+of\b", re.I)
_RELATIONSHIP = re.compile(r"\b(?:relationship|association|correlation|relation)\s+between\b", re.I)


def _measure_metrics(text, nodes, metrics, assumptions, periods):
    """Turn measure words (prevalence, hospitalisations, distribution, relationship) into explicit,
    clinically labelled metrics. Returns (metrics, recognised_spans)."""
    spans = []
    leaves = [node["item"] for node in nodes if node.get("item")]
    conditions = [item for item in leaves if item.get("type") == "condition" and not item.get("exclude")]
    labs = [node for node in nodes if node.get("kind") == "lab_outcome"]
    period = periods[0][2] if periods else None

    match = _PREVALENCE.search(text)
    if match and conditions:
        spans.append(match.span())
        names = " and ".join(item["label"] for item in conditions)
        metrics = [m for m in metrics if m.get("type") != "count_patients"]
        metrics.insert(0, {"type": "count_patients", "label": f"patients with {names}"})
        metrics.insert(1, {"type": "percentage", "label": f"prevalence of {names} (%)"})
        assumptions.append(
            f"Prevalence = patients with a recorded diagnosis of {names} (by the end of the period or year) divided "
            "by all registered, non-opted-out patients with the same age/sex/region filters. The data have no "
            "registration dates, so the base population is the same in every year."
        )
    match = _INCIDENCE.search(text)
    if match and conditions:
        spans.append(match.span())
        assumptions.append(
            "Counted as newly recorded diagnoses (onset date within the period or year). The data record a diagnosis "
            "onset date, not the date of first clinical presentation; this is an incidence proxy, not true incidence."
        )
    match = _STAYS_COUNTED.search(text)
    if match and not _HOSPITALISED_FOR.search(text):
        stays = [item for item in leaves if item.get("type") == "hospital_stay"]
        codes = []
        label = "inpatient stays"
        if stays:
            codes, label = list(stays[0].get("icd10") or []), f"inpatient stays for {stays[0]['label']}"
        elif conditions:
            codes, label = list(conditions[0].get("icd10") or []), f"inpatient stays for {conditions[0]['label']}"
        if stays or conditions:
            spans.append(match.span())
            metric = {"type": "count_encounters", "label": label, "icd10": codes}
            if period:
                metric.update(_period_fields(*period))
            metrics.insert(0, metric)
            metrics = [m for m in metrics if m.get("type") != "count_patients"] + [
                {"type": "count_patients", "label": "patients hospitalised"}]
            if conditions and not stays:
                for node in nodes:
                    item = node.get("item")
                    if item is conditions[0]:
                        node["item"] = {"type": "hospital_stay", "label": item["label"], "icd10": list(item["icd10"]),
                                        **_period_fields(*(period or (None, None)))}
            assumptions.append(
                f"'{label.capitalize()}' counts inpatient stays whose main diagnosis matches; a patient with several "
                "stays is counted once per stay. 'Patients hospitalised' counts each patient once. Emergency and "
                "outpatient visits are not counted."
            )
    match = _DISTRIBUTION.search(text)
    if match and labs:
        spans.append(match.span())
        lab = labs[0]["metric"]
        unit = _data_unit(lab["loinc"])
        name = re.sub(r"^mean latest ", "", lab["label"]).split(" (")[0]
        metric = {"type": "distribution", "label": f"{name} category" + (f" ({unit})" if unit else ""),
                  "loinc": lab["loinc"]}
        if period:
            metric.update(_period_fields(*period))
        metrics = [m for m in metrics if m is not lab] + [metric]
        assumptions.pop() if assumptions and assumptions[-1].startswith("Compared as the mean") else None
        assumptions.append(
            f"Distribution shown as the number of patients per clinical {name} category, using each patient's latest "
            f"{name} value (within the period if given). Individual values are never shown."
        )
    match = _RELATIONSHIP.search(text)
    if match and len(labs) >= 2:
        spans.append(match.span())
        outcome, by = labs[1]["metric"], labs[0]["metric"]
        by_name = re.sub(r"^mean latest ", "", by["label"]).split(" (")[0]
        metrics = [m for m in metrics if m is not outcome and m is not by]
        stratified = dict(outcome)
        stratified["stratify"] = {"loinc": by["loinc"], "label": by_name}
        metrics.append(stratified)
        assumptions[:] = [a for a in assumptions if not a.startswith("Compared as the mean")]
        assumptions.append(
            f"Relationship shown as the mean of each patient's latest {re.sub(r'^mean latest ', '', outcome['label'])} "
            f"within clinical {by_name} categories. This is a descriptive association in aggregate form, not a "
            "patient-level scatter plot and not evidence of causation."
        )
    return metrics, spans


_HOSPITALISED_FOR = re.compile(r"\b(?:hospitali[sz]ed|admitted)\s+(?:for|with|because)\b", re.I)


_REGION_GROUP = re.compile(
    r"\b(?:by\s+region|per\s+region|in\s+each\s+region|across\s+regions|between\s+regions|"
    r"regions?\s+with\s+the\s+(?:highest|lowest|most|fewest|greatest|largest|smallest)|"
    r"(?:which|what|identify|compare|comparing|rank)\b[^.;?]{0,40}\bregions|"
    r"regional\s+(?:differences?|comparison|variation|distribution|rates?|breakdown|pattern))\b", re.I)
_HOSPITAL_GROUP = re.compile(
    r"\b(?:by\s+hospital|per\s+hospital|in\s+each\s+hospital|across\s+hospitals|between\s+hospitals|"
    r"hospitals?\s+with\s+the\s+(?:highest|lowest|most|fewest|greatest|largest|smallest)|"
    r"(?:which|what|identify|compare|comparing|rank)\b[^.;?]{0,40}\bhospitals)\b", re.I)


def _grouping(text, assumptions):
    groups = []
    if TREND_WORDS.search(text):
        groups.append({"dimension": "year"})
        assumptions.append(YEARLY_ASSUMPTION)
        if re.search(r"\b(?:monthly|per\s+month|by\s+month)\b", text, re.I):
            assumptions.append("Monthly results are not available in this prototype; shown per year instead.")
    if re.search(r"\b(?:by|between|across|per)\s+(?:age(?:\s+(?:bands?|groups?))?|age\s*bands?)\b", text, re.I):
        groups.append({"dimension": "age_band", "bands": _AGE_BANDS})
        assumptions.append("Age-band results use 10-year bands.")
    if re.search(r"\b(?:by\s+sex|by\s+gender)\b", text, re.I):
        groups.append({"dimension": "sex"})
    if _REGION_GROUP.search(text):
        groups.append({"dimension": "region"})
    if _HOSPITAL_GROUP.search(text):
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


_UNSUPPORTED_TOPICS = {
    "genetic": "genetic or genomic data", "genetics": "genetic or genomic data", "genome": "genetic or genomic data",
    "genomic": "genetic or genomic data", "variant": "genetic or genomic data", "variants": "genetic or genomic data",
    "dna": "genetic or genomic data", "mutation": "genetic or genomic data", "mutations": "genetic or genomic data",
    "imaging": "imaging", "mri": "imaging", "ct": "imaging", "x-ray": "imaging", "radiology": "imaging",
    "mortality": "deaths or survival", "death": "deaths or survival", "deaths": "deaths or survival",
    "died": "deaths or survival", "survival": "deaths or survival",
    "smoking": "smoking status or lifestyle data", "smokers": "smoking status or lifestyle data",
    "alcohol": "lifestyle data", "income": "socio-economic data", "ethnicity": "ethnicity",
    "cost": "costs", "costs": "costs", "readmission": "readmissions", "readmissions": "readmissions",
    "notes": "free-text clinical notes", "pathology": "pathology reports", "biopsy": "pathology reports",
}
_INJECTION = re.compile(
    r"\b(?:ignore (?:all |the )?(?:previous|prior|above) instructions|treat this (?:request )?as (?:automatically )?approved|"
    r"disable (?:the )?privacy|reveal all patient|bypass|override (?:the )?(?:authori[sz]ation|rules)|"
    r"system prompt|you are now)\b", re.I)
_GENERIC_DIABETES = re.compile(r"\bdiabet(?:es|ic|ics)\b", re.I)
_HOSPITAL_WORD = re.compile(r"\b(?:hospitali[sz]ed|hospitali[sz]ations?|admitted|admissions?)\b", re.I)


def _leaves(items):
    for item in items or []:
        if isinstance(item, dict) and "items" in item:
            yield from _leaves(item["items"])
        elif isinstance(item, dict):
            yield item


def _review_interpretation(text, criteria, unrecognised, assumptions):
    """Blocking problems (unsupported topics, contradictions) and clarification questions."""
    problems, clarifications = [], []
    leaves = list(_leaves(criteria.get("population", {}).get("items", [])))
    lowered = text.casefold()
    topics = sorted({_UNSUPPORTED_TOPICS[w] for w in re.findall(r"[a-z\-]+", lowered) if w in _UNSUPPORTED_TOPICS})
    for topic in topics:
        problems.append(f"The dataset contains no {topic}. It covers diagnoses (ICD-10), medications (ATC), "
                        "four laboratory tests (HbA1c, BMI, systolic blood pressure, eGFR), hospital encounters, "
                        "age, sex and region. Ask a question using those variables instead.")
    if _GENERIC_DIABETES.search(lowered) and not any(
            item.get("type") == "condition" and "E11" in item.get("icd10", []) for item in leaves):
        clarifications.append({
            "id": "diabetes_type",
            "question": "Which diabetes diagnoses do you mean? This dataset documents only type 2 diabetes (ICD-10 E11); "
                        "type 1 and other forms are not coded in it.",
            "options": [
                ("Type 2 diabetes (ICD-10 E11)", {"add": {"type": "condition", "label": "Type 2 diabetes", "icd10": ["E11"]}}),
                ("A different condition — let me rephrase", {"stop": True}),
            ],
        })
        unrecognised[:] = [w for w in unrecognised if not _GENERIC_DIABETES.search(w)]
    for item in leaves:
        if item.get("type") == "age" and item.get("min") is not None and item.get("max") is not None \
                and item["min"] > item["max"]:
            problems.append(f"The age range is contradictory (older than {item['min'] - 1} and younger than "
                            f"{item['max'] + 1} at the same time). Please correct it.")
        for start_key, end_key in (("period_start", "period_end"), ("onset_start", "onset_end")):
            if item.get(start_key) and item.get(end_key) and item[start_key] > item[end_key]:
                problems.append("The time period ends before it starts. Please correct it.")
    conditions = [item for item in leaves if item.get("type") == "condition" and not item.get("exclude")]
    if _HOSPITAL_WORD.search(lowered) and conditions and not any(
            item.get("type") == "hospital_stay" for item in leaves):
        condition = conditions[0]
        period = {}
        for key in ("onset_start", "onset_end"):
            pass
        clarifications.append({
            "id": "hospitalisation",
            "question": f"Hospitalised and {condition['label']}: which patients do you mean?",
            "options": [
                (f"Patients admitted because of {condition['label']} (inpatient stay with it as the main diagnosis)",
                 {"replace_condition_with_stay": condition["label"]}),
                (f"Patients with {condition['label']} who had any inpatient stay",
                 {"add": {"type": "hospital_stay", "label": "Any main diagnosis", "icd10": []}}),
            ],
            "note": "Counts are unique patients; the number of stays is not available in this flow.",
        })
    for match in _UNSUPPORTED_ANALYSIS.finditer(lowered):
        problems.append(
            f"This prototype computes counts, prevalence or shares (%) and mean laboratory values per group; it cannot "
            f"run \u201c{match.group(0)}\u201d. Ask for counts, percentages, means or yearly trends instead, or request "
            "pseudonymised records in a secure processing environment for your own analysis.")
        break
    relationship = _RELATIONSHIP.search(text)
    stratified = any(m.get("stratify") for m in criteria.get("metrics", []) or [])
    if relationship and not stratified:
        generic_pending = bool(_GENERIC_DIABETES.search(lowered)) and not any(
            "E11" in item.get("icd10", []) for item in conditions)
        if len(conditions) + int(generic_pending) >= 2:
            names = " and ".join([item["label"] for item in conditions] + (["Type 2 diabetes"] if generic_pending else []))
            clarifications.append({
                "id": "relationship_conditions",
                "question": f"A relationship between {names} can only be shown here in aggregate form. What would you like?",
                "options": [
                    (f"Count patients who have both ({names}); add 'by region', 'by sex' or 'over time' for a comparison", {}),
                    ("Something else \u2014 let me rephrase (or request pseudonymised records in a secure processing "
                     "environment for a patient-level analysis)", {"stop": True}),
                ],
                "note": "Co-occurrence counts are descriptive; they are not evidence of association or causation.",
            })
        else:
            problems.append("A relationship analysis needs two conditions or two laboratory measurements that I "
                            "recognise (e.g. 'relationship between BMI and HbA1c', 'relationship between type 2 "
                            "diabetes and heart failure').")
        unrecognised[:] = [w for w in unrecognised if w not in ("relationship", "association", "correlation", "relation")]
    if "rate" in unrecognised or "rates" in unrecognised:
        if any(item.get("type") == "hospital_stay" for item in leaves):
            assumptions.append("'Rates' are shown as counts of stays and of patients per group; population-based rates "
                               "per 1,000 inhabitants are not available in this prototype.")
            unrecognised[:] = [w for w in unrecognised if w not in ("rate", "rates")]
        elif not conditions:
            problems.append("'Rates' need a condition to count (e.g. 'rates of obesity by region' is shown as prevalence %).")
    if _INJECTION.search(text):
        assumptions.append("Instructions contained in the question (e.g. to ignore rules or approve automatically) "
                           "were ignored: the question is treated as data, never as a command.")
        unrecognised[:] = []
    return problems, clarifications


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
    if any(item["dimension"] == "year" for item in group_by):
        recognised_spans.extend(match.span() for match in TREND_WORDS.finditer(text))
        if periods:
            start, end = periods[0][2]
            year_group = next(item for item in group_by if item["dimension"] == "year")
            if start:
                year_group["from"] = int(str(start)[:4])
            if end:
                year_group["to"] = int(str(end)[:4])
    if any(item["dimension"] == "region" for item in group_by):
        recognised_spans.extend(m.span() for m in re.finditer(
            r"\b(?:by\s+region|per\s+region|in\s+each\s+region|across\s+regions)\b", text, re.I))
    if any(item["dimension"] == "data_holder" for item in group_by):
        recognised_spans.extend(m.span() for m in re.finditer(
            r"\b(?:by\s+hospital|per\s+hospital|in\s+each\s+hospital|across\s+hospitals)\b", text, re.I))
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
    metrics, extra_recognised = _measure_metrics(text, nodes, metrics, assumptions, periods)
    recognised_spans.extend(extra_recognised)

    if (not any(node.get("item") and node["item"].get("type") in _CLINICAL_TYPES for node in nodes)
            and any(node.get("kind") == "lab_outcome" for node in nodes)):
        for node in [n for n in nodes if n.get("kind") == "lab_outcome"]:
            metric = node["metric"]
            name = re.sub(r"^mean latest ", "", metric["label"]).split(" (")[0]
            item = {"type": "lab", "label": name, "loinc": metric["loinc"], "op": "present",
                    **_period_fields(metric.get("period_start"), metric.get("period_end"))}
            nodes.append({"item": item, "start": node["start"], "end": node["end"], "clinical": True, "kind": "lab"})
        names = " and ".join(
            re.sub(r"^mean latest ", "", n["metric"]["label"]).split(" (")[0] for n in nodes if n.get("kind") == "lab_outcome")
        assumptions.append(f"Population: all registered, non-opted-out patients with at least one {names} measurement"
                           + (" in the period." if periods else "."))
    population = _make_population_nodes(text, nodes)
    criteria = {
        "population": population,
        "group_by": group_by,
        "metrics": metrics,
        "record_types": [],
    }
    spe_match = _ACCESS_SPE.search(text)
    if spe_match:
        criteria["access_hint"] = "health data access application"
        recognised_spans.extend(m.span() for m in _ACCESS_SPE.finditer(text))
        assumptions.append("Your question asks for pseudonymised, patient-level records: the request is prepared as a "
                           "health data access application (secure processing environment). You can change this on "
                           "the review screen; the purpose must explain why statistics are not enough.")
    elif _ACCESS_STATS.search(text):
        criteria["access_hint"] = "health data request"
        recognised_spans.extend(m.span() for m in _ACCESS_STATS.finditer(text))
    purpose_sentences = [part.strip() for part in re.split(r"(?<=[.!?;])\s+", text)
                         if part.strip() and _PURPOSE_CUE.search(part)]
    if purpose_sentences:
        criteria["purpose_hint"] = " ".join(purpose_sentences)
    for period_start, period_end, _ in periods:
        recognised_spans.append((period_start, period_end))
    timing_ambiguous = _diagnosis_timing(text, criteria, assumptions)
    unrecognised = _unrecognised_tokens(text, recognised_spans)
    problems, clarifications = _review_interpretation(text, criteria, unrecognised, assumptions)
    if timing_ambiguous:
        clarifications.append({
            "id": "diagnosis_timing",
            "question": "Per year, which patients do you mean?",
            "options": [
                ("Patients with a documented diagnosis by 31 December of each year (cumulative)", {"diagnosis_timing": "cumulative"}),
                ("Patients first diagnosed within each year (newly recorded)", {"diagnosis_timing": "new"}),
            ],
            "note": "Cumulative counts grow over time; newly recorded counts show how many were added each year.",
        })
    return Interpretation(problems=problems, clarifications=clarifications,
        
        criteria=criteria,
        assumptions=list(dict.fromkeys(assumptions)),
        unrecognised=unrecognised,
    )
