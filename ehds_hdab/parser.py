import re
from datetime import datetime

from .config import REFERENCE_DATE
from .plan import QueryPlan
from .terminology import entries, lookup, normalize

_OBS_TERMS = sorted(
    {alias for alias, concept in entries() if concept.kind == "observation"},
    key=len,
    reverse=True,
)
_BASE_STARTS = (
    "the", "their", "that", "mean", "average", "median", "%", "percent",
    "share", "proportion", "how many", "number", "count", "adults",
    "patients", "women", "men", "children", "people", "stays",
    "hospitalisation", "hospitalisations", "hospitalization", "hospitalizations",
    "admission", "admissions",
)
_DATE_TOKEN = (
    r"(?:\d{4}-\d{2}-\d{2}|\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|"
    r"[A-Za-z]+\s+\d{1,2},?\s+\d{4})"
)
_HOSPITALISATION_PHRASE = re.compile(
    r"\b(?P<cue>hospitali[sz](?:ed|ation)s?|"
    r"admitted(?:\s+to\s+(?:the\s+)?hospital)?|admissions?)"
    r"(?:\s+in\s+\d{4})?\s+(?:for|with)\s+",
    re.I,
)
_ENCOUNTER_COUNT_CUE = re.compile(
    r"\b(?:stays|encounters|visits|hospitali[sz]ations?|admissions?)\b",
    re.I,
)
_CARE_SETTING_CUE = re.compile(
    r"\b(?:hospitali[sz]\w*|admitted|admissions?|emergency|inpatient|"
    r"outpatient|visits?|stays?)\b",
    re.I,
)
_TIME_RESTRICTION_CUE = re.compile(
    rf"\bin\s+\d{{4}}\b|\bsince\b|\bduring\b|"
    rf"\bbetween\s+{_DATE_TOKEN}\s+and\s+{_DATE_TOKEN}\b|"
    rf"\bfrom\s+{_DATE_TOKEN}\s+to\s+{_DATE_TOKEN}\b",
    re.I,
)


def _slug(text):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", normalize(text))).strip("_")


def _date_bound(value, end=False):
    value = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", value.strip(), flags=re.I)
    if re.fullmatch(r"\d{4}", value):
        return f"{value}-{'12-31' if end else '01-01'}"
    for date_format in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value, date_format).date().isoformat()
        except ValueError:
            continue
    return None


def _encounter_date_filter(text):
    date_range = re.search(
        rf"\bbetween\s+({_DATE_TOKEN})\s+and\s+({_DATE_TOKEN})\b|"
        rf"\bfrom\s+({_DATE_TOKEN})\s+to\s+({_DATE_TOKEN})\b",
        text,
        re.I,
    )
    if date_range:
        start, end = (
            date_range.group(1, 2)
            if date_range.group(1)
            else date_range.group(3, 4)
        )
        start_bound = _date_bound(start)
        end_bound = _date_bound(end, end=True)
        if start_bound and end_bound:
            return {"start_start": start_bound, "start_end": end_bound}
    since = re.search(rf"\bsince\s+({_DATE_TOKEN})\b", text, re.I)
    if since:
        start_bound = _date_bound(since.group(1))
        if start_bound:
            return {"start_start": start_bound}
    year = re.search(r"\b(?:in|during)\s+(\d{4})\b", text, re.I)
    if year:
        return {"start_year": int(year.group(1))}
    return {}


def _cue_context(text, match):
    context = re.split(r"[,;.!?]", text[match.start():], maxsplit=1)[0]
    return re.sub(r"\s+", " ", context).strip()


def _merge_encounter_population(population, additional, encounter_key, plan):
    merged = list(population)
    encounter_codes = plan.concepts.get(encounter_key, {}).get(
        "primary_icd10_prefixes", []
    )
    for item in additional:
        spec = plan.concepts.get(item.get("concept"), {})
        if spec.get("kind") == "condition" and not item.get("negate"):
            condition_codes = spec.get("icd10_prefixes", [])
            if any(
                left.startswith(right) or right.startswith(left)
                for left in condition_codes
                for right in encounter_codes
            ):
                continue
        if item not in merged:
            merged.append(item)
    return merged


def _split_definitions(definitions):
    out = []
    for sentence in definitions or []:
        out.extend(part.strip() for part in re.split(r"[.;]+", sentence) if part.strip())
    return out


def _code_tokens(text):
    tokens = re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)?(?:\.[A-Z0-9]+)?\b", text.upper())
    return list(dict.fromkeys(token for token in tokens if token not in {"OR", "AND", "WITH"}))


def _parse_bands(text):
    values = re.findall(r"(?<!\d)(\d+)(?:\s*-\s*(\d+)|\s*(\+))", text)
    bands = []
    for low, high, plus in values:
        lo = int(low)
        bands.append({"lo": lo, "hi": None if plus else int(high), "label": f"{lo}+" if plus else f"{lo}-{high}"})
    if re.search(r"\b10[- ]year age bands?\b", text, re.I) and not bands:
        bands = [
            {"lo": lo, "hi": lo + 9 if lo < 90 else None, "label": f"{lo}-{lo + 9}" if lo < 90 else "90+"}
            for lo in range(0, 100, 10)
        ]
    return bands


def _name_matches(text, candidates):
    lowered = text.casefold()
    found = []
    occupied = []
    for phrase, key in sorted(candidates, key=lambda pair: len(pair[0]), reverse=True):
        pattern = r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)"
        for match in re.finditer(pattern, lowered):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            occupied.append((match.start(), match.end()))
            found.append((match.start(), key, match.start()))
    return sorted(found)


def _new_spec(plan, name, kind, codes=()):
    if name in plan.concepts:
        return name
    code_field = {"condition": "icd10_prefixes", "medication": "atc_prefixes", "observation": "loinc"}[kind]
    spec = {"kind": kind}
    if kind in ("condition", "medication"):
        spec[code_field] = list(codes)
    else:
        spec[code_field] = codes[0] if codes else ""
        spec["pick"] = "latest"
    plan.concepts[name] = spec
    return name


def _population(text, plan, aliases):
    lowered = text.casefold()
    population = []
    occupied = []
    matches = []
    for phrase, key in aliases:
        for match in re.finditer(r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)", lowered):
            matches.append((match.start(), match.end(), phrase, key))
    for start, end, phrase, key in sorted(matches, key=lambda item: (-(item[1] - item[0]), item[0])):
        if any(start < b and end > a for a, b in occupied):
            continue
        occupied.append((start, end))
        prefix = lowered[max(0, start - 32):start]
        negate = bool(re.search(r"\b(?:no|without|not)(?:\s+\w+){0,2}\s+$", prefix))
        item = {"concept": key, "negate": negate}
        if item not in population:
            population.append(item)

    special = []
    for word, spec in (
        ("adults", "__adults__"),
        ("adult", "__adults__"),
        ("women", "__female__"),
        ("woman", "__female__"),
        ("men", "__male__"),
        ("man", "__male__"),
        ("children", "__children__"),
        ("child", "__children__"),
    ):
        if re.search(rf"\b{word}\b", lowered):
            special.append(spec)
    for key in special:
        if not any(x["concept"] == key for x in population):
            population.insert(0, {"concept": key, "negate": False})

    for phrase, match in (
        (r"\b(?:aged?|age)\s+(\d+)\s+(?:or older|and over|and older|or above)\b", "min"),
        (r"\b(?:aged?|age)\s+(\d+)\s*[-–]\s*(\d+)\b", "range"),
        (r"\b(?:under|below|less than)\s+(\d+)\b", "under"),
    ):
        found = re.search(phrase, lowered)
        if found:
            if match == "min":
                spec = {"kind": "age", "min": int(found.group(1)), "max": None}
            elif match == "range":
                spec = {"kind": "age", "min": int(found.group(1)), "max": int(found.group(2))}
            else:
                spec = {"kind": "age", "min": None, "max": int(found.group(1)) - 1}
            key = f"__age_{spec.get('min')}_{spec.get('max')}"
            plan.concepts[key] = spec
            if not any(item["concept"] == key for item in population):
                population.append({"concept": key, "negate": False})

    observation_keys = [
        key for phrase, key in aliases
        if plan.concepts.get(key, {}).get("kind") == "observation"
        and (
            re.search(r"(?i)\b(?:with|have|has|having)\s+(?:a\s+|an\s+)?(?:latest\s+)?"
                      + re.escape(phrase) + r"(?:\s+(?:value|measurement))?", text)
            or re.search(r"(?i)\bat least one\s+" + re.escape(phrase) + r"\s+value", text)
        )
    ]
    for key in observation_keys:
        if not any(x["concept"] == key for x in population):
            population.append({"concept": key, "negate": False})
    return population


def _metric_parts(text):
    text = re.sub(r"\s*,?\s*(?:with\s+)?one\s+decimal\s*\.?\s*$", "", text, flags=re.I)
    starts = "|".join(re.escape(item) for item in _BASE_STARTS if item != "%")
    parts = re.split(
        r",\s+|\s+and\s+(?=%|(?:" + starts + r")\b)",
        text,
        flags=re.I,
    )
    cleaned = []
    for part in parts:
        part = re.sub(r"\s*,?\s*(?:with\s+)?one\s+decimal\s*\.?\s*$", "", part, flags=re.I)
        part = part.strip(" .")
        if part:
            cleaned.append(part)
    return cleaned


def _ratio_definition(sentence):
    match = re.match(r"^\s*([^=]+?)\s*=\s*(.+?)\s*$", sentence)
    if not match:
        return None
    ratio = re.fullmatch(
        r"(?P<numerator>.+?)\s*/\s*(?P<denominator>.+?)\s*"
        r"(?:[×x*]\s*100|times\s+100|per\s+100)"
        r"(?:\s*,?\s*(?:with\s+)?one\s+decimal)?\s*\.?",
        match.group(2),
        re.I,
    )
    if not ratio:
        return None
    return {
        "name": match.group(1).strip(),
        "numerator": ratio.group("numerator").strip(),
        "denominator": ratio.group("denominator").strip(),
    }


def _formula_metric_for_phrase(phrase, formulas):
    normalized = normalize(phrase)
    normalized = re.sub(r"^(?:the|a|an)\s+", "", normalized)
    for name, metric in formulas.items():
        if normalized == name:
            return metric
    return None


def _sentence_scope(sentences):
    return " ".join(sentences)


def parse_request(request):
    plan = QueryPlan()
    definitions = _split_definitions(request.get("definitions"))
    question = str(request.get("question", "") or "")
    all_text = " ".join([question, *definitions])
    definition_names = {}
    age_bands = []
    parsed_ratios = [
        (sentence, _ratio_definition(sentence))
        for sentence in definitions
    ]
    ratio_definitions = [parsed for _, parsed in parsed_ratios if parsed is not None]
    ratio_sentences = {sentence for sentence, parsed in parsed_ratios if parsed is not None}
    global_active = bool(re.search(r"\bmedication is active\b|\bactive (?:medication|prescription|.*?\bATC)\b", all_text, re.I))

    for sentence in definitions:
        if "age band" in sentence.casefold():
            age_bands = _parse_bands(sentence)
        if sentence in ratio_sentences:
            continue
        match = re.match(r"^\s*([^=]+?)\s*=\s*(.+)$", sentence)
        if not match:
            continue
        name, definition = match.group(1).strip(), match.group(2).strip()
        alias = normalize(name)
        lowered = definition.casefold()
        if re.search(r"\b(?:aged|age|under|below|less than)\s+\d+", lowered):
            age_match = re.search(r"\b(?:aged|age|under|below|less than)\s+(\d+)(?:\s*[-–]\s*(\d+))?", lowered)
            spec = {"kind": "age"}
            if age_match:
                low = int(age_match.group(1))
                high = int(age_match.group(2)) if age_match.group(2) else None
                if re.search(r"\b(?:under|below|less than)\s+\d+", lowered):
                    spec.update({"min": None, "max": int(re.search(r"\d+", lowered).group()) - 1})
                elif high is not None:
                    spec.update({"min": low, "max": high})
                elif re.search(r"\b(?:or older|and over|and older|over|or above)\b", lowered):
                    spec.update({"min": low, "max": None})
                else:
                    spec.update({"min": low, "max": None})
            plan.concepts[name] = spec
            definition_names[name] = name
            continue
        sex_match = re.search(r"\bsex\s+['\"]?(female|male)['\"]?", lowered)
        if sex_match:
            plan.concepts[name] = {"kind": "sex", "value": sex_match.group(1)}
            definition_names[name] = name
            continue
        icd_match = re.search(r"icd-?10(?: code)?s?.*?starting with\s+(.+)", definition, re.I)
        atc_match = re.search(r"atc code starting with\s+(.+)", definition, re.I)
        loinc_match = re.search(r"loinc\s+([a-z0-9-]+)", definition, re.I)
        encounter_match = re.search(r"encounter with type\s+['\"]?([a-z -]+)", definition, re.I)
        if encounter_match:
            primary = re.search(r"(?:primary_icd10|main diagnosis).*?starting with\s+(.+)", definition, re.I)
            plan.concepts[name] = {
                "kind": "encounter",
                "types": [encounter_match.group(1).strip().lower()],
                "primary_icd10_prefixes": _code_tokens(primary.group(1)) if primary else [],
            }
        elif icd_match:
            codes = _code_tokens(icd_match.group(1))
            plan.concepts[name] = {"kind": "condition", "icd10_prefixes": codes}
        elif atc_match:
            codes = _code_tokens(atc_match.group(1))
            plan.concepts[name] = {"kind": "medication", "atc_prefixes": codes, "active_only": global_active or "active" in lowered}
        elif loinc_match:
            observation_name = name
            if name.casefold() == "cohort":
                for term, concept in entries():
                    if concept.kind == "observation" and loinc_match.group(1) in concept.codes:
                        observation_name = {
                            "hba1c": "HbA1c",
                            "bmi": "BMI",
                            "systolic blood pressure": "systolic blood pressure",
                            "egfr": "eGFR",
                        }.get(concept.term, concept.term)
                        break
            plan.concepts[observation_name] = {
                "kind": "observation",
                "loinc": loinc_match.group(1),
                "pick": "latest",
            }
        elif re.search(r"\bregion\b", lowered):
            values = re.findall(r"['\"]([^'\"]+)['\"]", definition)
            plan.concepts[name] = {"kind": "region", "values": values}
        elif re.search(r"\bdata[_ -]?holder\b|\bhospital\b", lowered):
            values = re.findall(r"['\"]([^'\"]+)['\"]", definition)
            plan.concepts[name] = {"kind": "data_holder", "values": values}
        else:
            concept = lookup(name)
            if concept:
                kind = concept.kind
                if kind == "condition":
                    plan.concepts[name] = {"kind": kind, "icd10_prefixes": list(concept.codes)}
                elif kind == "medication":
                    plan.concepts[name] = {"kind": kind, "atc_prefixes": list(concept.codes), "active_only": global_active or "active" in lowered}
                else:
                    plan.concepts[name] = {"kind": kind, "loinc": concept.codes[0] if concept.codes else "", "pick": "latest"}
            else:
                plan.warnings.append(f"Could not interpret definition: {sentence}")
                continue
        if not (name.casefold() == "cohort" and loinc_match):
            definition_names[name] = name

    if global_active:
        for spec in plan.concepts.values():
            if spec.get("kind") == "medication":
                spec["active_only"] = True

    aliases = []
    for name, key in definition_names.items():
        aliases.append((name, key))
        aliases.append((_slug(name).replace("_", " "), key))
    for alias, concept in entries():
        existing = next((key for key, spec in plan.concepts.items() if spec.get("kind") == concept.kind and (
            concept.codes[0] in spec.get("icd10_prefixes", [])
            or concept.codes[0] in spec.get("atc_prefixes", [])
            or concept.codes[0] == spec.get("loinc")
        )), None)
        if existing:
            aliases.append((alias, existing))
        elif re.search(r"(?i)(?<!\w)" + re.escape(alias) + r"(?!\w)", all_text):
            key = _new_spec(plan, concept.term, concept.kind, concept.codes)
            if concept.kind == "medication":
                plan.concepts[key]["active_only"] = global_active or bool(re.search(r"(?i)\bactive\b", all_text))
            aliases.append((alias, key))
    aliases = list(dict.fromkeys((normalize(phrase), key) for phrase, key in aliases if phrase))

    pick_patterns = {
        "highest": "max",
        "maximum": "max",
        "max": "max",
        "lowest": "min",
        "minimum": "min",
        "min": "min",
        "first": "earliest",
        "earliest": "earliest",
        "last": "latest",
        "latest": "latest",
        "most recent": "latest",
    }
    for name, spec in list(plan.concepts.items()):
        if spec.get("kind") == "observation":
            terms = [phrase for phrase, key in aliases if key == name]
            for term in sorted(terms, key=len, reverse=True):
                concept_pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
                pick_names = "|".join(
                    re.escape(value)
                    for value in sorted(pick_patterns, key=len, reverse=True)
                )
                pick = re.search(
                    rf"\b({pick_names})\b[^.;]*{concept_pattern}",
                    all_text,
                    re.I,
                )
                if not pick:
                    pick = re.search(
                        rf"{concept_pattern}[^.;]*\b({pick_names})\b",
                        all_text,
                        re.I,
                    )
                if pick:
                    spec["pick"] = pick_patterns[pick.group(1).casefold()]
                    break

    for key, kind in (
        ("__adults__", "age"),
        ("__female__", "sex"),
        ("__male__", "sex"),
        ("__children__", "age"),
    ):
        if re.search(r"\b(adult|adults|woman|women|man|men|child|children)\b", all_text, re.I):
            if kind == "age":
                plan.concepts[key] = {"kind": "age", **({"min": 18, "max": None} if key == "__adults__" else {"min": None, "max": 17})}
            else:
                plan.concepts[key] = {"kind": "sex", "value": "female" if key == "__female__" else "male"}
    aliases.extend([
        ("adults", "__adults__"), ("adult", "__adults__"),
        ("women", "__female__"), ("woman", "__female__"),
        ("men", "__male__"), ("man", "__male__"),
        ("children", "__children__"), ("child", "__children__"),
    ])
    aliases = list(dict.fromkeys((normalize(p), k) for p, k in aliases))

    report_sources = [
        sentence.split(":", 1)[1].strip()
        for sentence in definitions
        if sentence.casefold().startswith("report per ") and ":" in sentence
    ]
    encounter_mentions = []
    condition_aliases = [
        (phrase, key)
        for phrase, key in aliases
        if plan.concepts.get(key, {}).get("kind") == "condition"
    ]
    encounter_sources = [("question", question)] + [
        (f"report_{index}", text)
        for index, text in enumerate(report_sources)
    ]
    for source_name, source in encounter_sources:
        for cue in _HOSPITALISATION_PHRASE.finditer(source):
            tail_start = cue.end()
            tail = source[tail_start:]
            tail_stop = re.search(
                r"[,;.!?]|\b(?:in\s+\d{4}|since|during|between|from|by|per)\b",
                tail,
                re.I,
            )
            tail_end = tail_start + (tail_stop.start() if tail_stop else len(tail))
            condition_matches = []
            for phrase, key in condition_aliases:
                pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
                for match in re.finditer(pattern, source, re.I):
                    if tail_start <= match.start() and match.end() <= tail_end:
                        condition_matches.append(
                            (match.start(), -(match.end() - match.start()), match.end(), key)
                        )
            if not condition_matches:
                continue
            _, _, condition_end, condition_key = min(condition_matches)
            prefixes = list(plan.concepts[condition_key].get("icd10_prefixes", []))
            if not prefixes:
                continue
            encounter_spec = {
                "kind": "encounter",
                "types": ["inpatient"],
                "primary_icd10_prefixes": prefixes,
            }
            encounter_spec.update(_encounter_date_filter(source))
            encounter_key = next(
                (
                    key
                    for key, spec in plan.concepts.items()
                    if spec.get("kind") == "encounter"
                    and "inpatient" in [value.casefold() for value in spec.get("types", [])]
                    and spec.get("primary_icd10_prefixes", []) == prefixes
                ),
                None,
            )
            if encounter_key is None:
                encounter_key = "__inpatient_" + _slug(condition_key)
                suffix = 2
                while encounter_key in plan.concepts:
                    encounter_key = f"__inpatient_{_slug(condition_key)}_{suffix}"
                    suffix += 1
                plan.concepts[encounter_key] = encounter_spec
            else:
                plan.concepts[encounter_key].update(
                    {key: value for key, value in encounter_spec.items() if key not in {"kind", "types", "primary_icd10_prefixes"}}
                )
            mention_phrase = source[cue.start():condition_end].strip()
            aliases.append((mention_phrase, encounter_key))
            encounter_mentions.append({
                "concept": encounter_key,
                "label_phrase": mention_phrase,
                "event": bool(_ENCOUNTER_COUNT_CUE.search(mention_phrase)),
                "source": source_name,
            })
    aliases = list(dict.fromkeys((normalize(phrase), key) for phrase, key in aliases if phrase))
    question_population = _population(question, plan, aliases)

    for sentence in definitions:
        lower = sentence.casefold()
        if "cohort =" in lower:
            cohort_text = sentence.split("=", 1)[1]
            plan.cohort = _population(cohort_text, plan, aliases)
        if re.search(r"report for each patient|patient-level listing|patient id", lower):
            plan.patient_level_listing = True

    # Encounter definitions can be constrained by separate date and primary-diagnosis clauses.
    for sentence in definitions:
        lower = sentence.casefold()
        if re.search(r"start_date.*\bin\s+\d{4}", lower):
            year = int(re.search(r"\bin\s+(\d{4})", lower).group(1))
            for spec in plan.concepts.values():
                if spec.get("kind") == "encounter":
                    spec["start_year"] = year
        primary = re.search(r"(?:primary_icd10|main diagnosis).*?starting with\s+(.+)", sentence, re.I)
        if primary:
            codes = _code_tokens(primary.group(1))
            for spec in plan.concepts.values():
                if spec.get("kind") == "encounter":
                    spec["primary_icd10_prefixes"] = codes
        onset_year = re.search(r"\bonset(?:_date)?\b.*?\bin\s+(\d{4})", sentence, re.I)
        onset_range = re.search(r"\bonset(?:_date)?\b.*?\bfrom\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", sentence, re.I)
        for spec in plan.concepts.values():
            if spec.get("kind") == "condition":
                if onset_year:
                    spec["onset_year"] = int(onset_year.group(1))
                if onset_range:
                    spec["onset_start"], spec["onset_end"] = onset_range.groups()
        start_range = re.search(r"\bstart_date\b.*?\bfrom\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", sentence, re.I)
        for spec in plan.concepts.values():
            if spec.get("kind") == "encounter" and start_range:
                spec["start_start"], spec["start_end"] = start_range.groups()

    for sentence in definitions:
        if "report per " not in sentence.casefold() and not re.search(r"\bby (?:the patient's )?(?:region|hospital|sex|age band)\b", sentence, re.I):
            continue
        scope = sentence.split(":", 1)[0]
        low = scope.casefold()
        dimensions = []
        if re.search(r"\bregion\b", low):
            dimensions.append({"dimension": "region"})
        if re.search(r"\b(?:band|age band)\b", low):
            dimensions.append({"dimension": "age_band", "bands": age_bands})
        if re.search(r"\b(?:hospital|data holder|data_holder|klinikum)\b", low):
            dimensions.append({"dimension": "data_holder"})
        if re.search(r"\bsex\b", low):
            dimensions.append({"dimension": "sex"})
        for dim in dimensions:
            if not any(x["dimension"] == dim["dimension"] for x in plan.group_by):
                plan.group_by.append(dim)
    if not plan.group_by:
        question_low = question.casefold()
        for token, dimension in (("region", "region"), ("hospital", "data_holder"), ("data_holder", "data_holder"), ("sex", "sex"), ("age band", "age_band")):
            if token in question_low and not any(x["dimension"] == dimension for x in plan.group_by):
                plan.group_by.append({"dimension": dimension, **({"bands": age_bands} if dimension == "age_band" else {})})
    if any(re.search(r"(?i)\b10[- ]year age bands?\b", text) for text in [question, *definitions]) and not age_bands:
        age_bands = _parse_bands("10-year age band")
        for dim in plan.group_by:
            if dim["dimension"] == "age_band":
                dim["bands"] = age_bands
    for dim in plan.group_by:
        if dim["dimension"] == "age_band" and not dim.get("bands"):
            dim["bands"] = age_bands or _parse_bands("10-year age band")

    report_text = " ".join(s.split(":", 1)[1] for s in definitions if s.casefold().startswith("report per ") and ":" in s)
    metric_sentences = [
        s for s in definitions
        if re.search(r"\bcount (?:stays?|encounters?|visits?|hospitali[sz]ations?|admissions?)\b", s, re.I)
    ]
    formula_metrics = {
        normalize(item["name"]): {
            "id": _slug(item["name"]),
            "label": _slug(item["name"]),
            "type": "percentage",
            "numerator_population": _population(item["numerator"], plan, aliases),
            "denominator_population": _population(item["denominator"], plan, aliases),
        }
        for item in ratio_definitions
    }
    added_formula_metrics = set()
    previous_population = []
    main_population = []
    if plan.cohort:
        previous_population = plan.cohort
    if report_text:
        phrases = _metric_parts(report_text)
    else:
        phrases = []
    for sentence in metric_sentences:
        encounter_key = next((k for k, v in plan.concepts.items() if v.get("kind") == "encounter"), None)
        if encounter_key:
            plan.concepts[encounter_key].update(_encounter_date_filter(sentence))

    for phrase in phrases:
        low = phrase.casefold().strip()
        if not low:
            continue
        label = _slug(phrase)
        formula_metric = _formula_metric_for_phrase(phrase, formula_metrics)
        if formula_metric:
            plan.metrics.append(formula_metric)
            added_formula_metrics.add(formula_metric["id"])
            continue
        phrase_population = _population(phrase, plan, aliases)
        encounter_key = next(
            (
                item["concept"]
                for item in phrase_population
                if plan.concepts.get(item["concept"], {}).get("kind") == "encounter"
            ),
            None,
        )
        if encounter_key and _ENCOUNTER_COUNT_CUE.search(low):
            population = _merge_encounter_population(
                phrase_population, question_population, encounter_key, plan
            )
            plan.metrics.append({
                "id": label or "encounter_count",
                "label": label or "encounter_count",
                "type": "count_encounters",
                "encounter": encounter_key,
                "population": population,
            })
            continue
        if re.search(r"\b(?:mean|average)\b", low):
            observation = next((k for p, k in sorted(aliases, key=lambda x: len(x[0]), reverse=True)
                                if plan.concepts.get(k, {}).get("kind") == "observation" and p in low), None)
            if observation:
                population = previous_population or plan.cohort or question_population or _population(phrase, plan, aliases)
                plan.metrics.append({
                    "id": label or f"mean_{observation}", "label": label or f"mean_{_slug(observation)}",
                    "type": "mean", "observation": observation, "population": population,
                })
                continue
        percentage = bool(re.search(r"%|percent|share|proportion", low))
        if percentage:
            med = _population(phrase, plan, aliases)
            if med:
                base = main_population or previous_population or plan.cohort
                numerator = list(previous_population or base)
                for item in med:
                    if item["concept"] not in {x["concept"] for x in numerator}:
                        numerator.append(item)
                plan.metrics.append({
                    "id": label or "percentage", "label": label or "percentage",
                    "type": "percentage", "numerator_population": numerator,
                    "denominator_population": base,
                })
            elif re.search(r"that share|their share|share in %", low) and previous_population:
                plan.metrics.append({
                    "id": label or "share_pct", "label": label or "share_pct",
                    "type": "percentage", "numerator_population": previous_population,
                    "denominator_population": main_population or plan.cohort,
                })
            continue
        # "how many of them" carries the previous count population forward.
        if re.search(r"how many of them|their\b", low) and previous_population:
            population = list(previous_population)
            for item in _population(phrase, plan, aliases):
                if item["concept"] not in {x["concept"] for x in population}:
                    population.append(item)
        else:
            population = phrase_population
        patient_encounter_key = next(
            (
                item["concept"]
                for item in population
                if plan.concepts.get(item["concept"], {}).get("kind") == "encounter"
            ),
            None,
        )
        if patient_encounter_key:
            population = _merge_encounter_population(
                population, question_population, patient_encounter_key, plan
            )
        if any(plan.concepts.get(item["concept"], {}).get("kind") == "observation" for item in population):
            for item in question_population:
                if item["concept"] not in {x["concept"] for x in population}:
                    population.append(item)
        if "cohort" in low and plan.cohort:
            population = plan.cohort
        if re.search(r"\b(?:number of|patients|adults|women|men|people|children|stays|count|how many)\b", low) or not plan.metrics:
            if _ENCOUNTER_COUNT_CUE.search(low) and any(v.get("kind") == "encounter" for v in plan.concepts.values()):
                continue
            if population:
                metric_id = label or "patient_count"
                metric = {"id": metric_id, "label": metric_id, "type": "count_patients", "population": population}
                plan.metrics.append(metric)
                previous_population = population
                if not re.search(r"how many of them|their\b", low):
                    main_population = population
    for name, metric in formula_metrics.items():
        if metric["id"] not in added_formula_metrics:
            plan.metrics.append(metric)
    if metric_sentences and not any(metric["type"] == "count_encounters" for metric in plan.metrics):
        encounter_key = next((k for k, v in plan.concepts.items() if v.get("kind") == "encounter"), None)
        if encounter_key:
            phrase = next(
                (
                    item for item in phrases
                    if _ENCOUNTER_COUNT_CUE.search(item)
                ),
                None,
            )
            label = _slug(phrase) if phrase else "encounter_count"
            plan.metrics.append({
                "id": label or "encounter_count",
                "label": label or "encounter_count",
                "type": "count_encounters",
                "encounter": encounter_key,
                "population": [],
            })
    if not plan.metrics and plan.cohort:
        plan.metrics.append({"id": "cohort_size", "label": "cohort_size", "type": "count_patients", "population": plan.cohort})
    if not plan.metrics and question:
        question_population = _population(question, plan, aliases)
        question_encounter = next(
            (
                item["concept"]
                for item in question_population
                if plan.concepts.get(item["concept"], {}).get("kind") == "encounter"
            ),
            None,
        )
        question_event = _ENCOUNTER_COUNT_CUE.search(question)
        if question_encounter and question_event:
            mention = next(
                (
                    item
                    for item in encounter_mentions
                    if item["source"] == "question"
                    and item["concept"] == question_encounter
                    and item["event"]
                ),
                None,
            )
            label = _slug(mention["label_phrase"]) if mention else _slug(question_event.group(0))
            plan.metrics.append({
                "id": label or "encounter_count",
                "label": label or "encounter_count",
                "type": "count_encounters",
                "encounter": question_encounter,
                "population": question_population,
            })
            previous_population = question_population
        elif question_population:
            label = _slug(re.sub(r"\b(?:how many|what is|what are|by .*)\b", "", question, flags=re.I))
            if re.search(r"\b(?:mean|average)\b", question, re.I):
                observation = next((key for key, spec in plan.concepts.items() if spec.get("kind") == "observation"), None)
                metric = {
                    "id": label or "mean_value",
                    "label": label or "mean_value",
                    "type": "mean",
                    "observation": observation,
                    "population": question_population,
                }
                metric["population"] = [
                    item for item in question_population
                    if item.get("concept") != observation
                ]
            else:
                metric = {
                    "id": label or "patient_count",
                    "label": label or "patient_count",
                    "type": "count_patients",
                    "population": question_population,
                }
            plan.metrics.append(metric)
            previous_population = question_population
    if not plan.cohort:
        plan.cohort = previous_population or (plan.metrics[0].get("population", []) if plan.metrics else [])

    if re.search(r"\b(report the size of the cohort|cohort size)\b", all_text, re.I) and not any(m["id"] == "cohort_size" for m in plan.metrics):
        plan.metrics = [{"id": "cohort_size", "label": "cohort_size", "type": "count_patients", "population": plan.cohort}]

    text_lower = question.casefold()
    if re.search(r"\bconditions?\b", text_lower):
        plan.record_types.append("conditions")
    if re.search(r"\bmedications?\b|\bprescriptions?\b", text_lower):
        plan.record_types.append("medications")
    if re.search(r"\bobservations?\b|\bmeasurements?\b|\blab(?:oratory)? values?\b|\b(?:hba1c|bmi|blood pressure) values?\b", text_lower):
        plan.record_types.append("observations")
    if re.search(r"\bencounters?\b|\bvisits?\b|\bstays?\b|\bhospitali[sz]ations?\b|\badmissions?\b", text_lower):
        plan.record_types.append("encounters")
    plan.record_types = list(dict.fromkeys(plan.record_types))
    if not plan.cohort and plan.metrics:
        plan.cohort = plan.metrics[0].get("population", [])

    referenced_concepts = set()
    for metric in plan.metrics:
        if metric.get("encounter") in plan.concepts:
            referenced_concepts.add(metric["encounter"])
        for field in ("population", "numerator_population", "denominator_population"):
            for item in metric.get(field, []):
                if item.get("concept") in plan.concepts:
                    referenced_concepts.add(item["concept"])
    for item in plan.cohort:
        if item.get("concept") in plan.concepts:
            referenced_concepts.add(item["concept"])
    referenced_specs = [
        plan.concepts[key]
        for key in referenced_concepts
    ]
    encounter_specs = [
        spec for spec in referenced_specs
        if spec.get("kind") == "encounter"
    ]
    temporal_fields = {
        "start_year", "start_start", "start_end",
        "onset_year", "onset_start", "onset_end",
    }
    time_applied = any(
        any(spec.get(field) is not None for field in temporal_fields)
        for spec in referenced_specs
    )
    cue_sources = [question, *report_sources]
    for source in cue_sources:
        for cue in _CARE_SETTING_CUE.finditer(source):
            cue_text = cue.group(0).casefold()
            expected_type = None
            if (
                re.fullmatch(r"hospitali[sz]\w*|admitted|admissions?", cue_text)
                or re.fullmatch(r"hospitali[sz]ations?", cue_text)
            ):
                expected_type = "inpatient"
            elif cue_text == "inpatient":
                expected_type = "inpatient"
            elif cue_text == "emergency":
                expected_type = "emergency"
            elif cue_text == "outpatient":
                expected_type = "outpatient"
            applied = any(
                not expected_type
                or expected_type in [kind.casefold() for kind in spec.get("types", [])]
                for spec in encounter_specs
            )
            if not applied:
                warning = f"Could not apply '{_cue_context(source, cue)}' — staff review required."
                if warning not in plan.warnings:
                    plan.warnings.append(warning)
        for cue in _TIME_RESTRICTION_CUE.finditer(source):
            if not time_applied:
                warning = f"Could not apply '{_cue_context(source, cue)}' — staff review required."
                if warning not in plan.warnings:
                    plan.warnings.append(warning)
    if not plan.group_by and not plan.metrics:
        plan.warnings.append("No interpretable query or metric was found.")
    return plan
