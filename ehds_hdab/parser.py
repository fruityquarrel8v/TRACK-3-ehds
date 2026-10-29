import re

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
)


def _slug(text):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", normalize(text))).strip("_")


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
    parts = re.split(r",\s+| and (?=(?:" + "|".join(re.escape(x) for x in _BASE_STARTS) + r")\b)", text, flags=re.I)
    return [part.strip(" .") for part in parts if part.strip(" .")]


def _sentence_scope(sentences):
    return " ".join(sentences)


def parse_request(request):
    plan = QueryPlan()
    definitions = _split_definitions(request.get("definitions"))
    question = str(request.get("question", "") or "")
    all_text = " ".join([question, *definitions])
    definition_names = {}
    age_bands = []
    global_active = bool(re.search(r"\bmedication is active\b|\bactive (?:medication|prescription|.*?\bATC)\b", all_text, re.I))

    for sentence in definitions:
        if "age band" in sentence.casefold():
            age_bands = _parse_bands(sentence)
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

    for name, spec in list(plan.concepts.items()):
        if spec.get("kind") == "observation":
            pick = re.search(rf"(?i)\b(latest|earliest|max(?:imum)?|min(?:imum)?)\b[^.]*{re.escape(name)}", all_text)
            if pick:
                spec["pick"] = {"max": "max", "maximum": "max", "min": "min", "minimum": "min"}.get(pick.group(1).casefold(), pick.group(1).casefold())

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
        if re.search(r"\bcount stays\b|\bcount (?:encounters|visits)\b", s, re.I)
    ]
    previous_population = []
    main_population = []
    count_metrics = []
    if plan.cohort:
        previous_population = plan.cohort
    if report_text:
        phrases = _metric_parts(report_text)
    else:
        phrases = []
    # Formula definitions are parsed as explicit percentage metrics.
    formula = next((s for s in definitions if re.search(r"\bprevalence\s*=", s, re.I)), None)
    if formula:
        parts = re.split(r"[/×x]", formula.split("=", 1)[1])
        if len(parts) >= 2:
            numerator = _population(parts[0], plan, aliases)
            denominator = _population(parts[1], plan, aliases)
            plan.metrics.append({
                "id": "prevalence_pct", "label": "prevalence_pct", "type": "percentage",
                "numerator_population": numerator, "denominator_population": denominator,
            })

    for sentence in metric_sentences:
        encounter_key = next((k for k, v in plan.concepts.items() if v.get("kind") == "encounter"), None)
        if encounter_key:
            start_year = re.search(r"\bin\s+(\d{4})", sentence)
            if start_year:
                plan.concepts[encounter_key]["start_year"] = int(start_year.group(1))
            plan.metrics.append({
                "id": "encounter_count", "label": "inpatient_stays", "type": "count_encounters",
                "encounter": encounter_key, "population": [],
            })

    for phrase in phrases:
        low = phrase.casefold().strip()
        if not low:
            continue
        label = _slug(phrase)
        if "prevalence" in low and formula:
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
        percentage = bool(re.search(r"%|percent|share|proportion|prevalence", low))
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
            elif formula:
                # The formula metric already carries the fully defined populations.
                continue
            continue
        # "how many of them" carries the previous count population forward.
        if re.search(r"how many of them|their\b", low) and previous_population:
            population = list(previous_population)
            for item in _population(phrase, plan, aliases):
                if item["concept"] not in {x["concept"] for x in population}:
                    population.append(item)
        else:
            population = _population(phrase, plan, aliases)
        if any(plan.concepts.get(item["concept"], {}).get("kind") == "observation" for item in population):
            for item in question_population:
                if item["concept"] not in {x["concept"] for x in population}:
                    population.append(item)
        if "cohort" in low and plan.cohort:
            population = plan.cohort
        if re.search(r"\b(?:number of|patients|adults|women|men|people|children|stays|count|how many)\b", low) or not plan.metrics:
            if re.search(r"\b(stays|encounters|visits)\b", low) and any(v.get("kind") == "encounter" for v in plan.concepts.values()):
                continue
            if population:
                metric_id = label or "patient_count"
                metric = {"id": metric_id, "label": metric_id, "type": "count_patients", "population": population}
                plan.metrics.append(metric)
                previous_population = population
                if not re.search(r"how many of them|their\b", low):
                    main_population = population
                count_metrics.append(metric)
    if formula and not any(m["type"] == "percentage" and "prevalence" in m["id"] for m in plan.metrics):
        metric = next(m for m in plan.metrics if m["type"] == "percentage" and m["id"] == "prevalence_pct")
        plan.metrics.append(metric)
    if not plan.metrics and plan.cohort:
        plan.metrics.append({"id": "cohort_size", "label": "cohort_size", "type": "count_patients", "population": plan.cohort})
    if not plan.metrics and question:
        question_population = _population(question, plan, aliases)
        if question_population:
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
    if re.search(r"\bencounters?\b|\bvisits?\b|\bstays?\b", text_lower):
        plan.record_types.append("encounters")
    plan.record_types = list(dict.fromkeys(plan.record_types))
    if not plan.cohort and plan.metrics:
        plan.cohort = plan.metrics[0].get("population", [])
    if not plan.group_by and not plan.metrics:
        plan.warnings.append("No interpretable query or metric was found.")
    return plan
