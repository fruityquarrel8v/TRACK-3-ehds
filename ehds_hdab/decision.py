import re
from dataclasses import asdict, dataclass


@dataclass
class Decision:
    decision: str
    articles: list[str]
    reason: str
    purpose_category: str
    triggered_rules: list[str]

    def to_dict(self):
        return asdict(self)


def _prohibited(request):
    purpose_question = " ".join(
        str(request.get(key, "") or "") for key in ("purpose", "question")
    ).casefold()
    org = str((request.get("applicant") or {}).get("organisation", "") or "").casefold()
    text = purpose_question
    credit_context = re.search(
        r"\b(?:insurance|insurer|credit|loan|bank|lender|lending)\b", text
    )
    credit_decision = re.search(
        r"\b(?:premiums?|pricing|prices?|underwrit\w*|risk[\s-]+scor(?:e|ing)|"
        r"credit[\s-]+scor(?:e|ing)|creditworth\w*|"
        r"scor(?:e|ing)\s+(?:of\s+)?(?:loan|credit|insurance)\s+applicants?|"
        r"(?:loan|credit|insurance)\s+applicants?|"
        r"eligibility\s+for\s+(?:insurance|cover(?:age)?|(?:a\s+)?loans?|"
        r"credit|(?:a\s+)?mortgages?)|"
        r"(?:exclude|deny)\s+(?:(?:loan|credit|insurance)\s+)?"
        r"(?:applicants?|customers?|policyholders?|clients?)|"
        r"deny\s+(?:loan|credit|insurance)\s+applications?|"
        r"approve\s+or\s+reject\s+(?:the\s+)?"
        r"(?:loans?|credit applications?|mortgages?))\b",
        text,
    )
    if credit_context and credit_decision:
        return "insurance_credit"
    audience = re.search(
        r"\b(?:prescrib\w*|gps?\b|physician\w*|doctors?\b|consumers?\b|customers?\b|residents?\b|households?\b|"
        r"people\b|individuals?\b|patients?\b|population\b)\b",
        text,
    )
    marketing_organisation = re.search(r"\b(?:marketing|advertising)\s+(?:company|agency|firm|department)\b", text)
    sales_representatives = re.search(
        r"\bsales\s+(?:representatives?|reps?|team|force|department)\b|"
        r"\b(?:our|medical|pharma\w*)\s+reps?\b",
        text,
    )
    marketing_language = re.search(
        r"\b(?:for|our|its|their)\s+(?:\w+\s+){0,2}(?:marketing|advertising)\b|"
        r"\b(?:marketing|advertising)\s+(?:purposes?|materials?|to|of\s+(?:our|its|their))\b|"
        r"\badvertis(?:e|ing)\s+(?:our|its|their|the|a|new)\b|"
        r"\badvertis(?:ements?|ing)\s+(?:for|of)\s+(?:\w+[- ])?(?:products?|services?|medicines?|drugs?|brands?|supplements?)\b",
        text,
    )
    nonmarketing_reference = re.search(
        r"\bmarketing\s+(?:authori[sz]ation|surveillance)\b|"
        r"\bpost[- ]marketing\s+surveillance\b",
        text,
    )
    sales_program = re.search(
        r"\b(?:sales|marketing|advertising)\s+(?:campaign|strategy|targets?)\b",
        text,
    )
    product_promotion = re.search(
        r"\bpromot\w*\b.{0,60}\b(?:products?|medicines?|drugs?|brands?)\b|"
        r"\b(?:products?|medicines?|drugs?|brands?)\b.{0,60}\bpromot\w*\b",
        text,
    )
    targeted_commercially = bool(
        audience
        and re.search(r"\btarget\w*\b", text)
        and re.search(
            r"\b(?:sales|marketing|advertis\w*|products?|brands?|client launches?)\b",
            text,
        )
    )
    if (
        product_promotion
        or sales_program
        or (marketing_language and not nonmarketing_reference)
        or (sales_representatives and audience)
        or targeted_commercially
        or (marketing_organisation and re.search(r"\btarget\w*\b", text))
    ):
        return "advertising_marketing"
    employment_context = re.search(
        r"\b(?:employer|hr|human resources|hir\w*|recruit\w*|job applicants?)\b|"
        r"\b(?:its|our|their)\s+(?:staff|employees?|workforce|workers?)\b|"
        r"\b(?:employee|staff|personnel)\s+list\b",
        text,
    )
    employment_decision = bool(
        re.search(
            r"\b(?:promot\w*|dismiss\w*|fir(?:e|ing)\w*|terminat\w*|hir\w*|recruit\w*|"
            r"contract\w*|salary|salaries|career|renew\w*|sick leave|absence|"
            r"performance|disciplin\w*)\b",
            text,
        )
    )
    list_matching = bool(
        re.search(r"\bmatch\w*\b", text)
        and re.search(r"\b(?:employee|staff|personnel)\s+list\b", text)
    )
    identifies_worker = bool(
        re.search(
            r"\b(?:which|identify|know)\b.{0,60}\b(?:its|our|the)?\s*"
            r"(?:staff|employees?|workforce|workers?|job applicants?)\b",
            text,
        )
    )
    if employment_context and (employment_decision or list_matching or identifies_worker):
        return "employment"
    detrimental_decision = re.search(
        r"\b(?:detrimental|adverse|harmful|negative|unfavourable)\s+"
        r"(?:decisions?|actions?|treatment|outcomes?)\b|"
        r"\badversely?\s+(?:affect|impact|treat)\w*\b|"
        r"\b(?:decid\w*|decision)\b[^.;]{0,80}\b"
        r"(?:detrimental|adverse|harmful|negative|unfavourable)\b",
        text,
    )
    identified_targets = re.search(
        r"\b(?:identified\s+)?(?:individuals?|persons?|people|patients?|"
        r"customers?|applicants?|policyholders?)\b",
        text,
    )
    if identified_targets and detrimental_decision:
        return "detrimental_individual_decisions"
    if re.search(r"\b(?:re-identif|reidentif|identify people|identify individuals|match records)\w*\b", text):
        return "reidentification"
    harmful = re.search(
        r"\b(?:tobacco|nicotine|cigarett\w*|e-cigarett\w*|vap(?:e|es|ing)|alcohol|illicit drugs?|weapons?|gambling)\b",
        text,
    )
    product_cue = re.search(
        r"\b(?:develop\w*|manufactur\w*|product development|commercial\w*|sell|sales|launch\w*|"
        r"position\w*|flavours?|flavors?|brands?|(?:bring|brought|go|going)\s+to\s+market)\b",
        text,
    )
    if harmful and product_cue:
        return "harmful_products"
    # An organisation may support an existing purpose signal, but cannot trigger a rejection alone.
    if not text and re.search(r"\b(?:insurance|marketing|tobacco|nicotine|hr)\b", org):
        return None
    return None


def _purpose_categories(request):
    text = " ".join(str(request.get(key, "") or "") for key in ("purpose", "question")).casefold()
    categories = []
    for pattern, label in (
        (r"\bresearch|study\b", "scientific research"),
        (r"\bpublic health|prevention\b|\b(?:prevalence|incidence|epidemiolog\w*|vaccinat\w*|immuni[sz]\w*|"
         r"disease surveillance|disease burden|burden of \w+|screening programmes?)\b", "public health"),
        (r"\bpolicy|guideline|quality programme|quality program\b", "policy-making"),
        (r"\bofficial statistics|annual health report\b", "official statistics"),
        (r"\bteach|teaching|education|training module\b", "education/teaching"),
        (
            r"\b(?:train\w*|develop\w*|validat\w*|test\w*|build\w*)\b"
            r"[^.;]{0,80}\b(?:machine learning|models?|algorithms?|a\.?i\.?|"
            r"classifiers?|prediction)\b|"
            r"\b(?:machine learning|models?|algorithms?|a\.?i\.?|classifiers?|"
            r"prediction)\b[^.;]{0,80}\b"
            r"(?:train\w*|develop\w*|validat\w*|test\w*|build\w*)\b",
            "training/testing algorithms",
        ),
        (r"\bimprov\w* care|earlier intervention|quality programme|quality program\b|"
         r"\b(?:quality reports?|clinical audit|patient outcomes|worse outcomes|better outcomes|quality of care)\b",
         "improving care"),
    ):
        if re.search(pattern, text) and label not in categories:
            categories.append(label)
    return categories


PERMITTED_PURPOSES = (
    "scientific research, public health, policy-making, official statistics, education or teaching, "
    "training or testing algorithms, improving care"
)
UNRESOLVED_PURPOSE_REASON = (
    "No prohibited purpose was found, but the purpose was not matched to a specific Art. 53 category "
    f"({PERMITTED_PURPOSES}). Eligibility cannot be established automatically, so no data is released until a "
    "staff member has assessed the purpose or the applicant states a permitted purpose."
)


def _purpose_statement(categories):
    if not categories:
        return UNRESOLVED_PURPOSE_REASON
    joined = " and ".join(categories)
    if len(categories) == 1:
        return f"{joined.capitalize()} is an allowed purpose (Art. 53)."
    return f"{joined.capitalize()} are allowed purposes (Art. 53)."


def _records_justified(text):
    inadequacy = (
        r"(?:not enough|insufficient|inadequate|not sufficient|"
        r"cannot answer|cannot address|can't answer|can't address)"
    )
    aggregate_terms = r"(?:aggregates?|statistics?|statistical data|anonymi[sz]ed data)"
    if re.search(
        rf"\b{aggregate_terms}\b[^.;]{{0,80}}\b{inadequacy}\b|"
        rf"\b{inadequacy}\b[^.;]{{0,80}}\b{aggregate_terms}\b",
        text,
    ):
        return True

    action = r"(?:train\w*|develop\w*|validat\w*|test\w*|build\w*)"
    model = r"(?:machine\s+learning|models?|algorithms?|a\.?i\.?|classifiers?|prediction(?:\s+models?)?)"
    if re.search(rf"\b{action}\b(?:\W+\w+){{0,6}}\W+{model}\b", text) or re.search(
        rf"\b{model}\b(?:\W+\w+){{0,6}}\W+{action}\b", text
    ):
        return True

    level = r"(?:individual|patient|record)[ -]level"
    records = r"(?:data|records?|histories|history|information)"
    need = r"(?:need\w*|requir\w*|necessary|essential)"
    return bool(
        re.search(rf"\b{need}\b(?:\W+\w+){{0,4}}\W+{level}\b(?:\s+{records})?", text)
        or re.search(
            rf"\b{level}\b(?:\s+{records})?(?:\W+\w+){{0,5}}\W+\b{need}\b",
            text,
        )
    )


def decide(request):
    prohibited = _prohibited(request)
    reason_text = {
        "insurance_credit": "Using health data to set insurance premiums or credit decisions is prohibited (Art. 54).",
        "advertising_marketing": "Using health data for advertising or marketing is prohibited (Art. 54).",
        "employment": "Using health data for decisions about employees is prohibited (Art. 54). Matching it with an employee list would also re-identify people.",
        "detrimental_individual_decisions": "Using health data for decisions that may adversely affect identified individuals or groups is prohibited (Art. 54).",
        "harmful_products": "Developing or marketing harmful products is prohibited (Art. 54).",
        "reidentification": "Using health data to re-identify individuals is prohibited (Art. 54).",
    }
    if prohibited:
        return Decision("REJECT", ["54"], reason_text[prohibited], "prohibited purpose", [prohibited])
    requested_as = str(request.get("requested_as", "")).strip().casefold()
    categories = _purpose_categories(request)
    if not categories:
        # Absence of a prohibited purpose is not evidence of a permitted one: hold the request, release nothing.
        return Decision("REVIEW_REQUIRED", ["53"], UNRESOLVED_PURPOSE_REASON, "", ["purpose_not_established"])
    purpose_statement = _purpose_statement(categories)
    category_text = " and ".join(categories)
    if requested_as == "health data request":
        return Decision(
            "APPROVE_STATISTICAL",
            ["53", "69"],
            f"{purpose_statement} Statistics answer the question (Art. 69).",
            category_text,
            ["statistical_request"],
        )
    if requested_as == "health data access application":
        text = " ".join(str(request.get(k, "") or "") for k in ("purpose", "question", "requested_format")).casefold()
        if _records_justified(text):
            return Decision(
                "APPROVE_SPE",
                ["53", "66", "68", "73"],
                f"{purpose_statement} The model or analysis needs patient-level data, so pseudonymised data is justified (Art. 66). Access only under a data permit inside a secure processing environment (Art. 68, 73).",
                category_text,
                ["individual_level_justification"],
            )
        return Decision(
            "APPROVE_STATISTICAL",
            ["53", "66", "69"],
            f"{purpose_statement} Pseudonymised records are not justified because statistics answer the question (Art. 66, 69).",
            category_text,
            ["statistics_sufficient"],
        )
    return Decision(
        "APPROVE_STATISTICAL",
        ["53", "69"],
        f"{purpose_statement} Statistics answer the question (Art. 69).",
        category_text,
        ["default_statistical"],
    )
