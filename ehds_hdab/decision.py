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
    if re.search(r"\b(?:insurance|insurer|credit|loan)\b", text) and re.search(
        r"\b(?:premium|pricing|underwrit|risk score|exclude|exclusion|eligibility)\w*\b", text
    ):
        return "insurance_credit"
    if re.search(r"\b(?:advertising campaign|marketing campaign|sales campaign|marketing to|advertising to|promot(?:e|ing|ion))\b", text):
        return "advertising_marketing"
    employment_context = bool(re.search(r"\b(?:employee|employees|employer|hr|hiring|employment|staff)\b", text))
    employment_decision = bool(re.search(r"\b(?:renew|terminate|hire|recruit|contract|sick leave|absence|performance)\w*\b", text))
    list_matching = bool(re.search(r"\bmatch\w*\b", text) and re.search(r"\b(?:employee|staff|personnel)\s+list\b", text))
    if employment_context and (employment_decision or list_matching):
        return "employment"
    if re.search(r"\b(?:re-identif|reidentif|identify people|identify individuals|match records)\w*\b", text):
        return "reidentification"
    harmful = re.search(r"\b(?:tobacco|nicotine|alcohol|illicit drugs?|weapons?|gambling)\b", text)
    product_cue = re.search(
        r"\b(?:develop\w*|manufactur\w*|product development|commercial\w*|sell|sales|market\w*|launch\w*)\b",
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
        (r"\bpublic health|prevention\b", "public health"),
        (r"\bpolicy|guideline|quality programme|quality program\b", "policy-making"),
        (r"\bofficial statistics|annual health report\b", "official statistics"),
        (r"\bteach|teaching|education|training module\b", "education/teaching"),
        (r"\btrain|validat|prediction model|algorithm\b", "training/testing algorithms"),
        (r"\bimprov\w* care|earlier intervention|quality programme|quality program\b", "improving care"),
    ):
        if re.search(pattern, text) and label not in categories:
            categories.append(label)
    return categories or ["public health"]


def decide(request):
    prohibited = _prohibited(request)
    reason_text = {
        "insurance_credit": "Using health data to set insurance premiums or credit decisions is prohibited (Art. 54).",
        "advertising_marketing": "Using health data for advertising or marketing is prohibited (Art. 54).",
        "employment": "Using health data for decisions about employees is prohibited (Art. 54). Matching it with an employee list would also re-identify people.",
        "harmful_products": "Developing or marketing harmful products is prohibited (Art. 54).",
        "reidentification": "Using health data to re-identify individuals is prohibited (Art. 54).",
    }
    if prohibited:
        return Decision("REJECT", ["54"], reason_text[prohibited], "prohibited purpose", [prohibited])
    requested_as = str(request.get("requested_as", "")).strip().casefold()
    categories = _purpose_categories(request)
    category_text = " and ".join(categories)
    if requested_as == "health data request":
        return Decision(
            "APPROVE_STATISTICAL",
            ["53", "69"],
            f"{category_text.capitalize()} are allowed purposes (Art. 53). Statistics answer the question (Art. 69).",
            category_text,
            ["statistical_request"],
        )
    if requested_as == "health data access application":
        text = " ".join(str(request.get(k, "") or "") for k in ("purpose", "question", "requested_format")).casefold()
        justifies_records = bool(
            re.search(r"aggregates? (?:are )?not enough|statistics? (?:are )?not enough|individual[- ]level|patient[- ]level|train\w*|validat\w*|prediction model|algorithm", text)
        )
        if justifies_records:
            return Decision(
                "APPROVE_SPE",
                ["53", "66", "68", "73"],
                f"{category_text.capitalize()} is an allowed purpose (Art. 53). The model or analysis needs patient-level data, so pseudonymised data is justified (Art. 66). Access only under a data permit inside a secure processing environment (Art. 68, 73).",
                category_text,
                ["individual_level_justification"],
            )
        return Decision(
            "APPROVE_STATISTICAL",
            ["53", "66", "69"],
            f"{category_text.capitalize()} is an allowed purpose (Art. 53). Pseudonymised records are not justified because statistics answer the question (Art. 66, 69).",
            category_text,
            ["statistics_sufficient"],
        )
    return Decision(
        "APPROVE_STATISTICAL",
        ["53", "69"],
        f"{category_text.capitalize()} are allowed purposes (Art. 53). Statistics answer the question (Art. 69).",
        category_text,
        ["default_statistical"],
    )
