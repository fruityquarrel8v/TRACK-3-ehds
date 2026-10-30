"""Clinically standard categories for aggregate distributions of continuous measurements.

Bins are half-open [lo, hi): a value equal to `hi` belongs to the next bin. Used to turn a
"distribution of BMI" question into patient counts per category, so that the small-cell rule
and the disclosure ledger apply to every category exactly as to any other count.
"""

STANDARD_BINS = {
    # LOINC 39156-5, body mass index (WHO classes)
    "39156-5": [
        {"lo": None, "hi": 18.5, "label": "<18.5 (underweight)"},
        {"lo": 18.5, "hi": 25, "label": "18.5–24.9 (normal)"},
        {"lo": 25, "hi": 30, "label": "25–29.9 (overweight)"},
        {"lo": 30, "hi": 35, "label": "30–34.9 (obesity I)"},
        {"lo": 35, "hi": 40, "label": "35–39.9 (obesity II)"},
        {"lo": 40, "hi": None, "label": "≥40 (obesity III)"},
    ],
    # LOINC 4548-4, HbA1c % (ADA glycaemic categories / common control thresholds)
    "4548-4": [
        {"lo": None, "hi": 5.7, "label": "<5.7"},
        {"lo": 5.7, "hi": 6.5, "label": "5.7–6.4"},
        {"lo": 6.5, "hi": 7.0, "label": "6.5–6.9"},
        {"lo": 7.0, "hi": 8.0, "label": "7.0–7.9"},
        {"lo": 8.0, "hi": 9.0, "label": "8.0–8.9"},
        {"lo": 9.0, "hi": None, "label": "≥9.0"},
    ],
    # LOINC 8480-6, systolic blood pressure mmHg (ESC/ESH grades)
    "8480-6": [
        {"lo": None, "hi": 120, "label": "<120 (optimal)"},
        {"lo": 120, "hi": 130, "label": "120–129 (normal)"},
        {"lo": 130, "hi": 140, "label": "130–139 (high-normal)"},
        {"lo": 140, "hi": 160, "label": "140–159 (grade 1)"},
        {"lo": 160, "hi": 180, "label": "160–179 (grade 2)"},
        {"lo": 180, "hi": None, "label": "≥180 (grade 3)"},
    ],
    # LOINC 62238-1, eGFR mL/min/1.73m² (KDIGO G stages)
    "62238-1": [
        {"lo": 90, "hi": None, "label": "≥90 (G1)"},
        {"lo": 60, "hi": 90, "label": "60–89 (G2)"},
        {"lo": 45, "hi": 60, "label": "45–59 (G3a)"},
        {"lo": 30, "hi": 45, "label": "30–44 (G3b)"},
        {"lo": 15, "hi": 30, "label": "15–29 (G4)"},
        {"lo": None, "hi": 15, "label": "<15 (G5)"},
    ],
}
BIN_SOURCES = {
    "39156-5": "WHO BMI classification",
    "4548-4": "ADA glycaemic categories and common HbA1c control thresholds",
    "8480-6": "ESC/ESH blood-pressure grades",
    "62238-1": "KDIGO GFR categories",
}


def bins_for(loinc):
    return [dict(b) for b in STANDARD_BINS.get(str(loinc), [])]
