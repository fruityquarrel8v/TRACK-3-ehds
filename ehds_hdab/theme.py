"""Design tokens for the AIM Health Data Navigator (single source of truth for colours).

The Streamlit CSS in app.py, the Altair charts in charts.py and .streamlit/config.toml all take their colours
from here, so the palette is defined once. Styling never touches calculations: changing a token cannot change a
released number.
"""

TOKENS = {
    "primary": "#C44779",         # raspberry: primary actions, selected states
    "primary-hover": "#A93A66",
    "primary-active": "#8F2F56",
    "brand": "#E88BA9",           # dusty rose: soft accents only (never for text)
    "pink-tint": "#FBE9F0",
    "secondary": "#B9A5F5",       # lavender: secondary actions, information cards
    "lavender-tint": "#F1ECFD",
    "accent": "#6C84C9",          # muted blue: supporting accent, focus ring
    "bg": "#FFF6FA",              # blush white page background
    "surface": "#FFFFFF",         # cards, forms, results, chart containers
    "text": "#182342",            # midnight navy
    "muted": "#5F6A89",           # 5.6:1 on white
    "border": "#F2DCE7",
    "focus": "#6C84C9",
    "success": "#1E8E5A",
    "success-tint": "#E8F7EF",
    "warning": "#9A6700",
    "warning-tint": "#FFF4DE",
    "error": "#B3261E",
    "error-tint": "#FDECEA",
    "info": "#4F66A8",
    "info-tint": "#ECF0FB",
}

# Chart series order: raspberry first for single-series charts; the following colours are chosen to stay distinct
# from each other (hue, not shade) so multi-series charts do not rely on near-identical pinks.
CHART_PALETTE = ["#C44779", "#6C84C9", "#B9A5F5", "#4A9D98", "#182342", "#E88BA9", "#C9A227", "#7A4F2A", "#5F6A89", "#F2B8CC"]
FIXED_COLOURS = {"female": "#C44779", "male": "#6C84C9"}
CHART_TEXT = {"title": TOKENS["text"], "label": "#3A4668", "grid": "#E9E3EE"}


def css_variables():
    return ":root {" + " ".join(f"--{name}: {value};" for name, value in TOKENS.items()) + "}"
