"""Design tokens for the AIM Health Data Navigator (single source of truth for colours).

Palette: soft pink + powder pink + turquoise + deep teal + white. The Streamlit CSS in app.py, the Altair charts in
charts.py and .streamlit/config.toml all take their colours from here. Styling never touches calculations: changing a
token cannot change a released number.
"""

TOKENS = {
    "primary": "#269D9B",         # ocean teal: borders, rings, selected outlines, chart series 1
    "primary-text": "#1C7A78",    # darker teal for small text on white (4.6:1)
    "button": "#193B46",          # deep teal navy: primary action fill (white text, 12:1)
    "button-hover": "#24525F",
    "button-active": "#10272F",
    "turquoise": "#78D5D0",       # soft turquoise: light accents, step rings
    "mint": "#E4F8F6",            # mint frost: information rows, hover fills, selected view
    "pink": "#EBA6C4",            # rose pink: small accents only (never text)
    "powder": "#F4B8D1",          # powder pink: chip borders, dividers
    "pale-rose": "#FCE9F1",       # chips, sidebar selection
    "bg": "#FFF8FB",              # blush, near-white page background
    "surface": "#FFFFFF",
    "text": "#193B46",            # deep teal navy
    "muted": "#56707A",           # 5.3:1 on white
    "border": "#E8EEF0",          # cool near-white hairline
    "focus": "#269D9B",
    "success": "#1E8E5A",
    "success-tint": "#E8F7EF",
    "warning": "#9A6700",
    "warning-tint": "#FFF4DE",
    "error": "#B3261E",
    "error-tint": "#FDECEA",
    "info": "#1C7A78",
    "info-tint": "#E4F8F6",
}

# Series order: ocean teal first; the rest differ in hue so multi-series charts stay distinguishable.
CHART_PALETTE = ["#269D9B", "#D9739F", "#193B46", "#78D5D0", "#6C84C9", "#C9A227", "#7A4F2A", "#56707A", "#F4B8D1", "#A5D8D4"]
FIXED_COLOURS = {"female": "#D9739F", "male": "#269D9B"}
CHART_TEXT = {"title": TOKENS["text"], "label": "#3C5560", "grid": "#E6ECEE"}


def css_variables():
    return ":root {" + " ".join(f"--{name}: {value};" for name, value in TOKENS.items()) + "}"
