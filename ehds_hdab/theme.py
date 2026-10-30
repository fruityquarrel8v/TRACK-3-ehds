"""Design tokens for HealthCare Rocket (single source of truth for colours).

Palette: white surfaces, rose-pink brand accents, delicate pink borders, dark navy text, a little turquoise.
The Streamlit CSS in app.py, the Altair charts in charts.py and .streamlit/config.toml all take their colours from
here. Styling never touches calculations: changing a token cannot change a released number.
"""

TOKENS = {
    "primary": "#E77FA8",         # rose pink: brand accent, borders of focused inputs, active badges
    "primary-hover": "#CD5F8C",
    "button": "#BE4F7A",          # rose for filled buttons: white text reaches 4.6:1 (AA)
    "button-hover": "#B04471",
    "button-active": "#9A3A62",
    "primary-text": "#B04471",    # rose for small text on white (5.6:1)
    "pale-pink": "#FFF0F6",
    "selected-pink": "#F8D6E5",
    "border": "#F2DDE6",          # delicate pink hairline
    "turquoise": "#78D5D0",       # thin accents only
    "teal": "#268C8A",            # focus ring, icons (3.9:1 on white)
    "pale-turquoise": "#F0FBFA",
    "bg": "#FFFFFF",
    "surface": "#FFFFFF",
    "text": "#1C2844",
    "muted": "#657087",           # 5.0:1 on white
    "focus": "#268C8A",
    "success": "#1E8E5A",
    "success-tint": "#E8F7EF",
    "warning": "#9A6700",
    "warning-tint": "#FFF4DE",
    "error": "#B3261E",
    "error-tint": "#FDECEA",
    "info": "#1F7A78",
    "info-tint": "#F0FBFA",
}

# Series order: rose first; the rest differ in hue/lightness so multi-series charts stay distinguishable.
CHART_PALETTE = ["#E77FA8", "#268C8A", "#1C2844", "#78D5D0", "#CD5F8C", "#6C84C9", "#C9A227", "#7A4F2A", "#657087", "#F2B8CC"]
FIXED_COLOURS = {"female": "#E77FA8", "male": "#268C8A"}
CHART_TEXT = {"title": TOKENS["text"], "label": "#44506B", "grid": "#EEE9EE"}


def css_variables():
    return ":root {" + " ".join(f"--{name}: {value};" for name, value in TOKENS.items()) + "}"
