"""Nord engineering theme: CSS injection + shared Plotly template ('smartbeam').

Render figures with st.plotly_chart(fig, theme=None) so Streamlit does not override the template.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Nord palette (https://www.nordtheme.com): Polar Night surfaces, Snow Storm text, Frost + Aurora accents.
COLORS = {
    "bg": "#2E3440",  # nord0  Polar Night: page background
    "panel": "#3B4252",  # nord1: cards / sidebar
    "panel_2": "#434C5E",  # nord2: raised surfaces, hover
    "border": "#4C566A",  # nord3
    "grid": "#434C5E",  # nord2
    "text": "#ECEFF4",  # nord6  Snow Storm
    "muted": "#A0A9BA",  # nord4 dimmed for secondary text (5:1 contrast on nord0)
    "cyan": "#88C0D0",  # nord8  Frost: primary accent, geometry, deflection
    "amber": "#EBCB8B",  # nord13 Aurora yellow: loads, maxima
    "green": "#A3BE8C",  # nord14 Aurora green: SAFE / pass
    "red": "#BF616A",  # nord11 Aurora red: YIELD PREDICTED / fail
    "violet": "#B48EAD",  # nord15 Aurora purple
    "blue": "#81A1C1",  # nord9  Frost
}
SERIES = [COLORS[k] for k in ("cyan", "amber", "violet", "green", "blue", "red")]
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"

pio.templates["smartbeam"] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT, color=COLORS["text"], size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=COLORS["panel"],
    colorway=SERIES,
    title=dict(font=dict(size=15, color=COLORS["text"]), x=0.01, xanchor="left"),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["border"], linecolor=COLORS["border"],
               tickcolor=COLORS["border"], title_font=dict(color=COLORS["muted"])),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["border"], linecolor=COLORS["border"],
               tickcolor=COLORS["border"], title_font=dict(color=COLORS["muted"])),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["muted"])),
    hoverlabel=dict(bgcolor=COLORS["panel_2"], bordercolor=COLORS["cyan"], font=dict(family=MONO, color=COLORS["text"])),
    margin=dict(l=60, r=24, t=48, b=48),
    colorscale=dict(diverging=[[0, "#5E81AC"], [0.5, "#ECEFF4"], [1, "#BF616A"]]),
))
# Merge once at import. A "a+b" default makes Plotly merge and re-validate both templates for every figure
# (~8 ms each); a single pre-merged template costs ~0.3 ms per figure.
pio.templates["smartbeam_dark"] = pio.templates.merge_templates("plotly_dark", "smartbeam")
pio.templates.default = "smartbeam_dark"
PLOTLY_TEMPLATE = "smartbeam_dark"

_CSS = f"""
<style>
:root {{ --sb-cyan:{COLORS['cyan']}; --sb-amber:{COLORS['amber']}; --sb-panel:{COLORS['panel']};
        --sb-border:{COLORS['border']}; --sb-muted:{COLORS['muted']}; }}
html, body, [class*="css"] {{ font-family:{FONT}; }}
.block-container {{ padding-top:2.2rem; max-width:1400px; }}
h1, h2, h3 {{ letter-spacing:-0.01em; }}
h1 {{ font-weight:700; }}
[data-testid="stSidebar"] {{ background:{COLORS['panel']}; border-right:1px solid var(--sb-border); }}
[data-testid="stMetric"] {{ background:var(--sb-panel); border:1px solid var(--sb-border); border-radius:10px;
        padding:14px 16px; }}
[data-testid="stMetricLabel"] {{ color:var(--sb-muted); font-size:0.8rem; letter-spacing:0.02em; }}
[data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] p {{ white-space:normal; overflow:visible;
        text-overflow:clip; }}
[data-testid="stMetricValue"] {{ font-family:{MONO}; font-weight:600; font-size:1.6rem; }}
[data-testid="stMetricValue"] div, [data-testid="stMetricValue"] p {{ white-space:normal; overflow:visible;
        text-overflow:clip; }}
[data-testid="stExpander"] {{ border:1px solid var(--sb-border); border-radius:10px; background:var(--sb-panel); }}
.stButton > button, .stDownloadButton > button {{ border-radius:8px; border:1px solid var(--sb-border);
        font-weight:600; transition:border-color .15s, color .15s; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:var(--sb-cyan); color:var(--sb-cyan); }}
/* Frost (nord8) is light: primary buttons and tags need Polar Night text, not white (2:1 -> 8:1 contrast). */
button[kind^="primary"], button[kind^="primary"] p, [data-baseweb="tag"], [data-baseweb="tag"] span {{
        color:{COLORS['bg']} !important; }}
button[kind^="primary"]:hover {{ background:#8FBCBB !important; border-color:#8FBCBB !important; }}
code {{ font-family:{MONO}; }}
.sb-header {{ border-left:3px solid var(--sb-cyan); padding:2px 0 2px 14px; margin-bottom:1.2rem; }}
.sb-header h1 {{ margin:0; padding:0; font-size:2rem; }}
.sb-header p {{ margin:4px 0 0; color:var(--sb-muted); }}
.sb-card {{ background:var(--sb-panel); border:1px solid var(--sb-border); border-radius:12px; padding:16px 18px; }}
.sb-badge {{ display:inline-block; padding:3px 10px; border-radius:999px; font-weight:700; font-size:0.78rem;
        letter-spacing:0.05em; border:1px solid currentColor; }}
</style>
"""

_BADGE = {"SAFE": COLORS["green"], "WARNING": COLORS["amber"], "YIELD PREDICTED": COLORS["red"],
          "BUCKLING PREDICTED": COLORS["red"],
          "PASS": COLORS["green"], "FAIL": COLORS["red"]}


def apply_theme() -> None:
    """Inject the global CSS (call once per run, from app.py)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str | None = None) -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="sb-header"><h1>{title}</h1>{sub}</div>', unsafe_allow_html=True)


def badge(text: str, color: str | None = None) -> str:
    """HTML for a status pill (SAFE / WARNING / YIELD PREDICTED / PASS / FAIL get their colours automatically)."""
    c = color or _BADGE.get(text.upper(), COLORS["cyan"])
    return f'<span class="sb-badge" style="color:{c}">{text}</span>'
