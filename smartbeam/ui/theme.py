"""Dark engineering theme: CSS injection + shared Plotly template ('smartbeam').

Render figures with st.plotly_chart(fig, theme=None) so Streamlit does not override the template.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

COLORS = {
    "bg": "#0B1220",  # page background (deep navy)
    "panel": "#111A2B",  # cards / sidebar
    "panel_2": "#16213A",  # raised surfaces, hover
    "border": "#24324D",
    "grid": "#1F2B42",
    "text": "#E6EDF7",
    "muted": "#8FA0BA",
    "cyan": "#22D3EE",  # primary accent: geometry, deflection, primary series
    "amber": "#F5A524",  # secondary accent: loads, highlights, maxima
    "green": "#34D399",  # SAFE / pass
    "red": "#F87171",  # YIELD PREDICTED / fail
    "violet": "#A78BFA",
    "blue": "#60A5FA",
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
    colorscale=dict(diverging=[[0, "#3B82F6"], [0.5, "#E6EDF7"], [1, "#EF4444"]]),
))
pio.templates.default = "plotly_dark+smartbeam"
PLOTLY_TEMPLATE = "plotly_dark+smartbeam"

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
