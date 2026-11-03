"""UI helpers for the design-tool pages (Design Comparison, Optimization, Design Advisor, Reports).
OWNER: design tools & reporting engineer. Streamlit/Plotly glue only; every number comes from the engine."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smartbeam import advisor, optimize
from smartbeam.metrics import DesignMetrics
from smartbeam.model import BeamModel
from smartbeam.report import format_check
from smartbeam.ui.plots_core import SHAPE_LABEL, dims_text, rgba
from smartbeam.ui.theme import COLORS

STATUS_COLOR = {"OPTIMIZED": COLORS["green"], "INFEASIBLE": COLORS["red"], "FAILED": COLORS["amber"],
                "PASS": COLORS["green"], "FAIL": COLORS["red"], "SAFE": COLORS["green"],
                "WARNING": COLORS["amber"], "YIELD PREDICTED": COLORS["red"], "BUCKLING PREDICTED": COLORS["red"]}


def pill(text: str, color: str | None = None) -> str:
    c = color or STATUS_COLOR.get(text.upper(), COLORS["cyan"])
    return f'<span class="sb-badge" style="color:{c};font-size:0.9rem;padding:4px 12px">{text}</span>'


def g(v, sig: int = 4) -> float:
    """Round to `sig` significant figures for display tables (keeps the cell numeric/sortable)."""
    if v is None or not np.isfinite(v) or v == 0:
        return v
    return float(f"{v:.{sig}g}")


SENSE = {"<=": "≤", ">=": "≥"}


def checks_frame(checks) -> pd.DataFrame:
    """ConstraintCheck list -> display table (lengths in mm)."""
    rows = []
    for c in checks:
        name, val, lim, unit = format_check(c)
        rows.append({"Constraint": name + (f" ({unit})" if unit else ""), "Value": g(val),
                     "Requirement": f"{SENSE.get(c.sense, c.sense)} {g(lim)}", "Margin (%)": g(c.margin_pct, 3),
                     "Result": "PASS" if c.satisfied else "FAIL"})
    return pd.DataFrame(rows)


def show_checks(checks) -> None:
    df = checks_frame(checks)
    if df.empty:
        st.caption("No constraints to show.")
        return
    style = df.style.map(lambda v: f"color:{STATUS_COLOR[v]};font-weight:700" if v in ("PASS", "FAIL") else "",
                         subset=["Result"]).format("{:.4g}", subset=["Value", "Margin (%)"], na_rep="—")
    st.dataframe(style, hide_index=True, width="stretch")


def metrics_row(m: DesignMetrics, **extra) -> dict:
    """One display row of DesignMetrics (mm, MPa, kg, ₹)."""
    return {**extra, "Material": m.material, "L (m)": g(m.L),
            "Section": SHAPE_LABEL[m.shape], "Dimensions": ", ".join(f"{k} {v * 1e3:.4g}" for k, v in m.dims.items()) + " mm",
            "δmax (mm)": g(m.delta_max * 1e3), "σmax (MPa)": g(m.sigma_max / 1e6), "Mmax (N·m)": g(m.M_max),
            "FOS": g(m.fos, 3), "A (mm²)": g(m.A * 1e6), "Volume (cm³)": g(m.volume * 1e6), "Mass (kg)": g(m.mass),
            "Cost (₹)": g(m.cost) if m.cost is not None else None, "Method": m.method}


def before_after_frame(before: DesignMetrics, after: DesignMetrics, labels=("Before", "After")) -> pd.DataFrame:
    """Side-by-side metrics with % change (after vs before)."""
    rows = [("Dimensions (mm)", ", ".join(f"{k} {v * 1e3:.4g}" for k, v in before.dims.items()),
             ", ".join(f"{k} {v * 1e3:.4g}" for k, v in after.dims.items()), None)]
    for label, attr, scale in (("Max deflection |v| (mm)", "delta_max", 1e3), ("Max stress |σ| (MPa)", "sigma_max", 1e-6),
                               ("Max moment |M| (N·m)", "M_max", 1.0), ("Yield FOS", "fos", 1.0),
                               ("Area A (mm²)", "A", 1e6), ("Mass (kg)", "mass", 1.0), ("Cost (₹)", "cost", 1.0)):
        b, a = getattr(before, attr), getattr(after, attr)
        if b is None or a is None:
            continue
        pct = (a - b) / b * 100 if b and math.isfinite(a) and math.isfinite(b) else None
        rows.append((label, f"{b * scale:.4g}", f"{a * scale:.4g}", None if pct is None else f"{pct:+.1f} %"))
    if before.material != after.material:
        rows.insert(1, ("Material", before.material, after.material, None))
    if before.shape != after.shape:
        rows.insert(1, ("Section", SHAPE_LABEL[before.shape], SHAPE_LABEL[after.shape], None))
    return pd.DataFrame(rows, columns=["Quantity", labels[0], labels[1], "Change"])


def section_overlay(items: list[tuple], height: int = 260) -> go.Figure:
    """Cross-sections drawn to the same scale. items: (Section, label, colour)."""
    fig = go.Figure()
    ext = 1e-9
    t = np.linspace(0, 2 * np.pi, 121)
    for s, label, color in items:
        if s.shape == "rect":
            b, h = s.b * 1e3 / 2, s.h * 1e3 / 2
            xs, ys = [-b, b, b, -b, -b], [-h, -h, h, h, -h]
            ext = max(ext, b, h)
        else:
            R = s.depth * 1e3 / 2
            xs, ys = list(R * np.cos(t)), list(R * np.sin(t))
            if s.shape == "hollow_circ":  # outer ring then inner hole (even-odd fill)
                r = s.d * 1e3 / 2
                xs += [None] + list(r * np.cos(t))
                ys += [None] + list(r * np.sin(t))
            ext = max(ext, R)
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", mode="lines", name=f"{label}: {dims_text(s)}",
                                 line=dict(color=color, width=2), fillcolor=rgba(color, 0.18), hoverinfo="name"))
    e = ext * 1.15
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.02, yanchor="top"),
                      xaxis=dict(range=[-e, e], scaleanchor="y", title="mm", showgrid=False, zeroline=False),
                      yaxis=dict(range=[-e, e], showgrid=False, zeroline=False))
    return fig


# --------------------------------------------------------------------------- cached engine calls
@st.cache_data(max_entries=32, show_spinner=False)
def run_advise(model_json: str, delta_allow: float, min_fos: float):
    return advisor.advise(BeamModel.from_json(model_json), delta_allow=delta_allow, min_fos=min_fos)


@st.cache_data(max_entries=16, show_spinner=False)
def run_pareto(model_json: str, problem: optimize.OptimizationProblem, material_keys: tuple[str, ...], n_points: int):
    return optimize.pareto_front(BeamModel.from_json(model_json), problem, list(material_keys), n_points)
