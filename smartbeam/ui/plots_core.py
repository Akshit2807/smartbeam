"""UI helpers for the core pages (Home, Design Studio, Analysis, FEM, Beam Theories, 3D, Safety, About).
OWNER: lead UI engineer. Plotting + Streamlit glue only; every number comes from the engine.
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from smartbeam import fem, sections
from smartbeam.model import BeamModel, ModelError, UnstableStructureError, classify
from smartbeam.ui.state import analyze_model, get_model
from smartbeam.ui.theme import COLORS

SHAPE_LABEL = {"rect": "Rectangular", "solid_circ": "Solid circular", "hollow_circ": "Hollow circular"}
METHOD_LABEL = {"analytical": "Analytical (statics + integration)", "fem-euler": "FEM · Euler–Bernoulli beam",
                "fem-frame": "FEM · planar frame (u, v, θ)"}
STATUS_COLOR = {"SAFE": COLORS["green"], "WARNING": COLORS["amber"], "YIELD PREDICTED": COLORS["red"],
                "BUCKLING PREDICTED": COLORS["red"]}


# --------------------------------------------------------------------------- text helpers
def dims_text(section) -> str:
    d = {k: v * 1e3 for k, v in section.dims().items()}
    if section.shape == "rect":
        return f"{d['b']:.4g} × {d['h']:.4g} mm"
    if section.shape == "solid_circ":
        return f"Ø{d['d']:.4g} mm"
    return f"Ø{d['D']:.4g} / Ø{d['d']:.4g} mm"


def model_caption(model: BeamModel) -> str:
    return (f"**{model.name or 'Untitled'}** · {SHAPE_LABEL[model.section.shape]} {dims_text(model.section)} · "
            f"{model.material.name} · L = {model.L:.4g} m · {classify(model)} · "
            f"{len(model.supports)} support(s), {len(model.loads)} load(s)")


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def status_badge(status: str) -> str:
    c = STATUS_COLOR.get(status, COLORS["cyan"])
    return f'<span class="sb-badge" style="color:{c};font-size:0.95rem;padding:5px 14px">{status}</span>'


def card(title: str, body: str, accent: str | None = None) -> None:
    c = accent or COLORS["cyan"]
    st.markdown(f'<div class="sb-card" style="border-top:2px solid {c}"><div style="font-weight:700;'
                f'margin-bottom:4px">{title}</div><div style="color:{COLORS["muted"]};font-size:0.9rem">{body}'
                f'</div></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- analysis access
def require_result(model: BeamModel | None = None, **kw):
    """Analyse the current (or given) model; on invalid/unstable input show the error and stop the page."""
    model = model or get_model()
    try:
        return analyze_model(model, **kw)
    except (ModelError, UnstableStructureError) as exc:
        kind = "Unstable structure" if isinstance(exc, UnstableStructureError) else "Invalid model"
        st.error(f"**{kind}.** {exc}")
        st.page_link("views/design_studio.py", label="Fix the design in Design Studio", icon=":material/edit:")
        st.stop()


def show_warnings(result) -> None:
    for w in result.warnings:
        st.warning(w, icon=":material/info:")


@st.cache_data(max_entries=64, show_spinner=False)
def fem_solve(model_json: str, n_elements: int, theory: str = "euler"):
    return fem.solve(BeamModel.from_json(model_json), n_elements=n_elements, theory=theory)


@st.cache_data(max_entries=32, show_spinner=False)
def fem_convergence(model_json: str, theory: str = "euler"):
    return fem.convergence_study(BeamModel.from_json(model_json), theory=theory)


@st.cache_data(max_entries=32, show_spinner=False)
def fem_compare(model_json: str, n_elements: int = 20):
    return fem.compare_theories(BeamModel.from_json(model_json), n_elements=n_elements)


# --------------------------------------------------------------------------- plots
def _hline_zero(fig, row=None, col=None):
    fig.add_hline(y=0, line=dict(color=COLORS["border"], width=1), row=row, col=col)


def field_trace(x, y, name: str, color: str, unit: str, fill: bool = True, dash: str | None = None):
    return go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=2.4, dash=dash),
                      fill="tozeroy" if fill else None, fillcolor=rgba(color, 0.14),
                      hovertemplate=f"x = %{{x:.4g}} m<br>{name} = %{{y:.5g}} {unit}<extra></extra>")


def label_range(y, ext_value: float, frac: float = 0.2) -> list[float]:
    """y-axis range with headroom on the side of the extreme, so its text label is not clipped."""
    y = np.asarray(y, dtype=float)
    lo, hi = min(float(y.min()), 0.0), max(float(y.max()), 0.0)
    pad = frac * ((hi - lo) or 1.0)
    below, above = (pad, 0.03 * pad) if ext_value < 0 else (0.03 * pad, pad)
    return [lo - below, hi + above]


def extreme_marker(ext, scale: float, unit: str, color: str, label: str, L: float | None = None):
    """Marker + label at a signed extreme. The label sits above a positive value and below a negative one, and
    leans inward near the ends of the span when L is given, so the plot edge does not clip it."""
    side = "center"
    if L:
        side = "right" if ext.x < 0.2 * L else "left" if ext.x > 0.8 * L else "center"
    return go.Scatter(x=[ext.x], y=[ext.value * scale], mode="markers+text", showlegend=False,
                      marker=dict(size=9, color=color, line=dict(color=COLORS["text"], width=1)),
                      text=[f"{label} {ext.value * scale:.4g} {unit}"],
                      textposition=f"{'top' if ext.value >= 0 else 'bottom'} {side}",
                      textfont=dict(color=color, size=12),
                      hovertemplate=f"{label} = %{{y:.5g}} {unit} at x = %{{x:.4g}} m<extra></extra>")


def field_figure(x, y, ext, name: str, unit: str, scale: float, color: str, height: int = 300,
                 title: str | None = None) -> go.Figure:
    """One diagram (e.g. BMD) with filled area and its signed extreme marked."""
    ys = np.asarray(y) * scale
    fig = go.Figure([field_trace(x, ys, name, color, unit),
                     extreme_marker(ext, scale, unit, color, "max", L=float(x[-1]))])
    _hline_zero(fig)
    fig.update_yaxes(range=label_range(ys, ext.value))
    fig.update_layout(height=height, title=title, showlegend=False, xaxis_title="x (m)",
                      yaxis_title=f"{name} ({unit})", margin=dict(t=48 if title else 24))
    return fig


@st.cache_resource(max_entries=256, show_spinner=False)
def _figure_cache(key: tuple, _build) -> go.Figure:
    return _build()


def cached_figure(key: tuple, build) -> go.Figure:
    """Build a figure once per key and reuse it across reruns and sessions.

    Plotly validates every shape/trace on construction, which dominates rerun time on small hosts. Sharing the
    object is safe because st.plotly_chart only reads it (to_dict / to_json); callers must not mutate the result.
    """
    return _figure_cache(key, build)


def beam_schematic(model: BeamModel, mesh=None, height: int = 230, show_dims: bool = True) -> go.Figure:
    """2D engineering sketch: beam, supports, loads (+ optional FE mesh nodes/elements). Cached per model."""
    mesh_key = None if mesh is None else (mesh.nodes.tobytes(), mesh.elements.tobytes())
    return cached_figure(("schematic", model.to_json(indent=None), mesh_key, height, show_dims),
                         lambda: _beam_schematic(model, mesh, height, show_dims))


def _beam_schematic(model: BeamModel, mesh, height: int, show_dims: bool) -> go.Figure:
    L = model.L
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=L, y0=-0.08, y1=0.08, fillcolor=rgba(COLORS["cyan"], 0.25),
                  line=dict(color=COLORS["cyan"], width=2))
    sz = 0.035 * L
    for k, sp in enumerate(model.supports):
        hover = f"Support {k + 1}: {sp.type} at x = {sp.x:.4g} m"
        if sp.type == "fixed":
            side = -1 if sp.x <= L / 2 else 1
            x0, x1 = sorted((sp.x, sp.x + side * 0.6 * sz))
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=-0.55, y1=0.55, fillcolor=rgba(COLORS["muted"], 0.35),
                          line=dict(color=COLORS["muted"], width=1))
            for yy in np.linspace(-0.5, 0.4, 6):  # hatch
                fig.add_shape(type="line", x0=sp.x + side * 0.6 * sz, x1=sp.x + side * 1.2 * sz, y0=yy,
                              y1=yy + 0.12, line=dict(color=COLORS["muted"], width=1))
            fig.add_trace(go.Scatter(x=[sp.x], y=[0], mode="markers", marker=dict(size=1, opacity=0),
                                     hovertemplate=hover + "<extra></extra>", showlegend=False))
        else:
            fig.add_trace(go.Scatter(x=[sp.x], y=[-0.24], mode="markers", showlegend=False,
                                     marker=dict(symbol="triangle-up", size=22, color=COLORS["muted"]),
                                     hovertemplate=hover + "<extra></extra>"))
            if sp.type == "roller":
                fig.add_trace(go.Scatter(x=[sp.x - 0.3 * sz, sp.x + 0.3 * sz], y=[-0.45, -0.45], mode="markers",
                                         marker=dict(size=7, color=COLORS["muted"]), hoverinfo="skip",
                                         showlegend=False))

    amber = COLORS["amber"]
    for ld in model.point_loads:
        dx, dy = math.cos(math.radians(ld.angle_deg)), math.sin(math.radians(ld.angle_deg))
        head_y = 0.08 if dy < -0.5 else (-0.08 if dy > 0.5 else 0.0)
        mag = f"{ld.magnitude / 1e3:.3g} kN" if ld.magnitude >= 1e3 else f"{ld.magnitude:.3g} N"
        fig.add_annotation(x=ld.x, y=head_y, ax=-dx * 55, ay=dy * 55, showarrow=True, arrowhead=2, arrowsize=1.3,
                           arrowwidth=2.4, arrowcolor=amber, text=mag, font=dict(color=amber, size=12),
                           hovertext=f"Point load {mag} at x = {ld.x:.4g} m, angle {ld.angle_deg:g}°")
    dls = model.distributed_loads
    qmax = max((max(abs(w.w1), abs(w.w2)) for w in dls), default=1.0) or 1.0
    for w in dls:
        sgn = 1 if w.direction == "down" else -1
        base = 0.08 * sgn
        h1, h2 = 0.15 + 0.5 * abs(w.w1) / qmax, 0.15 + 0.5 * abs(w.w2) / qmax
        xs = [w.x1, w.x1, w.x2, w.x2]
        ys = [base, base + sgn * h1, base + sgn * h2, base]
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", mode="lines", line=dict(color=amber, width=1.5),
                                 fillcolor=rgba(amber, 0.18), showlegend=False,
                                 hovertemplate=f"Distributed load {w.w1:.4g} → {w.w2:.4g} N/m ({w.direction}), "
                                               f"x = {w.x1:.4g}–{w.x2:.4g} m<extra></extra>"))
        for xi in np.linspace(w.x1, w.x2, max(3, int(12 * (w.x2 - w.x1) / L) + 1)):
            hi = base + sgn * (h1 + (h2 - h1) * (xi - w.x1) / (w.x2 - w.x1))
            fig.add_annotation(x=xi, y=base, ax=xi, ay=hi, axref="x", ayref="y", showarrow=True, arrowhead=2,
                               arrowwidth=1.3, arrowcolor=amber, text="")
        txt = (f"{abs(w.w1) / 1e3:.3g} kN/m" if w.w1 == w.w2 else
               f"{abs(w.w1) / 1e3:.3g} → {abs(w.w2) / 1e3:.3g} kN/m")
        fig.add_annotation(x=(w.x1 + w.x2) / 2, y=base + sgn * (max(h1, h2) + 0.12), text=txt, showarrow=False,
                           font=dict(color=amber, size=12))
    for mo in model.moment_loads:
        sym = "↺" if mo.M > 0 else "↻"
        fig.add_annotation(x=mo.x, y=0.0, text=sym, showarrow=False, font=dict(color=amber, size=34),
                           hovertext=f"Moment {abs(mo.M):.4g} N·m {'CCW' if mo.M > 0 else 'CW'} at x = {mo.x:.4g} m")
        fig.add_annotation(x=mo.x, y=0.55, text=f"{abs(mo.M) / 1e3:.3g} kN·m", showarrow=False,
                           font=dict(color=amber, size=12))

    if mesh is not None:
        nodes = mesh.nodes
        fig.add_trace(go.Scatter(x=nodes, y=np.zeros_like(nodes), mode="markers+text", showlegend=False,
                                 marker=dict(size=8, color=COLORS["text"], line=dict(color=COLORS["cyan"], width=2)),
                                 text=[str(i + 1) for i in range(len(nodes))] if len(nodes) <= 40 else None,
                                 textposition="bottom center", textfont=dict(color=COLORS["muted"], size=10),
                                 hovertemplate="node %{text} · x = %{x:.4g} m<extra></extra>"))
        if mesh.n_elements <= 40:
            mids = 0.5 * (nodes[mesh.elements[:, 0]] + nodes[mesh.elements[:, 1]])
            fig.add_trace(go.Scatter(x=mids, y=np.full_like(mids, 0.2), mode="text", showlegend=False,
                                     text=[f"e{i + 1}" for i in range(len(mids))], hoverinfo="skip",
                                     textfont=dict(color=COLORS["cyan"], size=10)))

    pad = 0.08 * L
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=20, b=36), showlegend=False,
                      xaxis=dict(range=[-pad, L + pad], title="x (m)" if show_dims else None, zeroline=False,
                                 showgrid=False),
                      yaxis=dict(range=[-1.0, 1.2], visible=False, fixedrange=True),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def section_figure(section, height: int = 230) -> go.Figure:
    """Cross-section sketch to scale with centroid / neutral axis. Cached per section."""
    return cached_figure(("section", repr(section), height), lambda: _section_figure(section, height))


def _section_figure(section, height: int) -> go.Figure:
    fig = go.Figure()
    cyan = COLORS["cyan"]
    if section.shape == "rect":
        b, h = section.b * 1e3, section.h * 1e3
        fig.add_shape(type="rect", x0=-b / 2, x1=b / 2, y0=-h / 2, y1=h / 2, fillcolor=rgba(cyan, 0.3),
                      line=dict(color=cyan, width=2))
        ext = max(b, h) / 2
    else:
        R = section.depth * 1e3 / 2
        fig.add_shape(type="circle", x0=-R, x1=R, y0=-R, y1=R, fillcolor=rgba(cyan, 0.3), line=dict(color=cyan, width=2))
        if section.shape == "hollow_circ":
            r = section.d * 1e3 / 2
            fig.add_shape(type="circle", x0=-r, x1=r, y0=-r, y1=r, fillcolor=COLORS["bg"], line=dict(color=cyan, width=2))
        ext = R
    e = ext * 1.35
    fig.add_shape(type="line", x0=-e, x1=e, y0=0, y1=0, line=dict(color=COLORS["amber"], width=1, dash="dash"))
    fig.add_annotation(x=e, y=0, text="N.A.", showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(color=COLORS["amber"], size=11))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis=dict(range=[-e, e], visible=False, scaleanchor="y"), yaxis=dict(range=[-e, e], visible=False),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def section_props_metrics(section, nu: float) -> None:
    p = sections.properties(section, nu)
    c = st.columns(2) + st.columns(2)  # 2×2: four columns are too narrow for values like "5.4e+05 mm⁴"
    c[0].metric("Area A", f"{p.A * 1e6:.4g} mm²")
    c[1].metric("Second moment I", f"{p.I * 1e12:.4g} mm⁴")
    c[2].metric("Section modulus Z", f"{p.Z * 1e9:.4g} mm³")
    c[3].metric("Radius of gyration r", f"{p.r_g * 1e3:.4g} mm")
