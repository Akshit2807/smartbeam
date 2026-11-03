"""Module 8 — Design comparison studio: saved designs, side-by-side metrics and parameter studies."""
import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from smartbeam import lab, sections
from smartbeam.materials import MATERIALS, get_material, material_keys
from smartbeam.metrics import evaluate
from smartbeam.model import BeamModel, ModelError, Section, UnstableStructureError, validate
from smartbeam.ui.design_tools import g, metrics_row
from smartbeam.ui.plots_core import model_caption, require_result, rgba
from smartbeam.ui.state import analyze_model, delete_design, get_model, save_design, saved_designs, set_model
from smartbeam.ui.theme import COLORS, SERIES, page_header

page_header("Design Comparison Studio",
            "Save design variants, compare them side by side with editable material rates, and see how each "
            "parameter drives stiffness, strength and weight. Every value comes from a full engine analysis.")

model = get_model()
st.markdown(model_caption(model))
require_result(model, record=False)


def metrics_of(m: BeamModel, rate=None):
    """DesignMetrics from the cached engine analysis, or None when the design cannot be analysed."""
    try:
        return evaluate(m, analyze_model(m, record=False), cost_per_kg=rate)
    except (ModelError, UnstableStructureError):
        return None


tab_cmp, tab_ovat, tab_sw = st.tabs(["Compare designs", "One-variable study", "Stiffness & weight studies"])

# ====================================================================== compare
with tab_cmp:
    saved = saved_designs()
    with st.container(border=True):
        c = st.columns([3, 1.3, 1.9], vertical_alignment="bottom")
        name = c[0].text_input("Design name", value=model.name or f"Design {len(saved) + 1}", max_chars=80)
        name = name.strip() or f"Design {len(saved) + 1}"
        if c[1].button("Save current design", type="primary", icon=":material/bookmark_add:", width="stretch"):
            save_design(name, model)
            st.toast(f"Saved '{name}'" + (" (replaced the previous version)" if name in saved else ""))
            st.rerun()
        if c[2].button("Save in every library material", icon=":material/layers:", width="stretch",
                       help="Saves one copy of the current design per library material, for a material comparison."):
            for k in material_keys():
                m = model.copy()
                m.material = get_material(k)
                m.name = f"{name} · {MATERIALS[k].name.split(' (')[0]}"
                save_design(m.name, m)
            st.rerun()

    with st.expander("Import / export saved designs (JSON)", icon=":material/swap_vert:"):
        e1, e2 = st.columns(2)
        e1.download_button("Download saved designs", json.dumps({n: m.to_dict() for n, m in saved.items()}, indent=2),
                           "smartbeam_designs.json", "application/json", disabled=not saved,
                           icon=":material/download:")
        up = e2.file_uploader("Load designs (.json)", type="json",
                              help="A file from 'Download saved designs', or a single model JSON (Playground / "
                                   "Design Studio schema).")
        if up is not None and st.session_state.get("comparison_import_id") != up.file_id:
            st.session_state["comparison_import_id"] = up.file_id
            added, problems = [], []
            try:
                data = json.loads(up.getvalue())
                items = {data.get("name") or up.name: data} if isinstance(data, dict) and "L" in data else data
                if not isinstance(items, dict):
                    raise ValueError("expected an object of {name: model}")
                for n, d in items.items():
                    try:
                        m = BeamModel.from_dict(d)
                        errs = validate(m)
                        if errs:
                            raise ValueError("; ".join(errs))
                        save_design(str(n), m)
                        added.append(str(n))
                    except (KeyError, TypeError, ValueError) as exc:
                        problems.append(f"{n}: {exc}")
            except (ValueError, json.JSONDecodeError) as exc:
                problems.append(f"Not a SmartBeam design file ({exc}).")
            st.session_state["comparison_import_msg"] = (added, problems)
            st.rerun()
        added, problems = st.session_state.pop("comparison_import_msg", ([], []))
        if added:
            st.success(f"Imported {len(added)} design(s): {', '.join(added)}")
        for p in problems:
            st.error(p)

    if not saved:
        st.info("No saved designs yet. Save the current design, change it in Design Studio or the Playground, and "
                "save again to compare. Or save it in every library material at once.", icon=":material/info:")
    else:
        # ---- editable cost rates (₹/kg), one per material in the saved set
        rates = st.session_state.setdefault("comparison_rates", {})
        mats = {}
        for m in saved.values():
            mats.setdefault(m.material.name, m.material.cost_per_kg)
        rate_df = pd.DataFrame({"Material": list(mats), "₹ per kg": [rates.get(n, r) for n, r in mats.items()]})
        st.markdown("##### Material cost rates")
        st.caption("Approximate Indian market estimates. Edit them to your supplier's prices; costs update instantly.")
        edited = st.data_editor(rate_df, hide_index=True, disabled=["Material"], width="stretch",
                                key="comparison_rates_" + "|".join(mats),
                                column_config={"₹ per kg": st.column_config.NumberColumn(min_value=0.0, step=1.0,
                                                                                         format="%.1f")})
        for n, r in zip(edited["Material"], edited["₹ per kg"]):
            rates[n] = None if r is None or (isinstance(r, float) and math.isnan(r)) else float(r)

        mets, rows, failed = {}, [], []
        for n, m in saved.items():
            met = metrics_of(m, rates.get(m.material.name))
            if met is None:
                failed.append(n)
                continue
            mets[n] = met
            rows.append(metrics_row(met, Design=n))
        if failed:
            st.warning(f"Not analysable (invalid or unstable): {', '.join(failed)}.")
        st.markdown("##### Comparison table")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        mc = st.columns([3, 1, 1], vertical_alignment="bottom")
        pick = mc[0].selectbox("Manage a saved design", list(saved))
        if mc[1].button("Load into editor", icon=":material/upload:", width="stretch",
                        help="Makes it the current design for every page."):
            set_model(saved[pick])
            st.rerun()
        if mc[2].button("Delete", icon=":material/delete:", width="stretch"):
            delete_design(pick)
            st.rerun()

        if mets:
            st.markdown("##### Charts")
            METRICS = {"Max deflection |v| (mm)": ("delta_max", 1e3, "min"),
                       "Max stress |σ| (MPa)": ("sigma_max", 1e-6, "min"), "Yield FOS": ("fos", 1.0, "max"),
                       "Max moment |M| (N·m)": ("M_max", 1.0, "min"), "Mass (kg)": ("mass", 1.0, "min"),
                       "Cost (₹)": ("cost", 1.0, "min")}
            left, right = st.columns(2)
            with left:
                label = st.selectbox("Bar chart metric", list(METRICS))
                attr, scale, better = METRICS[label]
                names = [n for n in mets if getattr(mets[n], attr) is not None]
                vals = [getattr(mets[n], attr) * scale for n in names]
                if vals:
                    best = (min if better == "min" else max)(vals)
                    fig = go.Figure(go.Bar(x=names, y=vals, text=[f"{v:.4g}" for v in vals], textposition="outside",
                                           marker_color=[COLORS["green"] if v == best else COLORS["cyan"] for v in vals],
                                           hovertemplate="%{x}: %{y:.5g}<extra></extra>"))
                    fig.update_layout(height=380, yaxis_title=label, title=f"{label} — best in green "
                                                                           f"({'lowest' if better == 'min' else 'highest'})")
                    st.plotly_chart(fig, theme=None, width="stretch")
                else:
                    st.caption("No values (cost needs a ₹/kg rate).")
            with right:
                axes = {"Deflection": lambda m: m.delta_max, "Stress": lambda m: m.sigma_max,
                        "Utilisation σ/σy": lambda m: 1 / m.fos if m.fos > 0 else math.nan,
                        "Moment": lambda m: m.M_max, "Mass": lambda m: m.mass, "Area": lambda m: m.A}
                if all(m.cost is not None for m in mets.values()):
                    axes["Cost"] = lambda m: m.cost
                peak = {k: max(f(m) for m in mets.values()) or 1.0 for k, f in axes.items()}
                fig = go.Figure()
                for i, (n, m) in enumerate(mets.items()):
                    r = [f(m) / peak[k] for k, f in axes.items()]
                    color = SERIES[i % len(SERIES)]
                    fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=list(axes) + list(axes)[:1], name=n, fill="toself",
                                                  line=dict(color=color, width=2), fillcolor=rgba(color, 0.12),
                                                  hovertemplate="%{theta}: %{r:.3f} of max<extra>" + n + "</extra>"))
                fig.update_layout(height=430, title="Normalised profile (smaller = better)",
                                  polar=dict(bgcolor=COLORS["panel"], radialaxis=dict(range=[0, 1.05],
                                             gridcolor=COLORS["grid"], tickfont=dict(color=COLORS["muted"])),
                                             angularaxis=dict(gridcolor=COLORS["grid"])),
                                  legend=dict(orientation="h", y=-0.12))
                st.plotly_chart(fig, theme=None, width="stretch")
                st.caption("Each axis is divided by the largest value among the saved designs, so 1.0 = worst on that "
                           "axis. Utilisation σ/σy = 1/FOS.")

# ====================================================================== one variable at a time
with tab_ovat:
    st.markdown("Change **one** parameter of the current design over a range while everything else stays fixed. "
                "Each point is a complete engine analysis.")
    dims = model.section.dims()
    VARS = {"L": ("Span L", "m", 1.0, model.L)}
    VARS |= {k: (f"Section dimension {k}", "mm", 1e-3, v) for k, v in dims.items()}
    VARS |= {"E": ("Young's modulus E", "GPa", 1e9, model.material.E), "load": ("Load multiplier", "×", 1.0, 1.0)}
    c = st.columns([2, 1, 1, 1.4])
    var = c[0].selectbox("Variable", list(VARS), format_func=lambda k: f"{VARS[k][0]} ({VARS[k][1]})")
    label, unit, scale, cur_si = VARS[var]
    cur = cur_si / scale
    lo = c[1].number_input(f"From ({unit})", min_value=1e-6, value=float(f"{cur * 0.5:.4g}"), format="%.4g",
                           key=f"ovat_lo_{var}")
    hi = c[2].number_input(f"To ({unit})", min_value=1e-6, value=float(f"{cur * 2:.4g}"), format="%.4g",
                           key=f"ovat_hi_{var}")
    n_pts = c[3].slider("Points", 5, 40, 15)
    if var == "L":
        st.caption("Supports and loads keep their relative positions x/L as the span changes.")
    elif var == "load":
        st.caption("Every load magnitude (forces, distributed intensities, moments) is multiplied by this factor.")
    if hi <= lo:
        st.error("'To' must be greater than 'From'.")
    else:
        xs = np.linspace(lo, hi, n_pts)
        rows = []
        for xv in xs:
            met = metrics_of(lab.scale_parameter(model, var, xv * scale / cur_si))
            rows.append((xv, *(np.nan,) * 4) if met is None else
                        (xv, met.delta_max * 1e3, met.sigma_max / 1e6, met.fos, met.mass))
        df = pd.DataFrame(rows, columns=["x", "δmax (mm)", "σmax (MPa)", "FOS", "Mass (kg)"])
        if df["δmax (mm)"].isna().any():
            st.warning(f"{int(df['δmax (mm)'].isna().sum())} point(s) are not valid designs (e.g. inner diameter ≥ "
                       "outer, or a support outside the span) and are left out.")
        base = metrics_of(model)
        fig = make_subplots(rows=2, cols=2, subplot_titles=list(df.columns[1:]), horizontal_spacing=0.1,
                            vertical_spacing=0.16)
        base_vals = [base.delta_max * 1e3, base.sigma_max / 1e6, base.fos, base.mass]
        for i, (col, color, bv) in enumerate(zip(df.columns[1:], (COLORS["cyan"], COLORS["amber"], COLORS["green"],
                                                                 COLORS["violet"]), base_vals)):
            r, cc = i // 2 + 1, i % 2 + 1
            fig.add_trace(go.Scatter(x=df["x"], y=df[col], mode="lines+markers", name=col, line=dict(color=color, width=2.4),
                                     marker=dict(size=5), hovertemplate=f"{label} = %{{x:.4g}} {unit}<br>{col} = %{{y:.4g}}"
                                                                        "<extra></extra>"), row=r, col=cc)
            fig.add_trace(go.Scatter(x=[cur], y=[bv], mode="markers", name="current design", showlegend=i == 0,
                                     marker=dict(size=11, color=COLORS["text"], symbol="star"),
                                     hovertemplate=f"current: %{{y:.4g}}<extra></extra>"), row=r, col=cc)
            fig.update_xaxes(title_text=f"{label} ({unit})", row=r, col=cc)
        fig.update_layout(height=620, legend=dict(orientation="h", y=1.08), margin=dict(t=70))
        st.plotly_chart(fig, theme=None, width="stretch")
        k_d = lab.loglog_slope(df["x"], df["δmax (mm)"])
        k_s = lab.loglog_slope(df["x"], df["σmax (MPa)"])
        k_m = lab.loglog_slope(df["x"], df["Mass (kg)"])
        m1, m2, m3 = st.columns(3)
        m1.metric("δmax ∝ xᵏ, fitted k", f"{k_d:.3f}" if math.isfinite(k_d) else "—")
        m2.metric("σmax ∝ xᵏ, fitted k", f"{k_s:.3f}" if math.isfinite(k_s) else "—")
        m3.metric("mass ∝ xᵏ, fitted k", f"{k_m:.3f}" if math.isfinite(k_m) else "—")
        st.caption("Exponents are least-squares slopes on log–log axes over the swept range. Compare with beam "
                   "theory: δ ∝ L³ (point load) or L⁴ (UDL), δ ∝ 1/E, δ ∝ h⁻³ and σ ∝ h⁻² for a rectangle.")
        with st.expander("Study data"):
            st.dataframe(df.rename(columns={"x": f"{label} ({unit})"}).map(g), hide_index=True, width="stretch")

# ====================================================================== stiffness & weight
with tab_sw:
    F = lab.total_load(model)
    s = model.section
    st.markdown("#### Thickness vs stiffness")
    if F <= 0:
        st.info("Stiffness k = Σ|load| / δmax needs at least one transverse force or distributed load in the "
                "current design.")
    else:
        st.latex(r"k = \frac{\sum |P|}{|v|_{\max}} \;\propto\; EI,\qquad I_{rect} = \frac{b h^3}{12},\quad "
                 r"I_{circ} = \frac{\pi d^4}{64},\quad I_{tube} = \frac{\pi (D^4 - d^4)}{64}")
        fig = go.Figure()
        slopes = []
        if s.shape == "hollow_circ":  # wall thickness t at fixed outer diameter D
            t_vals = np.linspace(0.02, 0.5, 20) * s.D
            ks, ms = [], []
            for t in t_vals:
                m = model.copy()
                m.section = Section.hollow_circ(s.D, max(s.D - 2 * t, 0.0)) if t < s.D / 2 else Section.solid_circ(s.D)
                met = metrics_of(m)
                ks.append(F / met.delta_max / 1e3 if met else np.nan)
                ms.append(met.mass if met else np.nan)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=t_vals * 1e3, y=ks, name="stiffness k", line=dict(color=COLORS["cyan"], width=2.6),
                                     hovertemplate="t = %{x:.3g} mm<br>k = %{y:.4g} kN/m<extra></extra>"))
            fig.add_trace(go.Scatter(x=t_vals * 1e3, y=ms, name="mass", line=dict(color=COLORS["amber"], width=2, dash="dash"),
                                     hovertemplate="t = %{x:.3g} mm<br>mass = %{y:.4g} kg<extra></extra>"),
                          secondary_y=True)
            t0 = (s.D - s.d) / 2
            fig.add_vline(x=t0 * 1e3, line=dict(color=COLORS["muted"], dash="dot"), annotation_text="current wall")
            fig.update_layout(height=420, xaxis_title=f"Wall thickness t (mm), outer D = {s.D * 1e3:.4g} mm fixed",
                              legend=dict(orientation="h", y=1.1))
            fig.update_yaxes(title_text="k (kN/m)", secondary_y=False)
            fig.update_yaxes(title_text="mass (kg)", secondary_y=True, showgrid=False)
            st.plotly_chart(fig, theme=None, width="stretch")
            st.caption("Stiffness saturates as the wall thickens because material near the neutral axis adds little "
                       "to I, while mass keeps growing. Thin walls give the best stiffness per kilogram, until local "
                       "wall buckling (not modelled) takes over.")
        else:
            factors = np.geomspace(0.5, 2.0, 15)
            params = [("h", "depth h"), ("b", "width b")] if s.shape == "rect" else [("d", "diameter d")]
            for (p, lab_txt), color in zip(params, (COLORS["cyan"], COLORS["amber"])):
                ks = []
                for f in factors:
                    met = metrics_of(lab.scale_parameter(model, p, f))
                    ks.append(F / met.delta_max / 1e3 if met else np.nan)
                slopes.append((lab_txt, lab.loglog_slope(factors, ks)))
                fig.add_trace(go.Scatter(x=factors, y=ks, mode="lines+markers", name=f"vary {lab_txt}",
                                         line=dict(color=color, width=2.6), marker=dict(size=5),
                                         hovertemplate="factor %{x:.3g}<br>k = %{y:.4g} kN/m<extra></extra>"))
            fig.update_layout(height=420, xaxis_type="log", yaxis_type="log", legend=dict(orientation="h", y=1.1),
                              xaxis_title="dimension ÷ current value", yaxis_title="stiffness k (kN/m)")
            st.plotly_chart(fig, theme=None, width="stretch")
            st.caption(" · ".join(f"k ∝ ({t})^{k:.2f}" for t, k in slopes) +
                       ". Fitted from the engine results on log–log axes. Depth acts cubically and width linearly, "
                       "which is why beams are deep rather than wide.")

    st.markdown("#### Geometry vs weight")
    st.markdown("Three section families sized for the **same range of second moment I** (around the current "
                "design's), each analysed with the current span, supports, loads and material. The chart shows how "
                "much mass each shape needs for a given deflection.")
    I0 = sections.second_moment(s)
    fams = {"Rectangle h = 2b": lambda I: Section.rect((1.5 * I) ** 0.25, 2 * (1.5 * I) ** 0.25),
            "Solid circle": lambda I: Section.solid_circ((64 * I / math.pi) ** 0.25),
            "Tube d/D = 0.8": lambda I: Section.hollow_circ((64 * I / (math.pi * (1 - 0.8 ** 4))) ** 0.25,
                                                            0.8 * (64 * I / (math.pi * (1 - 0.8 ** 4))) ** 0.25)}
    fig = go.Figure()
    for (fam, make), color in zip(fams.items(), (COLORS["cyan"], COLORS["amber"], COLORS["violet"])):
        pts = []
        for I in I0 * np.geomspace(0.25, 4.0, 13):
            m = model.copy()
            m.section = make(I)
            met = metrics_of(m)
            if met:
                pts.append((met.delta_max * 1e3, met.mass, ", ".join(f"{k} {v * 1e3:.3g}" for k, v in m.section.dims().items())))
        if pts:
            d, ms, txt = zip(*pts)
            fig.add_trace(go.Scatter(x=d, y=ms, mode="lines+markers", name=fam, text=txt,
                                     line=dict(color=color, width=2.6), marker=dict(size=5),
                                     hovertemplate="%{text} mm<br>δmax = %{x:.4g} mm<br>mass = %{y:.4g} kg"
                                                   "<extra>" + fam + "</extra>"))
    base = metrics_of(model)
    fig.add_trace(go.Scatter(x=[base.delta_max * 1e3], y=[base.mass], mode="markers", name="current design",
                             marker=dict(size=14, symbol="star", color=COLORS["text"]),
                             hovertemplate="current<br>δmax = %{x:.4g} mm<br>mass = %{y:.4g} kg<extra></extra>"))
    fig.update_layout(height=440, xaxis_type="log", yaxis_type="log", xaxis_title="max deflection |v| (mm)",
                      yaxis_title="mass (kg)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, theme=None, width="stretch")
    st.caption("Lower curve = lighter for the same stiffness. The tube wins because its area sits far from the neutral "
               "axis. Section dimensions come from the I formulas above; deflection and mass come from the engine.")
