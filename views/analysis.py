"""Classical analysis: reactions, SFD/BMD, stress & strain (incl. through-depth), slope & deflection, checks."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from smartbeam import sections
from smartbeam.analytical import stress_distribution
from smartbeam.model import classify
from smartbeam.ui.plots_core import (METHOD_LABEL, beam_schematic, extreme_marker, field_figure, field_trace,
                                     label_range, model_caption, require_result, show_warnings)
from smartbeam.ui.state import get_model
from smartbeam.ui.theme import COLORS, badge, page_header

C = COLORS
page_header("Analysis", "Reactions, internal forces, stresses, strains, slope and deflection of the current design.")
model = get_model()
result = require_result(model)
s, mat = model.section, model.material
st.markdown(f"{model_caption(model)} &nbsp; {badge(result.method.upper())}", unsafe_allow_html=True)
st.caption(f"Solver: **{METHOD_LABEL[result.method]}** · {result.stability.message}")
show_warnings(result)

m = st.columns(5)
m[0].metric("Max |V|", f"{abs(result.V_max.value) / 1e3:.4g} kN", help=f"at x = {result.V_max.x:.4g} m")
m[1].metric("Max |M|", f"{abs(result.M_max.value) / 1e3:.4g} kN·m", help=f"at x = {result.M_max.x:.4g} m")
m[2].metric("Max |σ|", f"{abs(result.sigma_max.value) / 1e6:.4g} MPa",
            help=f"{result.sigma_max.where} fibre at x = {result.sigma_max.x:.4g} m")
m[3].metric("Max |θ|", f"{abs(result.theta_max.value) * 1e3:.4g} mrad", help=f"at x = {result.theta_max.x:.4g} m")
m[4].metric("Max |v|", f"{abs(result.v_max.value) * 1e3:.4g} mm", help=f"at x = {result.v_max.x:.4g} m")

tabs = st.tabs(["Reactions & equilibrium", "Shear & moment", "Stress & strain", "Slope & deflection",
                "Textbook check", "Data"])

# --------------------------------------------------------------------------- reactions & equilibrium
with tabs[0]:
    st.plotly_chart(beam_schematic(model, height=220), theme=None, key="an_schem")
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        st.markdown("##### Support reactions (acting on the beam)")
        st.dataframe(pd.DataFrame([{
            "Support": r.support_index + 1, "Type": r.type, "x (m)": r.x, "Rx (kN)": r.Fx / 1e3,
            "Ry (kN)": r.Fy / 1e3, "M (kN·m, CCW +)": r.M / 1e3} for r in result.reactions]),
            hide_index=True, column_config={k: st.column_config.NumberColumn(format="%.5g")
                                            for k in ("x (m)", "Rx (kN)", "Ry (kN)", "M (kN·m, CCW +)")})
        st.caption("Rx, Ry positive in +x / +y (up); reaction moment positive counter-clockwise.")
        if result.method != "analytical":
            st.info("Reactions come from the stiffness (FEM) solution: " + result.stability.message)
    with b:
        st.markdown("##### Equilibrium checks")
        scale = sum(abs(p.magnitude) for p in model.point_loads) + sum(abs(w.resultant) for w in model.distributed_loads)
        mscale = scale * model.L + sum(abs(mo.M) for mo in model.moment_loads)
        rows = []
        for key, label, sc, unit in [("sum_Fx", "ΣFx", scale, "N"), ("sum_Fy", "ΣFy", scale, "N"),
                                     ("sum_M0", "ΣM about x = 0", mscale, "N·m")]:
            val = result.equilibrium[key]
            rel = abs(val) / sc if sc > 0 else 0.0
            rows.append({"Check": label, "Residual": f"{val:.3e} {unit}", "Relative": f"{rel:.1e}",
                         "Status": "✓ satisfied" if rel < 1e-6 else "✗ check"})
        for key, label in [("dM_dx_minus_V", "dM/dx − V"), ("dV_dx_minus_q", "dV/dx − q")]:
            if key in result.equilibrium:
                rel = result.equilibrium[key]
                rows.append({"Check": label, "Residual": "—", "Relative": f"{rel:.1e}",
                             "Status": "✓ satisfied" if rel < 1e-3 else "✗ check"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.latex(r"\sum F_x = 0,\quad \sum F_y = 0,\quad \sum M_{x=0} = 0,\qquad \frac{dV}{dx} = q(x),\quad "
                 r"\frac{dM}{dx} = V")
        st.caption("Residuals are the sums of all applied loads and reactions; relative values are divided by the "
                   "total applied load (or load × L). Differential checks are evaluated numerically on the sampled "
                   "diagrams (analytical route only).")

# --------------------------------------------------------------------------- SFD / BMD
with tabs[1]:
    axial = np.any(np.abs(result.N) > 0)
    rows = 3 if axial else 2
    titles = ["Shear force V (kN)", "Bending moment M (kN·m) · sagging +"] + (["Axial force N (kN) · tension +"]
                                                                              if axial else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=titles)
    for i, (arr, ext, name, unit, col) in enumerate([
            (result.V, result.V_max, "V", "kN", C["cyan"]), (result.M, result.M_max, "M", "kN·m", C["amber"]),
            (result.N, result.N_max, "N", "kN", C["violet"])][:rows], start=1):
        fig.add_trace(field_trace(result.x, arr / 1e3, name, col, unit), row=i, col=1)
        fig.add_trace(extreme_marker(ext, 1e-3, unit, col, "max", L=model.L), row=i, col=1)
        fig.add_hline(y=0, line=dict(color=C["border"], width=1), row=i, col=1)
        fig.update_yaxes(range=label_range(arr / 1e3, ext.value), row=i, col=1)
    fig.update_xaxes(title_text="x (m)", row=rows, col=1)
    fig.update_layout(height=260 * rows, showlegend=False, margin=dict(t=40))
    st.plotly_chart(fig, theme=None, key="an_vm")
    with st.expander("Sign convention and governing relations"):
        st.markdown("- **V(x)** = sum of vertical forces to the LEFT of the cut (up +).\n"
                    "- **M(x)** sagging positive (bottom fibre in tension); a counter-clockwise applied moment makes "
                    "M jump down.\n- **N(x)** tension positive. Vertical jumps occur at concentrated forces and "
                    "moments.")
        st.latex(r"V(x)=\sum_{\text{left}} F_y,\qquad M(x)=\sum_{\text{left}} F_y\,(x-a)-\sum_{\text{left}} C,"
                 r"\qquad \frac{dV}{dx}=q(x)=-w(x),\qquad \frac{dM}{dx}=V")

# --------------------------------------------------------------------------- stress & strain
with tabs[2]:
    fig = go.Figure([field_trace(result.x, result.sigma_top / 1e6, "σ top", C["red"], "MPa", fill=False),
                     field_trace(result.x, result.sigma_bottom / 1e6, "σ bottom", C["blue"], "MPa", fill=False)])
    fig.add_trace(extreme_marker(result.sigma_max, 1e-6, "MPa", C["amber"], "|σ|max", L=model.L))
    fig.add_hline(y=0, line=dict(color=C["border"]))
    for sgn in (1, -1):
        fig.add_hline(y=sgn * mat.yield_strength / 1e6, line=dict(color=C["red"], dash="dot", width=1),
                      annotation_text=f"{'+' if sgn > 0 else '−'}σy", annotation_font_color=C["red"])
    sig_span = max(np.abs(result.sigma_top).max(), np.abs(result.sigma_bottom).max()) / 1e6
    fig.update_layout(height=340, title="Extreme-fibre normal stress σ = N/A − M·y/I (tension +)",
                      xaxis_title="x (m)", yaxis_title="σ (MPa)", legend=dict(orientation="h", y=1.12, x=1,
                                                                             xanchor="right"),
                      yaxis_range=[-1.25 * max(sig_span, 1e-9), 1.25 * max(sig_span, 1e-9)])
    st.plotly_chart(fig, theme=None, key="an_sig")
    t, c_ = result.sigma_tension_max, result.sigma_compression_max
    k = st.columns(3)
    k[0].metric("Max tensile σ", f"{t.value / 1e6:.4g} MPa", help=f"{t.where} fibre at x = {t.x:.4g} m")
    k[1].metric("Max compressive σ", f"{c_.value / 1e6:.4g} MPa", help=f"{c_.where} fibre at x = {c_.x:.4g} m")
    k[2].metric("Max |ε|", f"{abs(result.sigma_max.value) / mat.E * 1e6:.4g} µε")

    st.markdown("##### Distribution through the depth")
    xq = st.slider("Section position x (m)", 0.0, float(model.L), float(result.sigma_max.x),
                   step=float(model.L) / 500, format="%.3f")
    i = int(np.argmin(np.abs(result.x - xq)))
    Mx, Nx, Vx = float(result.M[i]), float(result.N[i]), float(result.V[i])
    y, sig = stress_distribution(model, Mx, Nx, n=81)
    tau = sections.shear_stress(s, Vx, y)
    st.caption(f"At x = {result.x[i]:.4g} m: M = {Mx / 1e3:.4g} kN·m, V = {Vx / 1e3:.4g} kN, N = {Nx / 1e3:.4g} kN. "
               "At a jump the value just left of the point is used.")
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=["Normal stress σ (MPa)", "Strain ε (µε)", "Shear stress τ = VQ/(Ib) (MPa)"])
    for col, (vals, color, name) in enumerate([(sig / 1e6, C["amber"], "σ"), (sig / mat.E * 1e6, C["violet"], "ε"),
                                               (tau / 1e6, C["cyan"], "τ")], start=1):
        fig.add_trace(go.Scatter(x=vals, y=y * 1e3, mode="lines", line=dict(color=color, width=2.4), fill="tozerox",
                                 fillcolor=f"rgba(255,255,255,0.04)", name=name,
                                 hovertemplate=f"y = %{{y:.3g}} mm<br>{name} = %{{x:.4g}}<extra></extra>"),
                      row=1, col=col)
        fig.add_vline(x=0, line=dict(color=C["border"]), row=1, col=col)
    fig.add_hline(y=0, line=dict(color=C["muted"], dash="dash", width=1), annotation_text="neutral axis",
                  annotation_font_color=C["muted"])
    fig.update_yaxes(title_text="y from centroid (mm)", row=1, col=1)
    fig.update_layout(height=360, showlegend=False, margin=dict(t=40))
    st.plotly_chart(fig, theme=None, key="an_depth")
    st.latex(r"\sigma(x,y)=\frac{N}{A}-\frac{M\,y}{I},\qquad \varepsilon=\frac{\sigma}{E},\qquad "
             r"\tau(y)=\frac{V\,Q(y)}{I\,b(y)}")
    st.caption("Plane sections remain plane (Euler–Bernoulli); stress is linear in y. For circular sections τ is the "
               "average across the chord width b(y).")

    fig = go.Figure([field_trace(result.x, result.eps_top * 1e6, "ε top", C["red"], "µε", fill=False),
                     field_trace(result.x, result.eps_bottom * 1e6, "ε bottom", C["blue"], "µε", fill=False)])
    fig.update_layout(height=280, title="Extreme-fibre strain ε = σ/E", xaxis_title="x (m)", yaxis_title="ε (µε)",
                      legend=dict(orientation="h", y=1.15, x=1, xanchor="right"))
    st.plotly_chart(fig, theme=None, key="an_eps")

# --------------------------------------------------------------------------- slope & deflection
with tabs[3]:
    a, b = st.columns(2)
    a.plotly_chart(field_figure(result.x, result.theta, result.theta_max, "θ", "mrad", 1e3, C["violet"],
                                title="Slope θ = dv/dx (CCW +)"), theme=None, key="an_th")
    b.plotly_chart(field_figure(result.x, result.v, result.v_max, "v", "mm", 1e3, C["cyan"],
                                title="Deflection v (up +)"), theme=None, key="an_v")
    vmax = abs(result.v_max.value)
    st.caption(f"Span / max deflection: L/|v|max = {model.L / vmax:,.0f}" if vmax > 0 else "No deflection.")
    st.latex(r"EI\,\frac{d^2v}{dx^2}=M(x),\qquad \theta=\frac{dv}{dx}=\int\frac{M}{EI}\,dx + C_1,\qquad "
             r"v=\int\theta\,dx + C_2")
    st.caption("Analytical route: M/EI integrated twice numerically (trapezoid with end correction); C₁, C₂ from the "
               "support conditions. FEM routes: Hermite-cubic reconstruction between nodes.")

# --------------------------------------------------------------------------- textbook comparison
with tabs[4]:
    tb = result.textbook
    if tb is None:
        st.info(f"No textbook comparison for this configuration (classified **{classify(model)}**, "
                f"{len(model.loads)} load(s)). Closed-form formulas are applied only to: a cantilever with a single "
                "vertical end load, a cantilever with a full-length UDL, a simply supported beam with a central "
                "point load, and a simply supported beam with a full-length UDL. Any other case is solved "
                "numerically and no formula is forced onto it.")
    else:
        st.markdown(f"Configuration recognised as **{tb.case}**; closed-form values vs the engine:")
        names = {"R": "Max reaction R", "V_max": "Max |V|", "M_max": "Max |M|", "theta_max": "Max |θ|",
                 "delta_max": "Max |v|", "sigma_max": "Max |σ|"}
        unit = {"R": ("kN", 1e-3), "V_max": ("kN", 1e-3), "M_max": ("kN·m", 1e-3), "theta_max": ("mrad", 1e3),
                "delta_max": ("mm", 1e3), "sigma_max": ("MPa", 1e-6)}
        st.dataframe(pd.DataFrame([{
            "Quantity": names[k], "Formula": tb.formulas[k], "Unit": unit[k][0],
            "Textbook": tb.textbook[k] * unit[k][1], "SmartBeam": tb.numerical.get(k, np.nan) * unit[k][1],
            "Rel. error (%)": tb.rel_error.get(k, np.nan) * 100} for k in tb.textbook]), hide_index=True,
            column_config={"Textbook": st.column_config.NumberColumn(format="%.6g"),
                           "SmartBeam": st.column_config.NumberColumn(format="%.6g"),
                           "Rel. error (%)": st.column_config.NumberColumn(format="%.2e")})
        st.caption("Relative errors at round-off level confirm the engine reproduces the closed-form solutions.")

with tabs[5]:
    df = pd.DataFrame({"x (m)": result.x, "V (N)": result.V, "M (N·m)": result.M, "N (N)": result.N,
                       "theta (rad)": result.theta, "v (m)": result.v, "sigma_top (Pa)": result.sigma_top,
                       "sigma_bottom (Pa)": result.sigma_bottom, "eps_top": result.eps_top,
                       "eps_bottom": result.eps_bottom})
    st.dataframe(df, hide_index=True, height=360)
    st.download_button("Download sampled results (CSV, SI units)", df.to_csv(index=False),
                       file_name="smartbeam_analysis.csv", mime="text/csv", icon=":material/download:")
