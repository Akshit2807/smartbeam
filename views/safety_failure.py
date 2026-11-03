"""Safety & failure: yield FOS + first-yield load, von Mises, Euler buckling, stress-life fatigue (user data only)."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smartbeam import safety
from smartbeam.model import DistributedLoad, PointLoad
from smartbeam.ui.plots_core import STATUS_COLOR, model_caption, require_result, status_badge
from smartbeam.ui.state import get_model
from smartbeam.ui.theme import COLORS, page_header

C = COLORS
page_header("Safety & Failure", "Onset of yielding, combined stress, elastic buckling and fatigue life. "
            "Yield is not fracture: ultimate failure is not predicted.")
model = get_model()
result = require_result(model)
mat = model.material
st.markdown(model_caption(model))

top_l, top_r = st.columns([1, 3], vertical_alignment="center")
min_fos = top_l.number_input("Required minimum FOS", min_value=1.0, max_value=20.0, value=1.5, step=0.1,
                             format="%.2f", key="sf_minfos")
rep = safety.assess(model, result, min_fos)
yr, vm, bk = rep.yield_result, rep.combined, rep.buckling
top_r.markdown(f"<div style='display:flex;gap:14px;align-items:center'>Overall status {status_badge(rep.status)}"
               f"<span style='color:{C['muted']}'>worst of yield, von Mises and (when compression exists) buckling, "
               f"against FOS ≥ {min_fos:g}</span></div>", unsafe_allow_html=True)


def fos_txt(v):
    return "∞" if v is None or math.isinf(v) else f"{v:.3g}"


k = st.columns(5)
k[0].metric("Max |σ|", f"{abs(yr.sigma_max.value) / 1e6:.4g} MPa", help=f"x = {yr.sigma_max.x:.4g} m")
k[1].metric("Yield strength σy", f"{yr.sigma_y / 1e6:.4g} MPa", help="Approximate library value" if mat.approximate
            else "User-defined")
k[2].metric("Yield FOS", fos_txt(yr.fos))
k[3].metric("von Mises FOS", fos_txt(vm.fos))
k[4].metric("Buckling FOS", fos_txt(bk.fos) if bk.applicable else "n/a",
            help="Only when the member carries axial compression")

tabs = st.tabs(["Yield & first-yield load", "Combined stress (von Mises)", "Euler buckling", "Fatigue (stress-life)",
                "Summary"])

# --------------------------------------------------------------------------- yield
with tabs[0]:
    st.markdown(f"{status_badge(yr.status)} &nbsp; {yr.message}", unsafe_allow_html=True)
    util = np.maximum(np.abs(result.sigma_top), np.abs(result.sigma_bottom)) / yr.sigma_y
    fig = go.Figure(go.Scatter(x=result.x, y=util, mode="lines", line=dict(color=C["cyan"], width=2.4), fill="tozeroy",
                               fillcolor="rgba(136,192,208,0.12)",
                               hovertemplate="x = %{x:.4g} m<br>|σ|/σy = %{y:.4g}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=C["red"], dash="dash"), annotation_text="first yield (|σ| = σy)",
                  annotation_font_color=C["red"])
    fig.add_hline(y=1 / min_fos, line=dict(color=C["amber"], dash="dot"), annotation_text=f"allowable (1/{min_fos:g})",
                  annotation_font_color=C["amber"], annotation_position="bottom right")
    fig.update_layout(height=320, title="Utilisation |σ|max(x) / σy along the beam", xaxis_title="x (m)",
                      yaxis_title="utilisation", yaxis_range=[0, max(1.1, float(util.max()) * 1.15)])
    st.plotly_chart(fig, theme=None, key="sf_util")
    a, b = st.columns([1, 1.3], gap="large")
    with a:
        st.latex(r"\text{FOS}=\frac{\sigma_y}{|\sigma|_{\max}},\qquad \lambda_{\text{first yield}}=\text{FOS}")
        st.caption("Linear elasticity: stresses scale with the loads, so multiplying every load by λ brings the most "
                   "stressed fibre exactly to σy. Beyond that, plasticity spreads; the collapse load of a ductile "
                   "beam is higher, but it is not computed here.")
    with b:
        lam = yr.first_yield_load_factor
        if math.isfinite(lam):
            rows = []
            for i, ld in enumerate(model.loads, 1):
                if isinstance(ld, PointLoad):
                    rows.append({"Load": f"{i}: point at {ld.x:.4g} m", "Applied": f"{ld.magnitude / 1e3:.4g} kN",
                                 "At first yield": f"{lam * ld.magnitude / 1e3:.4g} kN"})
                elif isinstance(ld, DistributedLoad):
                    rows.append({"Load": f"{i}: distributed {ld.x1:.4g}–{ld.x2:.4g} m",
                                 "Applied": f"{ld.w1 / 1e3:.4g} → {ld.w2 / 1e3:.4g} kN/m",
                                 "At first yield": f"{lam * ld.w1 / 1e3:.4g} → {lam * ld.w2 / 1e3:.4g} kN/m"})
                else:
                    rows.append({"Load": f"{i}: moment at {ld.x:.4g} m", "Applied": f"{abs(ld.M) / 1e3:.4g} kN·m",
                                 "At first yield": f"{lam * abs(ld.M) / 1e3:.4g} kN·m"})
            st.markdown(f"##### First-yield loads (λ = {lam:.4g})")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.info("The beam is unstressed, so no finite first-yield load exists.")

# --------------------------------------------------------------------------- von Mises
with tabs[1]:
    a, b = st.columns([1, 1.2], gap="large")
    with a:
        q = st.columns(2)
        q[0].metric("σ at critical point", f"{vm.sigma / 1e6:.4g} MPa")
        q[1].metric("τ at critical point", f"{vm.tau / 1e6:.4g} MPa")
        q[0].metric("von Mises σvm", f"{vm.von_mises / 1e6:.4g} MPa")
        q[1].metric("FOS (σy/σvm)", fos_txt(vm.fos))
        st.caption(f"Critical point: x = {vm.x:.4g} m, y = {vm.y * 1e3:.3g} mm from the centroid.")
        st.latex(r"\sigma=\frac{N}{A}-\frac{My}{I},\quad \tau=\frac{VQ(y)}{I\,b(y)},\quad "
                 r"\sigma_{vm}=\sqrt{\sigma^2+3\tau^2}")
    with b:
        fig = go.Figure(go.Scatter(x=vm.vm_profile / 1e6, y=vm.y_profile * 1e3, mode="lines",
                                   line=dict(color=C["violet"], width=2.6), fill="tozerox",
                                   fillcolor="rgba(180,142,173,0.12)",
                                   hovertemplate="y = %{y:.3g} mm<br>σvm = %{x:.4g} MPa<extra></extra>"))
        fig.add_vline(x=yr.sigma_y / 1e6, line=dict(color=C["red"], dash="dash"), annotation_text="σy",
                      annotation_font_color=C["red"])
        fig.update_layout(height=330, title=f"von Mises stress through the depth at x = {vm.x:.4g} m",
                          xaxis_title="σvm (MPa)", yaxis_title="y (mm)")
        st.plotly_chart(fig, theme=None, key="sf_vm")
    st.markdown("**Assumptions**")
    st.markdown("\n".join(f"- {t}" for t in vm.assumptions))
    st.caption("The whole x–y grid is scanned, so mid-depth points with high shear and low bending (e.g. near "
               "supports of short beams) are considered as well as the extreme fibres.")

# --------------------------------------------------------------------------- buckling
with tabs[2]:
    inferred = safety.infer_end_condition(model)
    ec_opts = ["auto"] + list(safety.BUCKLING_K)
    a, b = st.columns(2, gap="large")
    ec_pick = a.selectbox(
        "End condition (effective-length factor K)", ec_opts, key="sf_ec",
        format_func=lambda k: (f"From supports: {inferred} (K = {safety.BUCKLING_K[inferred]})" if inferred
                               else "From supports: not identifiable") if k == "auto"
        else f"{k} (K = {safety.BUCKLING_K[k]})")
    P_model = safety.compressive_force(result)
    src = b.radio("Axial compressive load", ["From the model", "Enter a value"], horizontal=True, key="sf_psrc")
    if src == "Enter a value":
        P = b.number_input("Compressive load P (kN)", min_value=0.0, value=round(P_model / 1e3, 6), step=1.0,
                           key="sf_P") * 1e3
    else:
        P = P_model
        b.caption(f"Largest internal compression from the engine: {P_model / 1e3:.4g} kN"
                  + ("" if P_model > 0 else " (transverse loads only)."))
    ec = None if ec_pick == "auto" else ec_pick
    bres = safety.buckling(model, end_condition=ec, P_applied=P)
    if bres.end_condition is None:
        for w in bres.warnings:
            st.warning(w)
    else:
        q = st.columns(5)
        q[0].metric("K", f"{bres.K:g}")
        q[1].metric("L_eff = K·L", f"{bres.L_eff:.4g} m")
        q[2].metric("Euler Pcr", f"{bres.Pcr / 1e3:.4g} kN")
        q[3].metric("Slenderness L_eff/r", f"{bres.slenderness:.4g}", help=f"Johnson transition "
                                                                            f"{bres.slenderness_limit:.4g}")
        q[4].metric("Buckling FOS", fos_txt(bres.fos) if bres.fos else "n/a")
        if bres.fos is not None:
            st.markdown(status_badge(safety.STATUS_BUCKLING if bres.fos < 1 else safety.status_for(bres.fos, min_fos)),
                        unsafe_allow_html=True)
        a, b = st.columns([1.3, 1], gap="large")
        with a:
            fig = go.Figure(go.Scatter(x=bres.mode_x, y=bres.mode_shape, mode="lines",
                                       line=dict(color=C["amber"], width=3), fill="tozeroy",
                                       fillcolor="rgba(235,203,139,0.12)",
                                       hovertemplate="x = %{x:.4g} m<br>w/wmax = %{y:.3f}<extra></extra>"))
            fig.add_hline(y=0, line=dict(color=C["cyan"], width=4))
            fig.update_layout(height=300, title=f"First buckling mode ({bres.end_condition}), normalised",
                              xaxis_title="x (m)", yaxis_title="w / w_max", yaxis_range=[-1.15, 1.15])
            st.plotly_chart(fig, theme=None, key="sf_mode")
        with b:
            st.latex(r"P_{cr}=\frac{\pi^2 E I_{\min}}{(K L)^2},\qquad \lambda=\frac{K L}{r_{\min}},\qquad "
                     r"\lambda_t=\sqrt{\frac{2\pi^2E}{\sigma_y}}")
            st.dataframe(pd.DataFrame([{"End condition": k_, "K": v} for k_, v in safety.BUCKLING_K.items()]),
                         hide_index=True)
        for w in bres.warnings:
            st.caption(f"• {w}")
        if inferred is None and ec is not None:
            st.warning("The end condition was chosen manually; the model's supports do not identify it. Make sure it "
                       "matches the real restraint.")

# --------------------------------------------------------------------------- fatigue
with tabs[3]:
    st.caption("Only your data is used: no fatigue properties are assumed for library materials. Leave a field blank "
               "if you do not have it; the dependent results then show as 'insufficient data'.")
    smax_model = abs(result.sigma_max.value) / 1e6
    a, b, c = st.columns(3)
    s_max = a.number_input("σ_max of the cycle (MPa)", value=round(smax_model, 4), step=1.0, key="sf_smax",
                           help="Pre-filled with the design's |σ|max; edit for your load spectrum.")
    s_min = a.number_input("σ_min of the cycle (MPa)", value=0.0, step=1.0, key="sf_smin",
                           help="0 = repeated loading (load applied and removed).")
    uts = b.number_input("UTS (MPa)", value=round(mat.uts / 1e6, 4) if mat.uts else None, min_value=0.001,
                         key="sf_uts", help="From the material when available.")
    se = b.number_input("Endurance limit Se (MPa)", value=None, min_value=0.001, key="sf_se",
                        placeholder="your data")
    sfp = c.number_input("Fatigue strength coefficient σf′ (MPa)", value=None, min_value=0.001, key="sf_sfp",
                         placeholder="your data")
    bexp = c.number_input("Basquin exponent b (negative)", value=None, max_value=-0.001, min_value=-0.5,
                          step=0.005, format="%.4f", key="sf_b", placeholder="e.g. your test data")
    try:
        fr = safety.fatigue(s_max * 1e6, s_min * 1e6, None if uts is None else uts * 1e6,
                            None if sfp is None else sfp * 1e6, bexp, None if se is None else se * 1e6)
    except ValueError as exc:
        st.error(str(exc))
        fr = None
    if fr is not None:
        q = st.columns(5)
        q[0].metric("Amplitude σa", f"{fr.sigma_a / 1e6:.4g} MPa")
        q[1].metric("Mean σm", f"{fr.sigma_m / 1e6:.4g} MPa")
        q[2].metric("Goodman σar", f"{fr.sigma_ar / 1e6:.4g} MPa" if fr.sigma_ar is not None else "insufficient data")
        q[3].metric("Cycles to failure", ("∞" if math.isinf(fr.cycles_to_failure) else f"{fr.cycles_to_failure:.3e}")
                    if fr.cycles_to_failure is not None else "insufficient data")
        q[4].metric("Goodman FOS", fos_txt(fr.goodman_fos) if fr.goodman_fos is not None else "insufficient data")
        if fr.sigma_ar_gerber is not None:
            st.caption(f"Gerber equivalent amplitude (comparison): {fr.sigma_ar_gerber / 1e6:.4g} MPa. "
                       + (f"Infinite life: {'yes' if fr.infinite_life else 'no'}." if fr.infinite_life is not None
                          else ""))
        g1, g2 = st.columns(2, gap="large")
        if fr.sn_N.size:
            fig = go.Figure(go.Scatter(x=fr.sn_N, y=fr.sn_S / 1e6, mode="lines", line=dict(color=C["cyan"], width=2.6),
                                       name="S–N (Basquin)", hovertemplate="N = %{x:.3e}<br>S = %{y:.4g} MPa<extra></extra>"))
            amp = fr.sigma_ar if fr.sigma_ar is not None else fr.sigma_a
            if fr.cycles_to_failure and math.isfinite(fr.cycles_to_failure):
                fig.add_trace(go.Scatter(x=[fr.cycles_to_failure], y=[amp / 1e6], mode="markers", name="this cycle",
                                         marker=dict(size=12, color=C["amber"], symbol="x")))
            else:
                fig.add_hline(y=amp / 1e6, line=dict(color=C["amber"], dash="dot"), annotation_text="this cycle")
            fig.update_layout(height=320, title="S–N curve from your σf′, b" + (" and Se" if se else ""),
                              xaxis=dict(type="log", title="cycles N"), yaxis=dict(type="log", title="σ_ar (MPa)"),
                              showlegend=False)
            g1.plotly_chart(fig, theme=None, key="sf_sn")
        if se is not None and uts is not None:
            fig = go.Figure([go.Scatter(x=[0, uts], y=[se, 0], mode="lines", name="Goodman line",
                                        line=dict(color=C["red"], width=2.4)),
                             go.Scatter(x=[fr.sigma_m / 1e6], y=[fr.sigma_a / 1e6], mode="markers+text",
                                        text=["operating point"], textposition="top right", name="operating point",
                                        marker=dict(size=12, color=C["amber"]))])
            fig.update_layout(height=320, title="Goodman diagram", xaxis_title="σm (MPa)", yaxis_title="σa (MPa)",
                              showlegend=False)
            g2.plotly_chart(fig, theme=None, key="sf_goodman")
        st.latex(r"\sigma_a=\frac{\sigma_{\max}-\sigma_{\min}}{2},\quad \sigma_m=\frac{\sigma_{\max}+\sigma_{\min}}{2},"
                 r"\quad \sigma_{ar}=\frac{\sigma_a}{1-\sigma_m/S_u},\quad \sigma_{ar}=\sigma_f'(2N)^b")
        for w in fr.warnings:
            st.warning(w, icon=":material/info:")
        with st.expander("Assumptions"):
            st.markdown("\n".join(f"- {t}" for t in fr.assumptions))

# --------------------------------------------------------------------------- summary
with tabs[4]:
    for msg in rep.messages[:-1]:
        st.markdown(f"- {msg}")
    st.warning(safety.DISCLAIMER, icon=":material/warning:")
    st.caption("Status scale: " + " · ".join(f"<span style='color:{c}'>{s}</span>" for s, c in STATUS_COLOR.items()),
               unsafe_allow_html=True)
