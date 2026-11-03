"""Beam theories: Euler-Bernoulli vs Timoshenko on the current design, shear share vs L/h, guidance."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smartbeam import fem, sections
from smartbeam.model import BeamModel, DistributedLoad, MomentLoad, PointLoad, Support
from smartbeam.ui.plots_core import fem_compare, model_caption, require_result
from smartbeam.ui.state import get_model
from smartbeam.ui.theme import COLORS, page_header

C = COLORS
page_header("Beam Theories", "Euler–Bernoulli neglects shear deformation; Timoshenko includes it. "
            "How much does it matter for your beam, and when?")
model = get_model()
require_result(model, record=False)  # validates and stops with a clear message if the model is unusable
st.markdown(model_caption(model))

a, b = st.columns(2, gap="large")
with a:
    with st.container(border=True):
        st.markdown(f"#### <span style='color:{C['cyan']}'>Euler–Bernoulli</span>", unsafe_allow_html=True)
        st.markdown("- Plane sections remain plane **and perpendicular** to the deformed axis\n"
                    "- Shear strain is neglected (infinite shear stiffness)\n"
                    "- Section rotation equals the slope: ψ = dv/dx\n- Small deflections, linear elastic material")
        st.latex(r"EI\,\frac{d^4v}{dx^4}=q(x),\qquad M=EI\,\frac{d^2v}{dx^2},\qquad \psi=\frac{dv}{dx}")
with b:
    with st.container(border=True):
        st.markdown(f"#### <span style='color:{C['amber']}'>Timoshenko</span>", unsafe_allow_html=True)
        st.markdown("- Plane sections remain plane but **not** perpendicular to the axis\n"
                    "- Constant shear strain γ through the depth, corrected by κ\n"
                    "- Section rotation ψ and deflection v are independent fields\n"
                    "- Matters for short, deep or shear-flexible beams")
        st.latex(r"M=EI\,\frac{d\psi}{dx},\qquad V=\kappa G A\left(\frac{dv}{dx}-\psi\right),\qquad G=\frac{E}{2(1+\nu)}")

# --------------------------------------------------------------------------- current design
st.subheader("Your design: both theories on the same mesh")
cmp = fem_compare(model.to_json(indent=None), 20)
k = st.columns(6)
k[0].metric("|v|max Euler–Bernoulli", f"{cmp.v_eb * 1e3:.5g} mm")
k[1].metric("|v|max Timoshenko", f"{cmp.v_timoshenko * 1e3:.5g} mm")
k[2].metric("Shear contribution", f"{cmp.diff_pct:.3g} %", help="100·(v_T − v_EB)/v_EB")
k[3].metric("L / depth", f"{cmp.span_to_depth:.3g}")
k[4].metric("κ (shear factor)", f"{cmp.kappa:.4g}")
k[5].metric("φ = 12EI/(κGAL²)", f"{cmp.phi_beam:.3g}")

fig = go.Figure([
    go.Scatter(x=cmp.eb.x, y=cmp.eb.v * 1e3, name="Euler–Bernoulli", line=dict(color=C["cyan"], width=2.6)),
    go.Scatter(x=cmp.timoshenko.x, y=cmp.timoshenko.v * 1e3, name="Timoshenko",
               line=dict(color=C["amber"], width=2.6, dash="dash"))])
fig.update_layout(height=320, title="Deflection v(x) (mm)", xaxis_title="x (m)", yaxis_title="v (mm)",
                  legend=dict(orientation="h", y=1.15, x=1, xanchor="right"))
st.plotly_chart(fig, theme=None, key="bt_v")
st.caption(f"G = {cmp.G / 1e9:.4g} GPa from E and ν = {model.material.nu:g}. Both solutions use 20 requested "
           "elements; the Timoshenko element uses interdependent interpolation with φ per element, which is exact "
           "for prismatic beams (no shear locking, no arbitrary correction factor).")


# --------------------------------------------------------------------------- L/h sweep
def scaled(m: BeamModel, L_new: float) -> BeamModel:
    """Same beam, section and load magnitudes; span and every position scaled to L_new."""
    f = L_new / m.L
    loads = []
    for ld in m.loads:
        if isinstance(ld, PointLoad):
            loads.append(PointLoad(ld.x * f, ld.magnitude, ld.angle_deg))
        elif isinstance(ld, DistributedLoad):
            loads.append(DistributedLoad(ld.x1 * f, ld.x2 * f, ld.w1, ld.w2, ld.direction))
        else:
            loads.append(MomentLoad(ld.x * f, ld.M))
    return BeamModel(L_new, m.section, m.material, [Support(s.type, s.x * f) for s in m.supports], loads, m.name)


@st.cache_data(max_entries=16, show_spinner="Solving both theories across span/depth ratios…")
def sweep(model_json: str, ratios: tuple) -> pd.DataFrame:
    m = BeamModel.from_json(model_json)
    rows = []
    for r in ratios:
        c = fem.compare_theories(scaled(m, r * m.section.depth), n_elements=20)
        rows.append({"L/h": r, "L (m)": r * m.section.depth, "v_EB (mm)": c.v_eb * 1e3,
                     "v_T (mm)": c.v_timoshenko * 1e3, "Shear share (%)": c.diff_pct})
    return pd.DataFrame(rows)


st.subheader("Shear contribution versus slenderness")
st.caption("The span is varied with the section, material, supports and load magnitudes held fixed (positions scale "
           "with the span). Each point is a pair of FEM solutions.")
ratios = tuple(float(r) for r in np.unique(np.r_[np.geomspace(1.5, 80, 18), round(cmp.span_to_depth, 6)]))
df = sweep(model.to_json(indent=None), ratios)
fig = go.Figure()
fig.add_vrect(x0=1, x1=5, fillcolor=C["red"], opacity=0.08, line_width=0, annotation_text="deep: Timoshenko",
              annotation_position="top left", annotation_font_color=C["red"])
fig.add_vrect(x0=5, x1=10, fillcolor=C["amber"], opacity=0.07, line_width=0, annotation_text="transition",
              annotation_position="top left", annotation_font_color=C["amber"])
fig.add_vrect(x0=10, x1=100, fillcolor=C["green"], opacity=0.05, line_width=0, annotation_text="slender: E–B adequate",
              annotation_position="top left", annotation_font_color=C["green"])
fig.add_trace(go.Scatter(x=df["L/h"], y=df["Shear share (%)"], mode="lines+markers", name="Timoshenko − EB",
                         line=dict(color=C["cyan"], width=2.6),
                         hovertemplate="L/h = %{x:.3g}<br>shear share = %{y:.4g} %<extra></extra>"))
cur = df.iloc[(df["L/h"] - cmp.span_to_depth).abs().argmin()]
fig.add_trace(go.Scatter(x=[cur["L/h"]], y=[cur["Shear share (%)"]], mode="markers+text", name="your design",
                         marker=dict(size=13, color=C["amber"], symbol="star"), text=["your design"],
                         textposition="top right", textfont=dict(color=C["amber"])))
fig.update_layout(height=400, xaxis=dict(type="log", title="span / depth  L/h", range=[np.log10(1.4), np.log10(85)]),
                  yaxis=dict(type="log", title="extra deflection from shear (%)"), showlegend=False)
st.plotly_chart(fig, theme=None, key="bt_sweep")
with st.expander("Sweep data"):
    st.dataframe(df, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.5g")
                                                     for c in df.columns})
if len(df) > 2 and (df["Shear share (%)"] > 0).all():
    slope = np.polyfit(np.log(df["L/h"]), np.log(df["Shear share (%)"]), 1)[0]
    st.caption(f"Fitted log–log slope of the shear share: {slope:.2f}. Shear deflection scales with L while bending "
               f"deflection scales with L³, so the ratio falls roughly as (h/L)², i.e. a slope near −2 for slender "
               f"beams.")

# --------------------------------------------------------------------------- guidance
st.subheader("Which theory should I use?")
st.dataframe(pd.DataFrame([
    {"Situation": "Slender beams, L/h > 10 (most shafts, joists, frames)", "Recommended": "Euler–Bernoulli",
     "Why": "Shear adds well under a few percent of the deflection."},
    {"Situation": "Short/deep beams, L/h < 5 (deep girders, short brackets)", "Recommended": "Timoshenko",
     "Why": "Shear deformation is a large part of the deflection."},
    {"Situation": "5 ≤ L/h ≤ 10", "Recommended": "Check both",
     "Why": "The difference is typically a few percent up to ~10 %, depending on supports and section."},
    {"Situation": "Thin-walled tubes, sandwich or low-G materials", "Recommended": "Timoshenko",
     "Why": "Small κ or G (κ ≈ 0.5 for thin tubes) amplifies shear flexibility."},
    {"Situation": "Higher vibration modes, wave propagation", "Recommended": "Timoshenko",
     "Why": "Short wavelengths behave like short beams (beyond SmartBeam's static scope)."},
]), hide_index=True)
st.markdown("**Shear correction factor κ** used by SmartBeam:")
st.dataframe(pd.DataFrame([
    {"Section": "Rectangular", "κ": "5/6 ≈ 0.833", "Source": "classical energy value"},
    {"Section": "Solid circular", "κ": "0.9", "Source": "classical value (6/7 is also quoted in texts)"},
    {"Section": "Hollow circular", "κ": f"Cowper (1966): {sections.shear_correction_factor(model.section, model.material.nu):.4g} for your section"
     if model.section.shape == "hollow_circ" else "Cowper (1966), ≈ 0.53 for thin tubes",
     "Source": "κ = 6(1+ν)(1+m²)² / [(7+6ν)(1+m²)² + (20+12ν)m²], m = d/D"},
]), hide_index=True)
st.caption("Cowper's formula at m = 0 gives 0.886, slightly below the 0.9 used for solid circles; both are standard "
           "approximations.")
