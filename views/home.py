"""Home dashboard: hero, capability cards with navigation, live 3D example beam, recent analyses."""
import pandas as pd
import streamlit as st

from smartbeam import viz3d
from smartbeam.model import BeamModel
from smartbeam.ui.plots_core import METHOD_LABEL, cached_figure, model_caption, require_result
from smartbeam.ui.state import get_model, recent_analyses, set_model
from smartbeam.ui.theme import COLORS

C = COLORS
st.markdown(f"""
<div style="padding:34px 34px 30px;border:1px solid {C['border']};border-radius:16px;margin-bottom:22px;
  background:radial-gradient(1200px 300px at 15% -20%, rgba(136,192,208,0.16), transparent 60%),
             radial-gradient(800px 260px at 95% 120%, rgba(235,203,139,0.10), transparent 60%), {C['panel']};">
  <div style="color:{C['cyan']};font-weight:700;letter-spacing:0.14em;font-size:0.78rem">
    SOLID MECHANICS · BEAM ANALYSIS SUITE</div>
  <div style="font-size:2.7rem;font-weight:800;line-height:1.1;margin:10px 0 6px">SmartBeam 2.0</div>
  <div style="font-size:1.35rem;color:{C['text']};font-weight:500">
    Understand. <span style="color:{C['cyan']}">Simulate.</span> Predict.
    <span style="color:{C['amber']}">Optimize.</span></div>
  <div style="color:{C['muted']};max-width:780px;margin-top:12px;font-size:0.98rem">
    Design a beam, solve it with classical statics and an independent 1D finite-element engine, compare
    Euler–Bernoulli with Timoshenko theory, check yield, buckling and fatigue, and let the optimizer and advisor
    size it. Every number on every page comes from the calculation engine.</div>
</div>
""", unsafe_allow_html=True)

model = get_model()
result = require_result(model, record=False)

left, right = st.columns([1.35, 1], gap="large")
with left:
    st.subheader("Current beam in 3D")
    st.caption(model_caption(model))
    fig = cached_figure(("3d-home", model.to_json(indent=None)), lambda: viz3d.beam_figure(result, height=430))
    st.plotly_chart(fig, theme=None)
    st.caption(f"Deformation magnified ×{viz3d.auto_amplification(result):.3g}; colour = bending stress σ = N/A − My/I "
               "(beam theory). Drag to rotate, scroll to zoom.")
with right:
    st.subheader("Key results")
    a, b = st.columns(2)
    a.metric("Max deflection", f"{abs(result.v_max.value) * 1e3:.3g} mm", help=f"at x = {result.v_max.x:.4g} m")
    b.metric("Max |σ|", f"{abs(result.sigma_max.value) / 1e6:.3g} MPa",
             help=f"{result.sigma_max.where} fibre at x = {result.sigma_max.x:.4g} m")
    a.metric("Max |M|", f"{abs(result.M_max.value) / 1e3:.3g} kN·m", help=f"at x = {result.M_max.x:.4g} m")
    sig = abs(result.sigma_max.value)
    b.metric("Yield FOS", f"{model.material.yield_strength / sig:.3g}" if sig > 0 else "∞")
    st.caption(f"Solver route: **{METHOD_LABEL[result.method]}** · {result.stability.message}")
    st.page_link("views/design_studio.py", label="Edit this design", icon=":material/design_services:")
    st.page_link("views/analysis.py", label="Full analysis", icon=":material/functions:")
    st.page_link("views/visualization_3d.py", label="Open the 3D viewer", icon=":material/view_in_ar:")

st.divider()
st.subheader("Capabilities")
CAPS = [
    ("views/design_studio.py", "Design Studio", "design_services",
     "Section, material library, supports and loads with unit-aware validation."),
    ("views/analysis.py", "Analysis", "functions",
     "Reactions, SFD, BMD, stress/strain through the depth, slope and deflection, equilibrium checks."),
    ("views/numerical_fem.py", "Numerical FEM", "grid_on",
     "Independent 1D beam FEM: mesh, stiffness matrix, nodal results, convergence 2→32 elements."),
    ("views/beam_theories.py", "Beam Theories", "menu_book",
     "Euler–Bernoulli vs Timoshenko and the shear-deformation share versus L/h."),
    ("views/visualization_3d.py", "3D Visualization", "view_in_ar",
     "Deformed beam with stress colour map, supports, loads and max markers."),
    ("views/safety_failure.py", "Safety & Failure", "health_and_safety",
     "Yield FOS, von Mises, Euler buckling and stress-life fatigue from your data."),
    ("views/design_comparison.py", "Design Comparison", "compare",
     "Save designs and compare stiffness, strength, mass and cost."),
    ("views/optimization.py", "Optimization", "tune",
     "SLSQP sizing with independent re-verification and Pareto trade-offs."),
    ("views/design_advisor.py", "Design Advisor", "lightbulb",
     "Rule-based diagnosis of the limiting criterion with verified fixes."),
    ("views/virtual_lab.py", "Virtual Lab", "science",
     "Guided experiments, design challenges and sensitivity studies."),
    ("views/playground.py", "Structural Playground", "draw",
     "Drag-and-drop supports and loads on a live canvas."),
    ("views/reports.py", "Reports", "description", "PDF engineering report and CSV data export."),
]
for row in range(0, len(CAPS), 4):
    cols = st.columns(4)
    for col, (path, title, icon, desc) in zip(cols, CAPS[row:row + 4]):
        with col.container(border=True, height=150):
            st.page_link(path, label=title, icon=f":material/{icon}:")
            st.caption(desc)

st.divider()
st.subheader("Recent analyses")
recent = recent_analyses()
if not recent:
    st.info("No analyses yet in this session. Open the Analysis page (or any results page) and it will appear here.")
else:
    df = pd.DataFrame([{
        "Time": r["time"], "Name": r["name"], "Configuration": r["config"], "Solver": r["method"],
        "L (m)": r["L_m"], "|v|max (mm)": r["delta_max_m"] * 1e3, "|σ|max (MPa)": r["sigma_max_Pa"] / 1e6,
    } for r in recent])
    st.dataframe(df, hide_index=True,
                 column_config={"|v|max (mm)": st.column_config.NumberColumn(format="%.4g"),
                                "|σ|max (MPa)": st.column_config.NumberColumn(format="%.4g")})
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    pick = c1.selectbox("Restore a recent configuration", range(len(recent)),
                        format_func=lambda i: f"{recent[i]['time']} · {recent[i]['name']} ({recent[i]['config']})")
    if c2.button("Load into workspace", width="stretch"):
        set_model(BeamModel.from_dict(recent[pick]["model"]))
        st.toast("Configuration restored.")
        st.rerun()

st.divider()
st.subheader("How SmartBeam solves your beam")
t1, t2, t3, t4 = st.columns(4)
for col, (title, body, color) in zip((t1, t2, t3, t4), [
    ("1 · Classical engine", "Statically determinate transverse beams: reactions from ΣF = 0, ΣM = 0, exact V and M, "
     "slope and deflection by integrating M/EI.", C["cyan"]),
    ("2 · Stiffness FEM", "Indeterminate beams use Euler–Bernoulli beam elements; axial or inclined loads use planar "
     "frame elements with (u, v, θ) per node.", C["violet"]),
    ("3 · Safety checks", "Yield FOS, von Mises with stated assumptions, Euler buckling and stress-life fatigue "
     "(only with your data). Yield is never called fracture.", C["green"]),
    ("4 · Design tools", "Optimization re-verifies every candidate with the engine; the advisor re-analyses each "
     "suggested fix before recommending it.", C["amber"]),
]):
    with col:
        st.markdown(f'<div class="sb-card" style="border-top:2px solid {color};height:100%">'
                    f'<div style="font-weight:700;margin-bottom:6px">{title}</div>'
                    f'<div style="color:{C["muted"]};font-size:0.9rem">{body}</div></div>', unsafe_allow_html=True)
