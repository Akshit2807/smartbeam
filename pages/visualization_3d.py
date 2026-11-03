"""Interactive 3D view: deformed beam from the computed v(x), beam-theory stress colour map, supports, loads."""
import streamlit as st

from smartbeam import viz3d
from smartbeam.ui.plots_core import METHOD_LABEL, model_caption, require_result, show_warnings
from smartbeam.ui.state import get_model
from smartbeam.ui.theme import page_header

page_header("3D Visualization", "Rotate, zoom and pan the analysed beam. Colours show bending stress; the shape is "
            "the computed deflection, magnified.")
model = get_model()
result = require_result(model)
st.markdown(model_caption(model))
st.info(viz3d.DISCLAIMER, icon=":material/info:")

auto = viz3d.auto_amplification(result)
ctrl, view = st.columns([1, 3.2], gap="large")
with ctrl:
    st.markdown("##### Display")
    mode = st.radio("Deformation amplification", ["Auto", "True scale (×1)", "Custom"], key="v3d_mode",
                    help="Real deflections are tiny compared with the beam; the displayed displacement is "
                         "multiplied by this factor. Stress values are never scaled.")
    if mode == "Custom":
        amp = st.slider("Amplification factor ×", 1.0, max(10.0, 5 * auto), float(auto), key="v3d_amp")
    else:
        amp = auto if mode == "Auto" else 1.0
    st.markdown(f"Displacements shown **×{amp:.4g}**")
    show = {
        "show_deformed": st.toggle("Deformed beam", True, key="v3d_def"),
        "show_undeformed": st.toggle("Undeformed beam", True, key="v3d_undef"),
        "show_stress": st.toggle("Stress colour map σ", True, key="v3d_stress"),
        "show_supports": st.toggle("Supports", True, key="v3d_sup"),
        "show_loads": st.toggle("Loads", True, key="v3d_loads"),
        "show_markers": st.toggle("Max σ / max v markers", True, key="v3d_mark"),
    }
    st.divider()
    st.metric("Max |v|", f"{abs(result.v_max.value) * 1e3:.4g} mm", help=f"at x = {result.v_max.x:.4g} m")
    st.metric("Max |σ|", f"{abs(result.sigma_max.value) / 1e6:.4g} MPa",
              help=f"{result.sigma_max.where} fibre at x = {result.sigma_max.x:.4g} m")
    st.caption(f"Solver: {METHOD_LABEL[result.method]}")
with view:
    if not (show["show_deformed"] or show["show_undeformed"]):
        st.warning("Both beam layers are hidden; turn one on to see the beam.")
    fig = viz3d.beam_figure(result, amplification=amp, height=640, **show)
    st.plotly_chart(fig, theme=None, key="v3d_fig")
    st.caption("Blue = compression, red = tension (σ = N/A − M·y/I, linear through the depth, uniform across the "
               "width). Hover the surface for x, y, σ and v; hover the cyan centreline for v and θ. Supports and loads "
               "are symbols, not to scale.")
show_warnings(result)
