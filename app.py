"""SmartBeam 2.0 entry point:  streamlit run app.py"""
import streamlit as st

from smartbeam.ui.theme import apply_theme

st.set_page_config(page_title="SmartBeam 2.0", page_icon=":material/architecture:", layout="wide")
apply_theme()

P = lambda file, title, icon, **kw: st.Page(f"pages/{file}.py", title=title, icon=f":material/{icon}:", **kw)  # noqa: E731

nav = st.navigation({
    "Overview": [P("home", "Home", "home", default=True)],
    "Design & Analysis": [
        P("design_studio", "Design Studio", "design_services"),
        P("analysis", "Analysis", "functions"),
        P("numerical_fem", "Numerical FEM", "grid_on"),
        P("beam_theories", "Beam Theories", "menu_book"),
        P("visualization_3d", "3D Visualization", "view_in_ar"),
    ],
    "Evaluate & Improve": [
        P("safety_failure", "Safety & Failure", "health_and_safety"),
        P("design_comparison", "Design Comparison", "compare"),
        P("optimization", "Optimization", "tune"),
        P("design_advisor", "Design Advisor", "lightbulb"),
    ],
    "Explore": [
        P("virtual_lab", "Virtual Lab", "science"),
        P("playground", "Structural Playground", "draw"),
    ],
    "Output": [
        P("reports", "Reports", "description"),
        P("about", "About & Methodology", "info"),
    ],
}, expanded=True)

with st.sidebar:
    st.caption("SmartBeam 2.0 · Understand. Simulate. Predict. Optimize.")

nav.run()
