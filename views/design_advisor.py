"""Module 10 — Smart design advisor (rule-based). Every recommendation is a re-analysed design."""
import math

import plotly.graph_objects as go
import streamlit as st

from smartbeam import advisor
from smartbeam.ui.design_tools import before_after_frame, pill, run_advise, section_overlay, show_checks
from smartbeam.ui.plots_core import card, model_caption, require_result
from smartbeam.ui.state import get_model, set_model
from smartbeam.ui.theme import COLORS, page_header

page_header("Design Advisor",
            "Rule-based diagnosis of what limits the current design and how to fix it. Each fix is sized from beam "
            "theory and then re-analysed by the engine. No guesswork, no chatbot.")

model = get_model()
st.markdown(model_caption(model))
require_result(model, record=False)
model_json = model.to_json(indent=None)
if st.session_state.pop("advisor_applied", None) == model.name:
    st.success(f"Applied **{model.name}**: it is now the current design, re-diagnosed below.",
               icon=":material/check_circle:")

c = st.columns([1, 1, 2])
delta_mm = c[0].number_input("Allowable deflection δ_allow (mm)", min_value=1e-4,
                             value=float(f"{model.L / advisor.DEFAULT_SPAN_RATIO * 1e3:.4g}"), format="%.4g",
                             help=f"Default L/{advisor.DEFAULT_SPAN_RATIO}, a common serviceability guideline.")
min_fos = c[1].number_input("Minimum FOS", min_value=1.0, value=1.5, step=0.1, format="%.2f")
c[2].caption(f"Criteria: |v|max ≤ δ_allow, yield FOS ≥ FOS_min, and Euler buckling FOS ≥ FOS_min when the member is "
             f"in compression. Utilisation u = demand / capacity; u > 1 fails, and every u < "
             f"{advisor.EXCESS_UTILISATION:g} flags material excess.")
st.session_state["advisor_criteria"] = (delta_mm * 1e-3, min_fos)  # reused by the Reports page

try:
    rep = run_advise(model_json, delta_mm * 1e-3, min_fos)
except ValueError as exc:
    st.error(f"**Advisor cannot run.** {exc}")
    st.stop()

# ------------------------------------------------------------------ diagnosis
LIMIT_TEXT = {
    "deflection": ("Stiffness governs", "The beam deflects more than allowed: increase I (depth matters most) or E."),
    "stress": ("Strength governs", "The bending stress leaves too little margin to yield: increase Z or σy."),
    "buckling": ("Buckling governs", "Axial compression approaches the Euler load: increase I_min or shorten L_eff."),
    "material excess": ("Over-designed", "Every criterion has more than 2× reserve, so material can be removed."),
    "none": ("Design is adequate", "Every criterion passes with sensible utilisation; no change needed."),
}
title, text = LIMIT_TEXT[rep.limiting]
color = {"none": COLORS["green"], "material excess": COLORS["amber"]}.get(rep.limiting, COLORS["red"])
left, right = st.columns([2, 3])
with left:
    st.markdown(f"#### Limiting condition &nbsp; {pill(rep.limiting.upper(), color)}", unsafe_allow_html=True)
    card(title, text, color)
    st.markdown("")
    show_checks(rep.checks)
with right:
    names = list(rep.utilisation)
    vals = [rep.utilisation[k] for k in names]
    bar_c = [COLORS["red"] if v > 1 else COLORS["amber"] if v > 0.9 else COLORS["green"] for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=[n.title() for n in names], orientation="h", marker_color=bar_c,
                           text=[f"{v:.2f}" for v in vals], textposition="outside",
                           hovertemplate="%{y}: u = %{x:.3f}<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color=COLORS["red"], width=2), annotation_text="limit u = 1",
                  annotation_font_color=COLORS["red"])
    fig.add_vline(x=advisor.EXCESS_UTILISATION, line=dict(color=COLORS["amber"], dash="dot"),
                  annotation_text="excess threshold", annotation_position="bottom right",
                  annotation_font_color=COLORS["amber"])
    fig.update_layout(height=240, title="Utilisation u = demand / capacity", xaxis_title="u",
                      xaxis_range=[0, max(1.25, max(vals) * 1.2)], margin=dict(t=48, b=40))
    st.plotly_chart(fig, theme=None, width="stretch")

with st.expander("How the advisor sizes a fix", icon=":material/function:"):
    st.latex(r"u_\delta = \frac{|v|_{\max}}{\delta_{allow}},\qquad u_\sigma = \frac{\mathrm{FOS}_{\min}\,|\sigma|_{\max}}"
             r"{\sigma_y},\qquad u_{b} = \frac{\mathrm{FOS}_{\min}\,P}{P_{cr}}")
    st.latex(r"\delta \propto \frac{1}{EI},\quad \sigma \propto \frac{1}{Z},\quad P_{cr} \propto E\,I_{\min}"
             r"\;\;\Rightarrow\;\; h_{new} = h\,u_\delta^{1/3}\ (\text{rect depth}),\qquad"
             r" s = u_\sigma^{1/3}\ (\text{circle, all dims})")
    st.markdown("The scale factor from these laws is applied, dimensions are rounded **up** to 0.1 mm, and the "
                "modified beam is **re-analysed** with the full engine (statics or FEM). If a combined effect "
                "leaves a criterion unmet, the step repeats from the re-analysed values. Material suggestions keep "
                "the section and pick the lightest library material that passes every check.")

# ------------------------------------------------------------------ recommendations
st.divider()
if not rep.recommendations:
    st.success("No modification recommended: the current design meets every criterion without large excess "
               "capacity.", icon=":material/verified:")
    st.stop()

st.markdown(f"### Recommendations ({len(rep.recommendations)})")
QUANTITY = {"delta_max": ("Max deflection |v|", 1e3, "mm"), "sigma_max": ("Max normal stress |σ|", 1e-6, "MPa"),
            "buckling": ("Axial compression P (limit Pcr / FOS_min)", 1e-3, "kN"), "mass": ("Beam mass", 1.0, "kg")}
for i, rec in enumerate(rep.recommendations, 1):
    qname, qs, qu = QUANTITY[rec.quantity]
    kind = ("Change material" if rec.action.startswith("Change material") else
            "Alternative section" if rec.action.startswith("Alternative") else "Resize section")
    with st.container(border=True):
        st.markdown(f"#### {i}. {kind} &nbsp; {pill('SATISFIES ALL' if rec.satisfies_all else 'STILL FAILS', COLORS['green'] if rec.satisfies_all else COLORS['red'])}",
                    unsafe_allow_html=True)
        q1, q2 = st.columns(2)
        q1.markdown(f"**1 · What is wrong?**  \n{rec.issue}")
        lim = "—" if math.isnan(rec.limit) else f"{rec.limit * qs:.4g} {qu}"
        q2.markdown(f"**2 · Which quantity?**  \n{qname}: current **{rec.current * qs:.4g} {qu}**, "
                    f"{'target' if rec.quantity == 'mass' else 'limit'} **{lim}**")
        st.markdown(f"**3 · Recommended modification**  \n{rec.action}")
        a, b = st.columns([3, 2])
        with a:
            st.markdown("**4 · What changes after it?** (re-analysed)")
            st.dataframe(before_after_frame(rec.before, rec.after, ("Current", "Modified")), hide_index=True,
                         width="stretch")
        with b:
            st.plotly_chart(section_overlay([(model.section, "Current", COLORS["muted"]),
                                             (rec.modified_model.section, "Modified", COLORS["cyan"])], height=230),
                            theme=None, width="stretch", key=f"advisor_sec_{i}")
        st.markdown(f"**5 · Does the modified design satisfy all constraints?** "
                    f"{'Yes' if rec.satisfies_all else 'No'}, from a fresh analysis of the modified beam:")
        show_checks(rec.checks_after)
        if st.button("Apply this design", key=f"advisor_apply_{i}", icon=":material/done:"):
            m = rec.modified_model.copy()
            m.name = f"{model.name or 'Design'} (advisor fix {i})"
            set_model(m)
            st.session_state["advisor_applied"] = m.name
            st.rerun()
