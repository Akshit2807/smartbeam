"""Module 9 — Intelligent design optimization (SciPy SLSQP), re-verified by the analysis engine."""
import plotly.graph_objects as go
import streamlit as st

from smartbeam import optimize
from smartbeam.materials import MATERIALS, material_keys
from smartbeam.model import ModelError
from smartbeam.ui.design_tools import before_after_frame, g, pill, run_pareto, section_overlay, show_checks
from smartbeam.ui.plots_core import model_caption, require_result
from smartbeam.ui.state import get_model, set_model
from smartbeam.ui.theme import COLORS, SERIES, page_header

page_header("Design Optimization",
            "SciPy SLSQP sizes the cross-section for minimum volume, mass or cost under deflection and strength "
            "limits. Every result is re-checked by an independent engine analysis before it is called optimized.")

model = get_model()
st.markdown(model_caption(model))
require_result(model, record=False)  # stops the page with a clear error if the current design cannot be analysed
model_json = model.to_json(indent=None)
shape = model.section.shape
dims = model.section.dims()

with st.expander("Problem formulation", icon=":material/function:"):
    st.latex(r"\min_{\mathbf{d}}\; f(\mathbf{d}) \in \{\,V = A(\mathbf{d})\,L,\;\; m = \rho V,\;\; C = c_{kg}\, m\,\}")
    st.latex(r"\text{s.t.}\quad |v|_{\max}(\mathbf{d}) \le \delta_{allow},\qquad "
             r"\mathrm{FOS} = \frac{\sigma_y}{|\sigma|_{\max}(\mathbf{d})} \ge \mathrm{FOS}_{\min},\qquad "
             r"d_i^{\min} \le d_i \le d_i^{\max}")
    st.markdown(
        "- **Design variables** `d`: the section dimensions (rectangle b, h · solid circle d · tube D, d with wall "
        "(D − d)/2 ≥ t_min). Loads, supports, span and material stay as in the current design.\n"
        "- **Each evaluation** runs the full analysis engine (statics or FEM, whichever the structure needs).\n"
        "- **Buckling**: if the member carries axial compression with identifiable end conditions, an Euler "
        "constraint P_cr / P ≥ FOS_min is added automatically.\n"
        "- **Verification**: SLSQP starts from 6 points; each candidate is re-analysed from scratch and only one "
        "that passes every constraint is labelled OPTIMIZED. Otherwise the result is INFEASIBLE (with the reason) "
        "or FAILED.\n"
        "- **Not modelled**: local wall buckling and lateral–torsional buckling, so very thin or very narrow optima "
        "can be impractical. Tighten the bounds or t_min to keep proportions sensible.")

# ------------------------------------------------------------------ inputs
OBJECTIVES = {"volume": "Volume V = A·L", "mass": "Mass m = ρ·V", "cost": "Material cost C = ₹/kg · m"}
with st.form("optimization_form"):
    st.markdown("##### Objective and constraints")
    c = st.columns(4)
    objective = c[0].selectbox("Minimise", list(OBJECTIVES), format_func=OBJECTIVES.get)
    delta_mm = c[1].number_input("Allowable deflection δ_allow (mm)", min_value=1e-4,
                                 value=float(f"{model.L / 250 * 1e3:.4g}"), format="%.4g",
                                 help="Default L/250, a common serviceability limit.")
    fos_min = c[2].number_input("Minimum FOS", min_value=1.0, value=1.5, step=0.1, format="%.2f")
    rate0 = model.material.cost_per_kg
    rate = c[3].number_input("Material cost (₹/kg)", min_value=0.0, value=float(rate0 or 0.0), step=5.0,
                             help="Approximate market rate; edit to your supplier's price. Used for cost.")
    st.markdown("##### Dimension bounds (mm)")
    st.caption("Defaults are 0.2× to 5× the current dimensions.")
    lo_f, hi_f = optimize.DEFAULT_BOUND_FACTORS
    bounds, bc = {}, st.columns(2 * len(dims) + (1 if shape == "hollow_circ" else 0))
    for i, (k, v) in enumerate(dims.items()):
        lo = bc[2 * i].number_input(f"{k} min", min_value=0.01, value=float(f"{v * lo_f * 1e3:.4g}"), format="%.4g")
        hi = bc[2 * i + 1].number_input(f"{k} max", min_value=0.01, value=float(f"{v * hi_f * 1e3:.4g}"), format="%.4g")
        bounds[k] = (lo * 1e-3, hi * 1e-3)
    t_min = 1e-3
    if shape == "hollow_circ":
        t_min = bc[-1].number_input("Min wall t (mm)", min_value=0.01, value=1.0, format="%.4g") * 1e-3
    submitted = st.form_submit_button("Run optimization", type="primary", icon=":material/tune:")

problem = optimize.OptimizationProblem(objective=objective, delta_allow=delta_mm * 1e-3, fos_min=fos_min,
                                       bounds=bounds, t_min=t_min, cost_per_kg=rate if rate > 0 else None)

if submitted:
    try:
        with st.spinner("Optimizing: every iterate is a full engine analysis…"):
            res = optimize.optimize(model, problem)
        st.session_state["optimization_last"] = {"model": model_json, "problem": problem, "result": res}
    except (ValueError, ModelError) as exc:
        st.error(f"**Cannot run the optimization.** {exc}")

last = st.session_state.get("optimization_last")
applied = st.session_state.get("optimization_applied")
if applied == model_json:
    st.success("The optimized design is now the current design. Every page (Analysis, Safety, Reports…) uses it.",
               icon=":material/check_circle:")
if last and last["model"] != model_json:
    if applied != model_json:
        st.info("The last optimization belongs to a different design. Run it again for the current one.",
                icon=":material/history:")
    last = None

# ------------------------------------------------------------------ results
if last:
    res: optimize.OptimizationResult = last["result"]
    prob: optimize.OptimizationProblem = last["problem"]
    st.markdown(f"### Result &nbsp; {pill(res.status)}", unsafe_allow_html=True)
    if res.infeasibility_reasons:  # the message repeats the reasons; they are listed individually below
        st.error("No section within the bounds satisfies every constraint.", icon=":material/block:")
    else:
        (st.success if res.status == "OPTIMIZED" else st.warning)(res.message)

    ini, opt = res.initial, res.optimized
    unit = {"volume": ("cm³", 1e6), "mass": ("kg", 1.0), "cost": ("₹", 1.0)}[prob.objective]
    obj_i = optimize.objective_value(model, prob, res.initial_dims) * unit[1]
    cols = st.columns(4)
    if opt is not None:
        obj_o = optimize.objective_value(model, prob, res.optimal_dims) * unit[1]
        cols[0].metric(f"{prob.objective.title()} ({unit[0]})", f"{obj_o:.4g}", f"{-res.reduction_pct:+.1f} %",
                       delta_color="inverse", help=f"Initial: {obj_i:.4g} {unit[0]}")
        cols[1].metric("Max deflection |v|", f"{opt.delta_max * 1e3:.4g} mm",
                       f"{(opt.delta_max - ini.delta_max) * 1e3:+.3g} mm", delta_color="inverse",
                       help=f"Limit {prob.delta_allow * 1e3:.4g} mm")
        cols[2].metric("Max stress |σ|", f"{opt.sigma_max / 1e6:.4g} MPa",
                       f"{(opt.sigma_max - ini.sigma_max) / 1e6:+.3g} MPa", delta_color="inverse")
        cols[3].metric("Yield FOS", f"{opt.fos:.3g}", f"{opt.fos - ini.fos:+.3g}", help=f"Required ≥ {prob.fos_min:g}")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("##### Initial vs optimized")
            st.dataframe(before_after_frame(ini, opt, ("Initial", "Optimized")), hide_index=True, width="stretch")
        with right:
            st.markdown("##### Cross-sections to scale")
            st.plotly_chart(section_overlay([(model.section, "Initial", COLORS["muted"]),
                                             (res.optimized_model.section, "Optimized", COLORS["cyan"])]),
                            theme=None, width="stretch")
    else:
        cols[0].metric(f"Initial {prob.objective} ({unit[0]})", f"{obj_i:.4g}")
        cols[1].metric("Initial |v|max", f"{ini.delta_max * 1e3:.4g} mm", help=f"Limit {prob.delta_allow * 1e3:.4g} mm")
        cols[2].metric("Initial |σ|max", f"{ini.sigma_max / 1e6:.4g} MPa")
        cols[3].metric("Initial FOS", f"{ini.fos:.3g}", help=f"Required ≥ {prob.fos_min:g}")
        if res.infeasibility_reasons:
            st.markdown("##### Why no design is feasible")
            for r in res.infeasibility_reasons:
                st.markdown(f"- {r}")
            st.caption("Widen the dimension bounds, relax δ_allow or FOS_min, or choose a stiffer / stronger material "
                       "in Design Studio.")

    st.markdown("##### Constraint verification" + (" (independent re-analysis)" if res.feasible
                                                   else " at the strongest section within the bounds"))
    show_checks(res.checks)
    if res.feasible:
        st.caption(f"SLSQP iterations (best start): {res.n_iterations}. SciPy convergence flag: {res.success}. "
                   "The PASS/FAIL column comes from a fresh analysis of the final design, not from the optimizer.")
        if st.button("Apply optimized design", type="primary", icon=":material/done_all:"):
            m = res.optimized_model.copy()
            m.name = f"{model.name or 'Design'} (optimized)"
            set_model(m)
            st.session_state["optimization_applied"] = m.to_json(indent=None)
            st.rerun()

# ------------------------------------------------------------------ Pareto
st.divider()
st.markdown("### Pareto trade-off: material vs stiffness vs strength")
st.caption("Each material is sampled at space-filling section sizes inside the bounds above, and every sample is "
           "fully analysed. A design is **Pareto-optimal** if no other sample is at once cheaper on the objective, "
           "stiffer and stronger (higher FOS).")
pc = st.columns([3, 1, 1])
mats = pc[0].multiselect("Materials", material_keys(), default=material_keys(),
                         format_func=lambda k: MATERIALS[k].name)
n_pts = pc[1].slider("Sizes per material", 6, 40, 15)
pc[2].markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
if pc[2].button("Compute Pareto front", icon=":material/scatter_plot:", disabled=not mats):
    try:
        with st.spinner("Analysing every sample…"):
            df = run_pareto(model_json, problem, tuple(mats), n_pts)
        st.session_state["optimization_pareto"] = {"model": model_json, "objective": problem.objective, "df": df}
    except (ValueError, ModelError) as exc:
        st.error(f"**Cannot compute the Pareto front.** {exc}")

par = st.session_state.get("optimization_pareto")
if par and par["model"] == model_json and not par["df"].empty:
    df, axis = par["df"], par["objective"]
    ax_unit = {"volume": ("Volume (cm³)", 1e6), "mass": ("Mass (kg)", 1.0), "cost": ("Cost (₹)", 1.0)}[axis]
    fig_k, fig_f = go.Figure(), go.Figure()
    for i, (name, grp) in enumerate(df.groupby("material", sort=False)):
        color = SERIES[i % len(SERIES)]
        for fig, ycol, yscale in ((fig_k, "stiffness", 1e-3), (fig_f, "fos", 1.0)):
            for on_front in (False, True):
                sub = grp[grp["pareto"] == on_front]
                if sub.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=sub[axis] * ax_unit[1], y=sub[ycol] * yscale, mode="markers",
                    name=name if on_front else f"{name} (dominated)", legendgroup=name, showlegend=on_front,
                    marker=dict(color=color, size=11 if on_front else 7, opacity=1.0 if on_front else 0.35,
                                symbol=["diamond" if f else "diamond-open" for f in sub["feasible"]] if on_front
                                else ["circle" if f else "circle-open" for f in sub["feasible"]],
                                line=dict(color=COLORS["text"], width=1) if on_front else None),
                    customdata=sub[["delta_max", "sigma_max", "fos", "feasible"]].to_numpy(),
                    hovertemplate=(f"<b>{name}</b><br>{ax_unit[0]} = %{{x:.4g}}<br>"
                                   "δmax = %{customdata[0]:.3e} m<br>σmax = %{customdata[1]:.4g} Pa<br>"
                                   "FOS = %{customdata[2]:.3g}<br>meets constraints: %{customdata[3]}<extra></extra>")))
    fig_k.update_layout(height=420, xaxis_title=ax_unit[0], yaxis_title="Stiffness k = total load / δmax (kN/m)",
                        xaxis_type="log", yaxis_type="log", title="Objective vs stiffness")
    fig_f.update_layout(height=420, xaxis_title=ax_unit[0], yaxis_title="Yield FOS", xaxis_type="log",
                        yaxis_type="log", title="Objective vs strength")
    fig_f.add_hline(y=problem.fos_min, line=dict(color=COLORS["amber"], dash="dash"),
                    annotation_text=f"FOS_min {problem.fos_min:g}", annotation_font_color=COLORS["amber"])
    a, b = st.columns(2)
    a.plotly_chart(fig_k, theme=None, width="stretch")
    b.plotly_chart(fig_f, theme=None, width="stretch")
    front = df[df["pareto"]]
    st.markdown(
        f"**Reading the trade-off.** Large diamonds are Pareto-optimal; open markers miss δ_allow or FOS_min. "
        f"{len(front)} of {len(df)} samples are non-dominated. Moving right along a material's front buys stiffness "
        f"and strength with more {axis}; at equal {axis}, a material with higher E is stiffer and one with a "
        f"higher σy/ρ is stronger per kilogram. Stiffness is k = Σ|load| / δmax for the actual loading.")
    with st.expander("Pareto-optimal samples"):
        show = front.drop(columns=["material_key", "pareto"]).copy()
        for k in model.section.dims():
            show[k] = (show[k] * 1e3).map(g)
        show = show.rename(columns={k: f"{k} (mm)" for k in model.section.dims()})
        show["delta_max"] = (show["delta_max"] * 1e3).map(g)
        show["sigma_max"] = (show["sigma_max"] / 1e6).map(g)
        show["volume"] = (show["volume"] * 1e6).map(g)
        show["stiffness"] = (show["stiffness"] / 1e3).map(g)
        show = show.rename(columns={"delta_max": "δmax (mm)", "sigma_max": "σmax (MPa)", "volume": "Volume (cm³)",
                                    "mass": "Mass (kg)", "cost": "Cost (₹)", "stiffness": "k (kN/m)", "fos": "FOS",
                                    "feasible": "Meets constraints"})
        st.dataframe(show.round(4), hide_index=True, width="stretch")
    st.caption("Each material uses its own approximate library ₹/kg for cost. All material values are approximate.")
elif par and par["model"] != model_json:
    st.info("The Pareto data belongs to a different design. Compute it again.", icon=":material/history:")
