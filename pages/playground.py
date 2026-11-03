"""Structural Playground (Module 13): drag-and-drop beam editor wired to the SmartBeam engine."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smartbeam import fem, safety, sections
from smartbeam.materials import MATERIALS
from smartbeam.model import BeamModel, ModelError, UnstableStructureError
from smartbeam.playground import convert
from smartbeam.playground.component import beam_canvas
from smartbeam.ui.state import analyze_model, get_model, set_model
from smartbeam.ui.theme import COLORS, badge, page_header
from smartbeam.units import from_si

ss = st.session_state
KEY = "playground_canvas"
STATUS_COLOR = {safety.STATUS_SAFE: COLORS["green"], safety.STATUS_WARNING: COLORS["amber"],
                safety.STATUS_YIELD: COLORS["red"], safety.STATUS_BUCKLING: COLORS["red"]}
MATERIAL_DICTS = {k: asdict(m) for k, m in MATERIALS.items()}


# --------------------------------------------------------------------------- session state + canvas callback
def _load_into_canvas(config: dict) -> None:
    """Replace the canvas contents from Python (file / shared design). The canvas adopts it on config_rev."""
    ss.playground_config = config
    ss.playground_rev += 1
    ss.playground_result = None
    ss.playground_error = None


if "playground_config" not in ss:
    ss.playground_config = convert.model_to_config(get_model())  # start from the shared design
    ss.playground_rev = 0
    ss.playground_seen = 0  # canvas edit counter last received
    ss.playground_nonce = None  # last canvas message handled
    ss.playground_result = None  # AnalysisResult of the last successful analysis
    ss.playground_error = None  # rejection message of the last analysis attempt


def _analyse(record: bool) -> None:
    try:
        model = convert.config_to_model(ss.playground_config)
        ss.playground_result = analyze_model(model, record=record)
        ss.playground_error = None
    except (ModelError, UnstableStructureError) as e:
        ss.playground_result = None
        ss.playground_error = str(e)


def _on_canvas() -> None:
    """Runs before the script whenever the canvas sends a value; each nonce is handled once."""
    action = convert.accept_message(ss, ss.get(KEY))
    if action in ("analyse", "live"):
        _analyse(record=action == "analyse")  # live runs don't flood the recent-analyses list
    elif action == "edit":
        ss.playground_error = None


def _current_model() -> BeamModel | None:
    try:
        return convert.config_to_model(ss.playground_config)
    except ModelError:
        return None


@st.cache_data(max_entries=32, show_spinner=False)
def _fem_check(model_json: str):
    return fem.solve(BeamModel.from_json(model_json), n_elements=40, theory="euler", frame=False)


@st.cache_data(max_entries=32, show_spinner=False)
def _safety(model_json: str):
    model = BeamModel.from_json(model_json)
    return safety.assess(model, analyze_model(model, record=False))


def _diagram(x, y, title, unit, color, fill=True, extra=None):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=2.2), name=title,
                               fill="tozeroy" if fill else None, hovertemplate=f"x = %{{x:.4g}} m<br>%{{y:.5g}} {unit}<extra></extra>"))
    for tr in extra or []:
        fig.add_trace(tr)
    i = int(np.argmax(np.abs(y)))
    up = y[i] >= 0
    side = "right" if x[i] < x[0] + 0.2 * (x[-1] - x[0]) else "left" if x[i] > x[0] + 0.8 * (x[-1] - x[0]) else "center"
    fig.add_trace(go.Scatter(x=[x[i]], y=[y[i]], mode="markers+text", marker=dict(color=COLORS["amber"], size=9),
                             text=[f"{y[i]:.4g} {unit}"], textposition=f"{'top' if up else 'bottom'} {side}",
                             showlegend=False, hoverinfo="skip"))
    lo, hi = min(float(np.min(y)), 0.0), max(float(np.max(y)), 0.0)
    pad = 0.22 * ((hi - lo) or 1.0)  # headroom so the max label is not clipped by the plot edge
    fig.update_yaxes(range=[lo - (0 if up else pad), hi + (pad if up else 0)])
    fig.update_layout(title=title, height=290, xaxis_title="x (m)", yaxis_title=unit, showlegend=bool(extra),
                      margin=dict(l=60, r=20, t=44, b=44), legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    fig.add_hline(y=0, line_color=COLORS["border"])
    return fig


# --------------------------------------------------------------------------- page
page_header("Structural Playground",
            "Drag supports and loads onto the beam, edit them in the properties panel, and analyse with the same "
            "engine every other page uses.")

model = _current_model()
result = ss.playground_result
current = bool(result is not None and model is not None
               and result.model.to_json(indent=None) == model.to_json(indent=None))
units = {**convert.DEFAULT_UNITS, **ss.playground_config.get("units", {})}

beam_canvas(
    config=ss.playground_config, config_rev=ss.playground_rev, seen_version=ss.playground_seen,
    results=convert.result_payload(result) if result is not None else None, results_current=current,
    error=ss.playground_error, presets=convert.preset_configs(units), materials=MATERIAL_DICTS,
    key=KEY, on_change=_on_canvas,
)

# --------------------------------------------------------------------------- actions row
c1, c2, c3 = st.columns([1.1, 1.1, 2.2])
with c1:
    if st.button("Open in 3D Visualization", icon=":material/view_in_ar:", type="primary", disabled=model is None,
                 width="stretch", help="Make this beam the current design and open the 3D view"):
        set_model(model)
        st.switch_page("pages/visualization_3d.py")
with c2:
    if st.button("Use as current design", icon=":material/save:", disabled=model is None, width="stretch",
                 help="Send this beam to Design Studio, Analysis, Safety and the other pages"):
        set_model(model)
        st.toast("Playground beam is now the current design.")
with c3:
    if st.button("Load current design into canvas", icon=":material/download:", width="stretch",
                 help="Replace the canvas with the design from Design Studio"):
        _load_into_canvas(convert.model_to_config(get_model(), units))
        st.rerun()

with st.expander("Configuration files  ·  saved_configs/", icon=":material/folder_open:"):
    f1, f2 = st.columns(2)
    with f1:
        name = st.text_input("File name", value=(ss.playground_config.get("name") or "playground_beam"),
                             key="playground_save_name")
        if st.button("Save configuration", icon=":material/save:"):
            try:
                path = convert.save_config(ss.playground_config, name)
                st.success(f"Saved `{path.relative_to(convert.SAVED_DIR.parent)}`")
            except (ModelError, ValueError) as e:
                st.error(f"Not saved: {e}")
    with f2:
        files = convert.list_configs()
        pick = st.selectbox("Saved configurations", files, index=None, placeholder="Choose a file…",
                            key="playground_load_name")
        if st.button("Load into canvas", icon=":material/upload_file:", disabled=pick is None):
            try:
                _load_into_canvas(convert.load_config(pick))
                st.rerun()
            except (ModelError, ValueError, KeyError) as e:
                st.error(f"Could not load {pick}: {e}")
    st.caption("Files use the SmartBeam JSON schema (SI units), the same format as the canvas' Save JSON / Load JSON.")

# --------------------------------------------------------------------------- results
if ss.playground_error:
    st.error(f"**Analysis rejected.** {ss.playground_error}", icon=":material/block:")
    st.stop()
if result is None:
    st.info("Build a beam on the canvas, then press **▶ Analyse Structure** (or enable **Live**) to compute reactions, "
            "diagrams, deflection and stresses.", icon=":material/touch_app:")
    st.stop()
if not current:
    st.warning("The canvas has changed since this analysis: the results below are for the previous configuration. "
               "Press **▶ Analyse Structure** to update.", icon=":material/history:")

m, r = result.model, result
fu, mu = units["force"], units["moment"]
FU, MU = (1e3 if fu == "kN" else 1.0), (1e3 if mu == "kN·m" else 1.0)
METHOD = {"analytical": "Analytical (statics + integration)", "fem-euler": "FEM · Euler–Bernoulli beam elements",
          "fem-frame": "FEM · planar frame elements (u, v, θ)"}

st.subheader("Results")
st.markdown(f"{badge(METHOD[r.method], COLORS['cyan'])} &nbsp; <span style='color:{COLORS['muted']}'>"
            f"{r.stability.message}</span>", unsafe_allow_html=True)
for w in r.warnings:
    st.caption(f"⚠ {w}")

report = _safety(m.to_json(indent=None))
fos = report.yield_result.fos
k = st.columns(5)
k[0].metric("Max deflection (mm)", f"{abs(r.v_max.value) * 1e3:.4g}", f"at x = {r.v_max.x:.4g} m", delta_color="off", delta_arrow="off")
k[1].metric(f"Max |M| ({mu})", f"{abs(r.M_max.value) / MU:.4g}", f"at x = {r.M_max.x:.4g} m", delta_color="off", delta_arrow="off")
k[2].metric(f"Max |V| ({fu})", f"{abs(r.V_max.value) / FU:.4g}", f"at x = {r.V_max.x:.4g} m", delta_color="off", delta_arrow="off")
k[3].metric("Max |σ| (MPa)", f"{abs(r.sigma_max.value) / 1e6:.4g}", f"{r.sigma_max.where} fibre, x = {r.sigma_max.x:.4g} m",
            delta_color="off", delta_arrow="off")
k[4].metric("Yield FOS", "∞" if not np.isfinite(fos) else f"{fos:.3g}", report.status, delta_color="off", delta_arrow="off")

t_diag, t_react, t_cmp, t_safe, t_theory = st.tabs(
    ["Diagrams", "Reactions & equilibrium", "FEM vs analytical", "Safety", "Theory"])

with t_diag:
    a, b = st.columns(2)
    a.plotly_chart(_diagram(r.x, r.V / FU, "Shear force V(x)", fu, COLORS["cyan"]), theme=None, width="stretch")
    b.plotly_chart(_diagram(r.x, r.M / MU, "Bending moment M(x) · sagging +", mu, COLORS["amber"]), theme=None,
                   width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(_diagram(r.x, r.v * 1e3, "Deflection v(x) · up +", "mm", COLORS["cyan"]), theme=None,
                   width="stretch")
    b.plotly_chart(_diagram(r.x, r.theta * 1e3, "Slope θ(x) · CCW +", "mrad", COLORS["violet"], fill=False), theme=None,
                   width="stretch")
    a, b = st.columns(2)
    bottom = go.Scatter(x=r.x, y=r.sigma_bottom / 1e6, mode="lines", name="bottom fibre",
                        line=dict(color=COLORS["blue"], width=2, dash="dot"))
    a.plotly_chart(_diagram(r.x, r.sigma_top / 1e6, "Bending stress σ (top / bottom) · tension +", "MPa",
                            COLORS["amber"], fill=False, extra=[bottom]), theme=None, width="stretch")
    eb = go.Scatter(x=r.x, y=r.eps_bottom * 1e6, mode="lines", name="bottom fibre",
                    line=dict(color=COLORS["blue"], width=2, dash="dot"))
    b.plotly_chart(_diagram(r.x, r.eps_top * 1e6, "Strain ε = σ/E (top / bottom)", "µε", COLORS["green"], fill=False,
                            extra=[eb]), theme=None, width="stretch")
    if np.any(np.abs(r.N) > 0):
        st.plotly_chart(_diagram(r.x, r.N / FU, "Axial force N(x) · tension +", fu, COLORS["green"]), theme=None,
                        width="stretch")

with t_react:
    st.dataframe(pd.DataFrame([{
        "Support": f"#{re.support_index + 1}", "Type": re.type, "x (m)": re.x,
        f"Fx ({fu})": re.Fx / FU, f"Fy ({fu})": re.Fy / FU, f"M ({mu}, CCW +)": re.M / MU} for re in r.reactions]),
        hide_index=True, width="stretch")
    st.markdown("**Equilibrium residuals** (loads + reactions; ≈ 0 means the solution balances)")
    st.dataframe(pd.DataFrame([{"Check": k_, "Residual": v_} for k_, v_ in r.equilibrium.items()]),
                 hide_index=True, width="stretch")
    if not r.stability.determinate:
        st.info(f"Statically indeterminate (degree {r.stability.degree}): the "
                f"{r.stability.required} equilibrium equations cannot give {r.stability.n_restraints} unknown "
                "reactions, so they come from the stiffness (FEM) solution with deformation compatibility.",
                icon=":material/info:")

with t_cmp:
    if r.method == "analytical":
        fr = _fem_check(m.to_json(indent=None))
        rows = [("Max deflection |v|", abs(r.v_max.value) * 1e3, abs(fr.v_max.value) * 1e3, "mm"),
                ("Max slope |θ|", abs(r.theta_max.value) * 1e3, abs(fr.theta_max.value) * 1e3, "mrad"),
                ("Max |M|", abs(r.M_max.value) / MU, abs(fr.M_max.value) / MU, mu),
                ("Max |V|", abs(r.V_max.value) / FU, abs(fr.V_max.value) / FU, fu),
                ("Max |σ|", abs(r.sigma_max.value) / 1e6, abs(fr.sigma_max.value) / 1e6, "MPa")]
        for ra, rf in zip(r.reactions, sorted(fr.reactions, key=lambda q: q.x)):
            rows.append((f"Reaction Fy @ x = {ra.x:g} m", ra.Fy / FU, rf.Fy / FU, fu))
            if ra.type == "fixed":
                rows.append((f"Reaction M @ x = {ra.x:g} m", ra.M / MU, rf.M / MU, mu))
        df = pd.DataFrame(rows, columns=["Quantity", "Analytical", "FEM (40 EB elements)", "Unit"])
        df["Difference %"] = [abs(a_ - f_) / abs(a_) * 100 if abs(a_) > 1e-12 else abs(f_) for a_, f_ in
                              zip(df["Analytical"], df["FEM (40 EB elements)"])]
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"Difference %": st.column_config.NumberColumn(format="%.2e")})
        fig = go.Figure([
            go.Scatter(x=r.x, y=r.v * 1e3, name="Analytical", line=dict(color=COLORS["cyan"], width=2.5)),
            go.Scatter(x=fr.mesh.nodes, y=fr.v_nodes * 1e3, name="FEM nodes", mode="markers",
                       marker=dict(color=COLORS["amber"], size=7, symbol="circle-open", line=dict(width=2)))])
        fig.update_layout(title="Deflection: analytical curve vs FEM nodal values", height=320, xaxis_title="x (m)",
                          yaxis_title="v (mm)")
        st.plotly_chart(fig, theme=None, width="stretch")
        st.caption(f"FEM: {fr.mesh.n_elements} elements, {fr.n_dof} DOF ({fr.n_free} free), condition number "
                   f"{fr.condition_number:.3g}. Hermite elements are exact at the nodes for these loads, so the "
                   "differences are round-off.")
        if r.textbook is not None:
            tb = r.textbook
            st.markdown(f"**Textbook check · {tb.case}**")
            st.dataframe(pd.DataFrame([{"Quantity": q, "Formula": tb.formulas[q], "Textbook (SI)": tb.textbook[q],
                                        "Engine (SI)": tb.numerical.get(q), "Rel. error": tb.rel_error.get(q)}
                                       for q in tb.textbook]), hide_index=True, width="stretch",
                         column_config={"Rel. error": st.column_config.NumberColumn(format="%.2e")})
    else:
        why = ("an axial / inclined load is present, so the beam is solved with planar frame elements"
               if r.method == "fem-frame" else "the beam is statically indeterminate, so statics alone is insufficient")
        st.info(f"No closed-form comparison: {why}. The FEM solution is the primary result here.",
                icon=":material/info:")
        fr = r.fem
        if fr is not None:
            st.caption(f"FEM: {fr.mesh.n_elements} {fr.element} elements, {fr.n_dof} DOF ({fr.n_free} free), "
                       f"condition number {fr.condition_number:.3g}.")

with t_safe:
    st.markdown(badge(report.status, STATUS_COLOR.get(report.status)), unsafe_allow_html=True)
    for msg in report.messages:
        st.markdown(f"- {msg}")
    p = sections.properties(m.section, m.material.nu)
    st.caption(f"Section: A = {from_si(p.A, 'mm²'):.4g} mm², I = {from_si(p.I, 'mm⁴'):.4g} mm⁴, "
               f"Z = {from_si(p.Z, 'mm³'):.4g} mm³ · {m.material.name}, E = {m.material.E / 1e9:.4g} GPa, "
               f"σy = {m.material.yield_strength / 1e6:.4g} MPa.")

with t_theory:
    st.markdown("Sign convention: x from the left end, y up; sagging moment positive; tension positive.")
    st.latex(r"\frac{dV}{dx} = q(x), \qquad \frac{dM}{dx} = V(x)")
    st.latex(r"EI\,\frac{d^2 v}{dx^2} = M(x), \qquad \theta = \frac{dv}{dx}")
    st.latex(r"\sigma(x,y) = \frac{N}{A} - \frac{M\,y}{I}, \qquad \varepsilon = \frac{\sigma}{E}")
    st.markdown("Statically determinate transverse beams are solved from equilibrium "
                r"($\sum F_y = 0,\ \sum M = 0$) and integrated twice for slope and deflection. Indeterminate beams use "
                "Euler–Bernoulli beam elements; axial or inclined loads use planar frame elements, "
                r"$K\,U = F$, with reactions from $R = K U - F$.")
