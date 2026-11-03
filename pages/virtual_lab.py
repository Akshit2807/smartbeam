"""Engineering Virtual Laboratory (REQUIREMENTS Module 11). OWNER: lab engineer.

Explorer · guided experiments · design challenges · analytical vs numerical · sensitivity · lab reports.
All numbers come from smartbeam.lab / analytical / fem / optimize; this page only collects inputs and draws.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from smartbeam import lab, sections
from smartbeam.materials import MATERIALS, get_material
from smartbeam.metrics import evaluate
from smartbeam.model import BeamModel, ModelError, PointLoad, Section, UnstableStructureError
from smartbeam.report import ReportMeta
from smartbeam.ui.lab_plots import experiment_figure, lab_pdf
from smartbeam.ui.plots_core import (METHOD_LABEL, SHAPE_LABEL, beam_schematic, card, dims_text, fem_solve,
                                     field_trace, model_caption)
from smartbeam.ui.state import analyze_model, get_model, set_model
from smartbeam.ui.theme import COLORS, badge, page_header

SS = st.session_state
K = "virtual_lab_"
SS.setdefault(K + "results", {})  # experiment key -> (signature, ExperimentResult)
SS.setdefault(K + "grades", {})  # challenge key -> (design dict, ChallengeGrade)
SS.setdefault(K + "reports", {})  # report id -> ExperimentResult (for the Lab reports tab)

MAT_KEYS = list(MATERIALS)
SUPPORT_LABEL = {"cantilever": "Cantilever (fixed at x = 0)", "simply supported": "Simply supported (pin + roller)",
                 "propped cantilever": "Propped cantilever (fixed + roller)", "fixed-fixed": "Fixed–fixed",
                 "overhanging": "Overhanging (pin 0, roller 0.75 L)"}
LOAD_LABEL = {"point": "Point load (down)", "udl": "Full-span UDL (down)", "triangular": "Triangular 0 → w at L (down)",
              "moment": "Applied moment (CCW +)"}


def short(mat_name: str) -> str:
    return mat_name.split(" (")[0]


def meta() -> ReportMeta:
    return ReportMeta(project=SS.get(K + "meta_project", ""), team=SS.get(K + "meta_team") or "[Team members]",
                      institute=SS.get(K + "meta_institute") or "[Institute]",
                      guide=SS.get(K + "meta_guide") or "[Guide]", date=dt.date.today().isoformat())


def report_buttons(res: lab.ExperimentResult, key: str) -> None:
    """PDF (report.build_lab_report, generated on click) + CSV of the result; registers it in the Reports tab."""
    SS[K + "reports"][res.key] = res
    c1, c2, _ = st.columns([1, 1, 2])
    m = meta()  # read session state now; the callable runs later, outside the script run
    c1.download_button("Generate lab report (PDF)", data=lambda: lab_pdf(res, m), file_name=f"lab_{res.key}.pdf",
                       mime="application/pdf", icon=":material/picture_as_pdf:", key=f"{K}pdf_{key}",
                       on_click="ignore", type="primary")
    c2.download_button("Results (CSV)", data=lab.display_table(res.table).to_csv(index=False),
                       file_name=f"lab_{res.key}.csv", mime="text/csv", icon=":material/table:",
                       key=f"{K}csv_{key}", on_click="ignore")


def show_table(df: pd.DataFrame) -> None:
    st.dataframe(lab.display_table(df), hide_index=True, width="stretch")


def pct_delta(new: float, old: float) -> str | None:
    if not (math.isfinite(new) and math.isfinite(old)) or old == 0:
        return None
    d = (new / old - 1) * 100
    return None if abs(d) < 0.005 else f"{d:+.2f}%"


def material_picker(key: str, default: str = "structural_steel", editable_props: bool = True):
    """Library material select (+ optional editable E, σy). Returns a Material copy."""
    mk = st.selectbox("Material", MAT_KEYS, index=MAT_KEYS.index(default), format_func=lambda k: MATERIALS[k].name,
                      key=f"{key}_mat")
    mat = get_material(mk)
    if editable_props:
        a, b = st.columns(2)
        mat.E = a.number_input("E (GPa)", 1.0, 1000.0, mat.E / 1e9, 1.0, format="%.4g", key=f"{key}_E_{mk}") * 1e9
        mat.yield_strength = b.number_input("Yield σy (MPa)", 1.0, 3000.0, mat.yield_strength / 1e6, 5.0,
                                            format="%.4g", key=f"{key}_sy_{mk}") * 1e6
        if (mat.E, mat.yield_strength) != (MATERIALS[mk].E, MATERIALS[mk].yield_strength):
            mat.name = f"{mat.name} (edited)"
            mat.approximate = False
    return mat


def base_picker(key: str) -> BeamModel:
    """Which beam an experiment / study starts from."""
    opts = ["Lab reference beam", "Current design", "Explorer beam"]
    src = st.segmented_control("Base beam", opts, default=opts[0], key=f"{key}_src", required=True)
    if src == "Current design":
        return get_model()
    if src == "Explorer beam" and K + "explorer_model" in SS:
        return BeamModel.from_dict(SS[K + "explorer_model"])
    return lab.default_base_model()


# =========================================================================== page
page_header("Engineering Virtual Laboratory",
            "Experiment with real beam mechanics: every curve, ratio and conclusion below is computed live by the "
            "SmartBeam engine.")
tabs = st.tabs([":material/tune: Live explorer", ":material/science: Guided experiments",
                ":material/emoji_events: Design challenges", ":material/compare_arrows: Analytical vs numerical",
                ":material/insights: Sensitivity", ":material/description: Lab reports"])


# --------------------------------------------------------------------------- 1. live explorer
def _set_baseline(d):
    SS[K + "baseline"] = d


def _push_shared(d):
    set_model(BeamModel.from_dict(d))
    st.toast("Explorer beam is now the current design.", icon=":material/check_circle:")


def explorer_results(model: BeamModel) -> None:
    try:
        res = analyze_model(model, record=False)
    except (ModelError, UnstableStructureError) as exc:
        st.error(f"**Cannot analyse this beam.** {exc}")
        return
    base_model = BeamModel.from_dict(SS[K + "baseline"])
    met = evaluate(model, res)
    try:
        bmet = evaluate(base_model, analyze_model(base_model, record=False))
    except (ModelError, UnstableStructureError):
        bmet = None
    W = lab.total_load(model)
    k = W / met.delta_max if W and met.delta_max else math.nan
    bk = lab.total_load(base_model) / bmet.delta_max if bmet and bmet.delta_max else math.nan
    cols = st.columns(3) + st.columns(3)  # 2 rows of 3: six metrics do not fit one row of this panel
    rows = [("Max deflection", met.delta_max * 1e3, "mm", bmet and bmet.delta_max * 1e3, "inverse"),
            ("Max stress", met.sigma_max / 1e6, "MPa", bmet and bmet.sigma_max / 1e6, "inverse"),
            ("Max moment", met.M_max / 1e3, "kN·m", bmet and bmet.M_max / 1e3, "off"),
            ("Yield FOS", met.fos, "", bmet and bmet.fos, "normal"),
            ("Stiffness W/δ", k / 1e3, "N/mm", bk / 1e3, "normal"),
            ("Mass", met.mass, "kg", bmet and bmet.mass, "inverse")]
    for c, (label, v, u, b0, dc) in zip(cols, rows):
        c.metric(label, f"{v:.4g} {u}".strip() if math.isfinite(v) else "—",
                 pct_delta(v, b0) if b0 is not None else None, delta_color=dc,
                 help=f"Δ vs baseline ({base_model.name or 'baseline'})")
    status = "SAFE" if met.fos >= 1.5 else ("WARNING" if met.fos >= 1 else "YIELD PREDICTED")
    st.markdown(f"{badge(status)} &nbsp; <span style='color:{COLORS['muted']}'>Yield FOS {met.fos:.3g} "
                f"(warning below 1.5) · solved by {METHOD_LABEL.get(res.method, res.method)} · "
                f"deltas vs baseline: {base_model.name or 'baseline'}</span>", unsafe_allow_html=True)
    st.plotly_chart(beam_schematic(model, height=190), theme=None, key=K + "ex_schem")

    fig = make_subplots(2, 2, subplot_titles=("Shear force V (kN)", "Bending moment M (kN·m)",
                                              "Deflection v (mm)", "Fibre stress σ (MPa)"),
                        vertical_spacing=0.16, horizontal_spacing=0.09)
    fig.add_trace(field_trace(res.x, res.V / 1e3, "V", COLORS["amber"], "kN"), 1, 1)
    fig.add_trace(field_trace(res.x, res.M / 1e3, "M", COLORS["violet"], "kN·m"), 1, 2)
    fig.add_trace(field_trace(res.x, res.v * 1e3, "v", COLORS["cyan"], "mm"), 2, 1)
    fig.add_trace(field_trace(res.x, res.sigma_top / 1e6, "σ top", COLORS["red"], "MPa", fill=False), 2, 2)
    fig.add_trace(field_trace(res.x, res.sigma_bottom / 1e6, "σ bottom", COLORS["blue"], "MPa", fill=False), 2, 2)
    fig.update_xaxes(title_text="x (m)", row=2)
    fig.update_layout(height=520, showlegend=False, margin=dict(t=40, b=40))
    fig.update_annotations(font=dict(color=COLORS["muted"], size=13))
    st.plotly_chart(fig, theme=None, key=K + "ex_fig")

    r1, r2 = st.columns([1.3, 1])
    with r1:
        st.markdown("**Support reactions** (on the beam, +y up, M CCW +)")
        st.dataframe(pd.DataFrame([{"Support": f"{r.type} @ {r.x:g} m", "Fy (kN)": r.Fy / 1e3,
                                    "M (kN·m)": r.M / 1e3} for r in res.reactions]),
                     hide_index=True, width="stretch")
        for w in res.warnings:
            st.caption(f"⚠ {w}")
    with r2:
        st.markdown("**Baseline**")
        st.caption(model_caption(base_model))
        st.button("Set this beam as baseline", on_click=_set_baseline, args=(model.to_dict(),),
                  icon=":material/push_pin:", width="stretch", key=K + "ex_setbase")
        st.button("Baseline = current design", on_click=_set_baseline, args=(get_model().to_dict(),),
                  icon=":material/restart_alt:", width="stretch", key=K + "ex_resetbase")
        st.button("Use as current design", on_click=_push_shared, args=(model.to_dict(),),
                  icon=":material/send:", width="stretch", key=K + "ex_push",
                  help="Replaces the shared design used by Analysis, FEM, 3D, Safety… pages.")


with tabs[0]:
    ctrl, out = st.columns([1, 2.3], gap="large")
    with ctrl, st.container(border=True):
        st.markdown("**Beam**")
        support = st.selectbox("Supports", lab.SUPPORT_CONFIGS, format_func=SUPPORT_LABEL.get, key=K + "ex_sup")
        L = st.slider("Span L (m)", 0.2, 6.0, 1.0, 0.05, key=K + "ex_L")
        shape = st.segmented_control("Section", list(SHAPE_LABEL), default="rect", format_func=SHAPE_LABEL.get,
                                     key=K + "ex_shape", required=True)
        if shape == "rect":
            b = st.slider("Width b (mm)", 5, 300, 30, key=K + "ex_b")
            h = st.slider("Depth h (mm)", 5, 500, 60, key=K + "ex_h")
            section = Section.rect(b / 1e3, h / 1e3)
        elif shape == "solid_circ":
            section = Section.solid_circ(st.slider("Diameter d (mm)", 5, 300, 50, key=K + "ex_d") / 1e3)
        else:
            D = st.slider("Outer diameter D (mm)", 10, 400, 60, key=K + "ex_D")
            r = st.slider("Bore ratio d/D", 0.0, 0.95, 0.8, 0.01, key=K + "ex_r")
            st.caption(f"Inner diameter d = {r * D:.4g} mm, wall {D * (1 - r) / 2:.4g} mm")
            section = Section.hollow_circ(D / 1e3, r * D / 1e3) if r > 0 else Section.solid_circ(D / 1e3)
        st.markdown("**Material**")
        material = material_picker(K + "ex")
        st.markdown("**Load**")
        ltype = st.selectbox("Load type", lab.LOAD_TYPES, format_func=LOAD_LABEL.get, key=K + "ex_lt")
        unit = {"point": "kN", "udl": "kN/m", "triangular": "kN/m (peak)", "moment": "kN·m"}[ltype]
        mag = st.slider(f"Magnitude ({unit})", 0.1, 50.0, 1.0, 0.1, key=K + "ex_mag") * 1e3
        pos = None
        if ltype in ("point", "moment"):
            pos = st.slider("Position a/L", 0.0, 1.0, 1.0, 0.01, key=K + "ex_pos") * L
            st.caption(f"x = {pos:.4g} m")
    model = lab.build_beam(L, section, material, support, ltype, mag, pos, name="Explorer beam")
    SS[K + "explorer_model"] = model.to_dict()
    SS.setdefault(K + "baseline", get_model().to_dict())

    with out:
        explorer_results(model)


# --------------------------------------------------------------------------- 2. guided experiments
def experiment_controls(key: str, base: BeamModel):
    """Experiment-specific inputs -> (base model, swept values)."""
    s = base.section
    W_default = lab.total_load(base) / 1e3 or 1.0

    def W_input():
        return st.number_input("Total load W (kN)", 0.01, 1000.0, float(W_default), 0.1, format="%.4g",
                               key=f"{K}{key}_W") * 1e3

    def n_input(lo, hi, v):
        return st.slider("Number of runs", lo, hi, v, key=f"{K}{key}_n")

    if key == "height_vs_stiffness":
        h0 = (s.h if s.shape == "rect" else s.depth) * 1e3
        lo, hi = st.slider("Depth range h (mm)", 5.0, 500.0, (max(5.0, float(round(h0 * 0.5))), min(500.0, float(round(h0 * 3)))),
                           1.0, key=f"{K}{key}_rng")
        return base, np.linspace(lo, hi, n_input(4, 20, 8)) / 1e3
    if key == "length_vs_deflection":
        lo, hi = st.slider("Span range L (m)", 0.1, 10.0, (max(0.1, float(round(base.L * 0.5, 2))), min(10.0, float(round(base.L * 2, 2)))),
                           0.05, key=f"{K}{key}_rng")
        return base, np.linspace(lo, hi, n_input(4, 20, 6))
    if key == "steel_vs_aluminium":
        a, b = st.columns(2)
        ma = a.selectbox("Material A", MAT_KEYS, 0, format_func=lambda m: MATERIALS[m].name, key=f"{K}{key}_a")
        mb = b.selectbox("Material B", MAT_KEYS, MAT_KEYS.index("aluminium_6061_t6"),
                         format_func=lambda m: MATERIALS[m].name, key=f"{K}{key}_b")
        return base, [ma, mb]
    if key in ("rect_vs_circular", "solid_vs_hollow"):
        A = st.number_input("Cross-section area A (mm²)", 10.0, 1e6, float(round(sections.area(s) * 1e6, 1)), 50.0,
                            format="%.6g", key=f"{K}{key}_A") / 1e6
        base = base.copy()
        base.section = Section.rect(math.sqrt(A / 2), math.sqrt(2 * A))  # h = 2b, area A
        if key == "rect_vs_circular":
            return base, None
        rmax = st.slider("Largest bore ratio d/D", 0.5, 0.95, 0.9, 0.05, key=f"{K}{key}_r")
        return base, list(np.round(np.linspace(0.0, rmax, n_input(4, 15, 7)), 4))
    if key in ("point_vs_distributed", "load_position", "cantilever_vs_ss", "eb_vs_timoshenko", "mesh_convergence"):
        W = W_input()
        base = base.copy()
        base.loads = [PointLoad(base.L, W, -90.0)]  # experiments read the total load W from the base beam
        if key == "load_position":
            return base, list(np.round(np.linspace(0.05, 0.95, n_input(5, 37, 19)), 4))
        if key == "cantilever_vs_ss":
            sups = st.multiselect("Support layouts", lab.SUPPORT_CONFIGS[:4], list(lab.SUPPORT_CONFIGS[:4]),
                                  format_func=SUPPORT_LABEL.get, key=f"{K}{key}_sups")
            return base, sups or ["cantilever"]
        if key == "eb_vs_timoshenko":
            lo, hi = st.slider("Span-to-depth range L/h", 1.0, 100.0, (2.0, 30.0), 0.5, key=f"{K}{key}_rng")
            return base, list(np.round(np.geomspace(lo, hi, n_input(4, 20, 10)), 3))
        if key == "mesh_convergence":
            nmax = st.select_slider("Finest mesh (elements)", [8, 16, 32, 64, 128], 64, key=f"{K}{key}_nmax")
            return base, [2 ** i for i in range(int(math.log2(nmax)) + 1)]
        return base, None
    if key == "volume_vs_performance":
        lo, hi = st.slider("Section scale factor range s", 0.3, 3.0, (0.6, 1.8), 0.05, key=f"{K}{key}_rng")
        return base, list(np.round(np.linspace(lo, hi, n_input(4, 15, 7)), 4))
    return base, None


with tabs[1]:
    exp_key = st.selectbox("Experiment", list(lab.EXPERIMENTS), format_func=lambda k: lab.EXPERIMENTS[k].title,
                           key=K + "exp_sel")
    exp = lab.EXPERIMENTS[exp_key]
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        card("Objective", exp.objective)
        st.write("")
        card("Hypothesis", exp.hypothesis, COLORS["amber"])
        st.write("")
        with st.container(border=True):
            st.markdown("**Parameters**")
            base = base_picker(K + "exp")
            base, values = experiment_controls(exp_key, base)
            st.caption(f"Base: {model_caption(base)}. The experiment changes only what it studies "
                       "(supports and loads are set by the experiment where stated).")
            sig = (exp_key, base.to_json(indent=None), repr(None if values is None else [str(v) for v in values]))
            if st.button("Run experiment", type="primary", icon=":material/play_arrow:", width="stretch",
                         key=K + "exp_run"):
                try:
                    with st.spinner("Running sweep through the engine…"):
                        SS[K + "results"][exp_key] = (sig, lab.run_experiment(exp_key, base, values))
                except (ModelError, UnstableStructureError, ValueError) as exc:
                    st.error(f"Experiment could not run: {exc}")
    with right:
        stored = SS[K + "results"].get(exp_key)
        if stored is None:
            st.info("Set the parameters and press **Run experiment**. Each run is a real parameter sweep through "
                    "the analysis engine.", icon=":material/science:")
        else:
            run_sig, r = stored
            if run_sig != sig:
                st.warning("Parameters changed since this run — press **Run experiment** to update.",
                           icon=":material/update:")
            st.plotly_chart(experiment_figure(r), theme=None, key=K + "exp_fig")
            st.markdown(f"<div class='sb-card' style='border-left:3px solid {COLORS['green']}'><b>Conclusion "
                        f"(computed from this run)</b><br>{r.conclusion}</div>", unsafe_allow_html=True)
            st.write("")
            with st.expander("Results table", icon=":material/table:"):
                show_table(r.table)
            with st.expander("Theory and governing equations", icon=":material/function:"):
                st.write(r.explanation)
                for eq in r.equations:
                    st.latex(eq)
            report_buttons(r, "exp_" + exp_key)


# --------------------------------------------------------------------------- 3. design challenges
with tabs[2]:
    ch = lab.CHALLENGE_BY_KEY[st.selectbox("Challenge", [c.key for c in lab.CHALLENGES],
                                           format_func=lambda k: lab.CHALLENGE_BY_KEY[k].title, key=K + "ch_sel")]
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        card(ch.title, ch.brief, COLORS["amber"])
        chips = []
        if ch.delta_allow is not None:
            chips.append(f"δ ≤ {ch.delta_allow * 1e3:.4g} mm")
        if ch.min_fos is not None:
            chips.append(f"FOS ≥ {ch.min_fos:g}")
        if ch.max_mass is not None:
            chips.append(f"mass ≤ {ch.max_mass:g} kg")
        if ch.beat_material:
            chips.append(f"lighter than optimum {short(MATERIALS[ch.beat_material].name)}")
        if ch.min_wall:
            chips.append(f"wall ≥ {ch.min_wall * 1e3:g} mm")
        st.markdown(" ".join(badge(c, COLORS["cyan"]) for c in chips), unsafe_allow_html=True)
        st.plotly_chart(beam_schematic(ch.base_model, height=180), theme=None, key=K + "ch_schem")
        st.caption(f"Span {ch.base_model.L:g} m · supports and loads are fixed by the problem. "
                   f"{lab.SCORE_RULE}")
        with st.container(border=True):
            st.markdown("**Your design**")
            shape = st.segmented_control("Section shape", list(ch.shapes), default=ch.shapes[0],
                                         format_func=SHAPE_LABEL.get, key=f"{K}ch_{ch.key}_shape", required=True)
            bounds = lab.challenge_bounds(ch, shape)
            start = ch.base_model.section.dims() if ch.base_model.section.shape == shape else {
                kk: math.sqrt(lo * hi) for kk, (lo, hi) in bounds.items()}
            names = {"b": "Width b", "h": "Depth h", "d": "Diameter d" if shape == "solid_circ" else "Inner diameter d",
                     "D": "Outer diameter D"}
            dcols = st.columns(len(start))
            dims = {kk: dcols[i].number_input(f"{names[kk]} (mm)", 0.1, 5000.0, round(start[kk] * 1e3, 2), 0.5,
                                              format="%.4g", key=f"{K}ch_{ch.key}_{shape}_{kk}") / 1e3
                    for i, kk in enumerate(start)}
            if "material" in ch.editable:
                mat = material_picker(f"{K}ch_{ch.key}", editable_props=False)
            else:
                mat = ch.base_model.material
                st.caption(f"Material fixed: {mat.name}")
            submit = st.button("Submit design for grading", type="primary", icon=":material/grading:",
                               width="stretch", key=f"{K}ch_{ch.key}_submit")
        if submit:
            design = ch.base_model.copy()
            design.section, design.material = Section(shape, **dims), mat
            try:
                with st.spinner("Analysing your design and computing the reference optimum…"):
                    SS[K + "grades"][ch.key] = (design.to_dict(), lab.grade_challenge(ch, design))
            except (ModelError, UnstableStructureError, ValueError) as exc:
                st.error(f"Design rejected: {exc}")
    with right:
        graded = SS[K + "grades"].get(ch.key)
        if graded is None:
            st.info("Enter a section (and material, where allowed) and press **Submit**. The engine grades every "
                    "constraint; the reference optimum is revealed only after you submit.", icon=":material/flag:")
        else:
            ddict, g = graded
            m = g.metrics
            st.markdown(f"{badge('PASS' if g.passed else 'FAIL')} &nbsp; <span style='color:{COLORS['muted']}'>"
                        f"{SHAPE_LABEL[m.shape]} {dims_text(BeamModel.from_dict(ddict).section)} · {m.material}"
                        "</span>", unsafe_allow_html=True)
            c = st.columns(5)
            c[0].metric("Score", f"{g.score:.1f} / 100")
            c[1].metric("Deflection", f"{m.delta_max * 1e3:.4g} mm")
            c[2].metric("Max stress", f"{m.sigma_max / 1e6:.4g} MPa")
            c[3].metric("Yield FOS", f"{m.fos:.3g}")
            c[4].metric("Mass", f"{m.mass:.4g} kg")
            rep = lab.challenge_report(ch, g)
            tbl = rep.table.rename(columns={"constraint": "Constraint", "value": "Value", "limit": "Limit",
                                            "unit": "Unit", "sense": "Rule", "result": "Result",
                                            "margin_pct": "Margin (%)", "utilisation": "Utilisation"})
            st.dataframe(tbl, hide_index=True, width="stretch",
                         column_config={"Utilisation": st.column_config.ProgressColumn(
                             "Utilisation", min_value=0.0, max_value=1.5, format="%.3f")})
            for f in g.feedback:
                st.markdown(f"- {f}")
            ref = g.reference
            with st.expander("Reference optimum (revealed after submission)", icon=":material/workspace_premium:"):
                if ref is not None and ref.optimized is not None:
                    o = ref.optimized
                    st.markdown(f"**{SHAPE_LABEL[o.shape]} {dims_text(Section(o.shape, **o.dims))}** in "
                                f"{o.material} — mass **{o.mass:.4g} kg**, δ = {o.delta_max * 1e3:.4g} mm, "
                                f"σ = {o.sigma_max / 1e6:.4g} MPa, FOS = {o.fos:.3g}.")
                    st.caption(ref.message + " Local and lateral-torsional buckling are not modelled, so the optimum "
                                             "can be a thin or narrow section; the challenge bounds keep it practical.")
                elif ref is not None:
                    st.warning(ref.message)
                    for reason in ref.infeasibility_reasons:
                        st.caption(reason)
            report_buttons(rep, "ch_" + ch.key)


# --------------------------------------------------------------------------- 4. analytical vs numerical
@st.cache_data(max_entries=32, show_spinner=False)
def _numerical(model_json: str, n_list: tuple, theory: str) -> pd.DataFrame:
    return lab.numerical_comparison(BeamModel.from_json(model_json), n_list, theory)


def numerical_tab() -> None:
    c0, c1, c2, c3 = st.columns([1.4, 1.2, 1, 1])
    with c0:
        src = st.segmented_control("Beam", ["Current design", "Explorer beam"], default="Current design",
                                   key=K + "num_src", required=True)
    nmodel = get_model() if src == "Current design" else BeamModel.from_dict(SS[K + "explorer_model"])
    n_el = c1.slider("Elements n", 1, 64, 4, key=K + "num_n")
    theory = c2.segmented_control("FE theory", ["euler", "timoshenko"], default="euler", key=K + "num_th",
                                  required=True, format_func={"euler": "Euler–Bernoulli", "timoshenko": "Timoshenko"}.get)
    nmax = c3.select_slider("Convergence up to", [8, 16, 32, 64, 128], 64, key=K + "num_nmax")
    st.caption(model_caption(nmodel))
    try:
        ares = analyze_model(nmodel, record=False)
        fres = fem_solve(nmodel.to_json(indent=None), n_el, theory)
    except (ModelError, UnstableStructureError) as exc:
        st.error(f"**Cannot analyse this beam.** {exc}")
        return
    independent = ares.method == "analytical"
    ref_name = "Analytical (statics + integration)" if independent else "Engine FEM (40 elements)"
    if not independent:
        st.info("This beam is statically indeterminate or carries axial load, so the engine itself solves it by FEM: "
                "there is no independent closed-form reference. Differences below are against the engine's "
                "40-element solution and only show mesh dependence.", icon=":material/info:")
    if theory == "timoshenko":
        st.caption("Timoshenko elements include shear deformation; the reference is Euler–Bernoulli, so part of "
                   "the difference is the theory (shear), not discretisation error.")
    mc = st.columns(4)
    for col, (label, a, f, sc, u) in zip(mc, [
            ("Max deflection", ares.v_max.value, fres.v_max.value, 1e3, "mm"),
            ("Max moment", ares.M_max.value, fres.M_max.value, 1e-3, "kN·m"),
            ("Max stress", ares.sigma_max.value, fres.sigma_max.value, 1e-6, "MPa"),
            ("Max slope", ares.theta_max.value, fres.theta_max.value, 1e3, "mrad")]):
        diff = abs(abs(f) - abs(a)) / abs(a) * 100 if a else 0.0
        col.metric(f"{label} · FEM n={fres.mesh.n_elements}", f"{abs(f) * sc:.5g} {u}", f"{diff:.3g}% vs ref",
                   delta_color="off", help=f"{ref_name}: {abs(a) * sc:.6g} {u}")
    g1, g2 = st.columns(2)
    for col, (title, ax, fx, nodes, scale, unit, color) in zip((g1, g2), [
            ("Deflection v", ares.v, fres.v, fres.v_nodes, 1e3, "mm", COLORS["cyan"]),
            ("Bending moment M", ares.M, fres.M, None, 1e-3, "kN·m", COLORS["violet"])]):
        fig = go.Figure([go.Scatter(x=ares.x, y=ax * scale, name=ref_name, line=dict(color=COLORS["muted"], width=5)),
                         go.Scatter(x=fres.x, y=fx * scale, name=f"FEM ({fres.mesh.n_elements} el.)",
                                    line=dict(color=color, width=2, dash="dash"))])
        if nodes is not None:
            fig.add_trace(go.Scatter(x=fres.mesh.nodes, y=nodes * scale, mode="markers", name="FE nodes",
                                     marker=dict(size=8, color=COLORS["amber"])))
        fig.update_layout(title=title, height=330, xaxis_title="x (m)", yaxis_title=f"{title} ({unit})",
                          legend=dict(orientation="h", y=-0.25))
        col.plotly_chart(fig, theme=None, key=K + "num_" + title)

    with st.expander("Nodal comparison table", icon=":material/table:", expanded=fres.mesh.n_nodes <= 17):
        xn = fres.mesh.nodes
        v_ref, th_ref = np.interp(xn, ares.x, ares.v), np.interp(xn, ares.x, ares.theta)
        st.dataframe(pd.DataFrame({
            "Node": np.arange(1, len(xn) + 1), "x (m)": xn, "v FEM (mm)": fres.v_nodes * 1e3, "v ref (mm)": v_ref * 1e3,
            "Δv (mm)": (fres.v_nodes - v_ref) * 1e3, "θ FEM (mrad)": fres.theta_nodes * 1e3,
            "θ ref (mrad)": th_ref * 1e3, "Δθ (mrad)": (fres.theta_nodes - th_ref) * 1e3}),
            hide_index=True, width="stretch")

    st.markdown("##### Mesh convergence")
    n_list = tuple(2 ** i for i in range(int(math.log2(nmax)) + 1))
    conv = _numerical(nmodel.to_json(indent=None), n_list, theory)
    fig = go.Figure()
    if independent:
        for col_, name, color in (("err_v_pct", "Max deflection", COLORS["cyan"]),
                                  ("err_M_pct", "Max moment (equilibrium)", COLORS["violet"]),
                                  ("err_sigma_pct", "Max stress", COLORS["red"]),
                                  ("err_M_fe_pct", "Moment from curvature EI·v″", COLORS["amber"])):
            y = conv[col_].where(conv[col_] > 0)
            fig.add_trace(go.Scatter(x=conv.n_elements, y=y, name=name, mode="lines+markers",
                                     line=dict(color=color, width=2.2)))
        fig.update_layout(yaxis_title="error vs analytical (%)")
    else:
        for col_, name, color in (("v_fem", "Max deflection", COLORS["cyan"]), ("M_fem", "Max moment", COLORS["violet"]),
                                  ("sigma_fem", "Max stress", COLORS["red"])):
            ch_ = (conv[col_] / conv[col_].iloc[-1] - 1).abs() * 100
            fig.add_trace(go.Scatter(x=conv.n_elements, y=ch_.where(ch_ > 0), name=name, mode="lines+markers",
                                     line=dict(color=color, width=2.2)))
        fig.update_layout(yaxis_title="change vs finest mesh (%)")
    fig.update_xaxes(type="log", title="number of elements n")
    fig.update_yaxes(type="log", exponentformat="power")
    fig.update_layout(height=360, legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, theme=None, key=K + "num_conv")
    st.caption("Hermite beam elements are exact at the nodes for prismatic beams with consistent loads, so nodal "
               "deflection/moment errors sit at round-off (points at exactly zero are not drawn on the log axis). The "
               "moment taken from the curvature of the cubic field (EI·v″) is the quantity that genuinely converges "
               "under distributed loads. Meshes can be finer than requested because nodes are forced at supports "
               "and loads.")
    with st.expander("Convergence data", icon=":material/table:"):
        st.dataframe(conv, hide_index=True, width="stretch")


with tabs[3]:
    numerical_tab()


# --------------------------------------------------------------------------- 5. sensitivity
@st.cache_data(max_entries=32, show_spinner=False)
def _tornado(model_json: str, pct: float) -> pd.DataFrame:
    m = BeamModel.from_json(model_json)
    return lab.tornado(m, pct, lab.sensitivity_params(m))


@st.cache_data(max_entries=32, show_spinner=False)
def _sweep(model_json: str, param: str, factors: tuple) -> pd.DataFrame:
    return lab.sensitivity_sweep(BeamModel.from_json(model_json), param, np.array(factors))


def sensitivity_tab() -> None:
    a, b, c = st.columns([1.4, 1, 1])
    with a:
        smodel = base_picker(K + "sens")
    pct = b.slider("Parameter change ± (%)", 1, 50, 10, key=K + "sens_pct")
    out_key = c.selectbox("Tornado output", list(lab.OUTPUTS), format_func=lab.OUTPUTS.get, key=K + "sens_out")
    st.caption(model_caption(smodel))
    try:
        tor = _tornado(smodel.to_json(indent=None), float(pct))
    except (ModelError, UnstableStructureError, ValueError) as exc:
        st.error(f"**Cannot run the sensitivity study.** {exc}")
        return
    skipped = [lab.PARAM_LABELS[p] for p in lab.sensitivity_params(smodel) if p not in set(tor.param)]
    if skipped:
        st.caption(f"Left out (±{pct}% gives an invalid section): {', '.join(skipped)}.")
    t1, t2 = st.columns([1.1, 1])
    with t1:
        d = tor[tor.output == out_key].copy()
        d["span"] = (d.high_pct - d.low_pct).abs()
        d = d.sort_values("span")
        labels = [lab.PARAM_LABELS[p] for p in d.param]
        fig = go.Figure([
            go.Bar(y=labels, x=d.low_pct, orientation="h", name=f"−{pct}%", marker_color=COLORS["blue"],
                   hovertemplate="%{y}: %{x:+.3g}%<extra>−" + str(pct) + "%</extra>"),
            go.Bar(y=labels, x=d.high_pct, orientation="h", name=f"+{pct}%", marker_color=COLORS["amber"],
                   hovertemplate="%{y}: %{x:+.3g}%<extra>+" + str(pct) + "%</extra>")])
        fig.update_layout(barmode="overlay", title=f"Tornado — {lab.OUTPUTS[out_key]}", height=360,
                          xaxis_title=f"change in {lab.OUTPUTS[out_key].lower()} (%)",
                          legend=dict(orientation="h", y=-0.2))
        fig.add_vline(x=0, line=dict(color=COLORS["border"]))
        st.plotly_chart(fig, theme=None, key=K + "sens_tornado")
    with t2:
        piv = tor.pivot(index="param", columns="output", values="elasticity").reindex(
            index=list(dict.fromkeys(tor.param)), columns=[o for o in lab.OUTPUTS if o in set(tor.output)])
        z = piv.to_numpy(float)
        fig = go.Figure(go.Heatmap(z=z, x=[lab.OUTPUTS[o] for o in piv.columns],
                                   y=[lab.PARAM_LABELS[p] for p in piv.index], zmid=0,
                                   colorscale=[[0, COLORS["blue"]], [0.5, COLORS["panel_2"]], [1, COLORS["red"]]],
                                   text=np.vectorize(lambda v: f"{v:+.2f}")(z), texttemplate="%{text}",
                                   hovertemplate="%{y} → %{x}: %{z:+.3f}<extra></extra>", colorbar=dict(title="n")))
        fig.update_layout(title="Normalized sensitivity (elasticity, % per %)", height=360,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, theme=None, key=K + "sens_heat")

    st.markdown("##### One-at-a-time sweep")
    s1, s2 = st.columns([1, 2])
    params = lab.sensitivity_params(smodel)
    sp = s1.selectbox("Parameter", params, format_func=lab.PARAM_LABELS.get, key=K + "sens_param")
    lo, hi = s1.slider("Factor range", 0.25, 4.0, (0.5, 2.0), 0.05, key=K + "sens_rng")
    factors = tuple(np.unique(np.r_[np.geomspace(lo, hi, 17), 1.0]))
    sw = _sweep(smodel.to_json(indent=None), sp, factors)
    with s2:
        if sw.empty or not (sw.factor == 1.0).any():
            st.warning("No valid models in this range (e.g. inner diameter ≥ outer diameter).")
        else:
            y0 = sw[sw.factor == 1.0].iloc[0]
            fig = go.Figure()
            for i, (o, name) in enumerate(lab.OUTPUTS.items()):
                ratio = sw[o] / y0[o]
                if not np.isfinite(ratio).any() or y0[o] == 0:
                    continue
                n = lab.loglog_slope(sw.factor, ratio)
                fig.add_trace(go.Scatter(x=sw.factor, y=ratio, mode="lines+markers", name=f"{name} · slope {n:.3f}",
                                         line=dict(width=2.2, color=[COLORS[k] for k in ("cyan", "red", "violet",
                                                                                         "green", "amber")][i])))
            fig.update_xaxes(type="log", title=f"{lab.PARAM_LABELS[sp]} / base value")
            fig.update_yaxes(type="log", title="output / base value")
            fig.update_layout(height=360, legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig, theme=None, key=K + "sens_sweep")
    with st.expander("How to read this", icon=":material/school:"):
        st.write("Each parameter is changed on its own while everything else is held fixed, and the engine re-analyses "
                 "the beam. On log–log axes a power law Y ∝ Xⁿ is a straight line of slope n, so the slopes in the "
                 "legend are the exponents: for a cantilever with an end load δ ∝ L³, δ ∝ h⁻³, δ ∝ 1/E, σ ∝ h⁻².")
        st.latex(r"S = \frac{\partial Y / Y}{\partial X / X} \approx \frac{\ln(Y^+/Y^-)}{\ln(X^+/X^-)}")
        st.latex(r"\delta = \frac{P L^3}{3 E I},\quad I = \frac{b h^3}{12},\quad \sigma = \frac{M\,y}{I}")
        st.write("Stiffness k = W/δ is the total transverse load per unit peak deflection. Stress in a statically "
                 "determinate beam does not depend on E, so its E-elasticity is zero.")
    report_buttons(lab.sensitivity_report(smodel, tor, pct), "sens")


with tabs[4]:
    sensitivity_tab()


# --------------------------------------------------------------------------- 6. lab reports
with tabs[5]:
    st.markdown("Every experiment, challenge submission and sensitivity study you run is collected here. Each PDF "
                "contains the objective, initial conditions, governing equations, parameter changes, results table, "
                "graph and the computed conclusions; the CSV holds the full results table.")
    with st.container(border=True):
        st.markdown("**Report details** (editable placeholders — left blank, the report shows the placeholder)")
        m1, m2, m3, m4 = st.columns(4)
        m1.text_input("Project", key=K + "meta_project")
        m2.text_input("Team members", key=K + "meta_team", placeholder="[Team members]")
        m3.text_input("Institute", key=K + "meta_institute", placeholder="[Institute]")
        m4.text_input("Guide", key=K + "meta_guide", placeholder="[Guide]")
    reports = SS[K + "reports"]
    if not reports:
        st.info("No lab results yet. Run a guided experiment or submit a challenge design.", icon=":material/info:")
    for rid, rr in reports.items():
        with st.container(border=True):
            st.markdown(f"**{rr.title}**")
            st.caption(rr.conclusion[:300] + ("…" if len(rr.conclusion) > 300 else ""))
            report_buttons(rr, "all_" + rid)
