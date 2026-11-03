"""Module 12 — Engineering report generator: ReportLab PDF with Matplotlib figures, plus CSV exports."""
import datetime as dt
import re

import streamlit as st

from smartbeam import report, safety
from smartbeam.ui.design_tools import pill, run_advise
from smartbeam.ui.plots_core import METHOD_LABEL, beam_schematic, fem_convergence, fem_solve, model_caption, require_result
from smartbeam.ui.state import get_model
from smartbeam.ui.theme import page_header

page_header("Reports",
            "Download a complete engineering report (PDF) of the current design with embedded diagrams, plus the "
            "numerical data as CSV. Every number comes from the engine; sections you leave out are omitted, "
            "never filled with placeholders.")

model = get_model()
st.markdown(model_caption(model))
result = require_result(model)
model_json = model.to_json(indent=None)
slug = re.sub(r"[^A-Za-z0-9]+", "_", model.name or "beam").strip("_").lower() or "beam"
st.plotly_chart(beam_schematic(model, height=200), theme=None, width="stretch")

# ------------------------------------------------------------------ CSV (instant)
fem_res = result.fem if result.fem is not None else fem_solve(model_json, 40)
st.markdown("### Numerical data (CSV)")
c = st.columns(3)
c[0].download_button("Field arrays: x, V, M, N, θ, v, σ, ε", report.analysis_csv(result), f"{slug}_fields.csv",
                     "text/csv", icon=":material/table:", width="stretch",
                     help=f"{len(result.x)} rows, SI units. Positions repeat at concentrated loads (left then right "
                          "value).")
c[1].download_button("Summary: reactions, maxima, properties", report.summary_csv(model, result), f"{slug}_summary.csv",
                     "text/csv", icon=":material/summarize:", width="stretch")
c[2].download_button(f"FEM nodal results ({fem_res.mesh.n_nodes} nodes)", report.fem_csv(fem_res),
                     f"{slug}_fem_nodes.csv", "text/csv", icon=":material/grid_on:", width="stretch",
                     help="node, x, u, v, θ and reactions at restrained nodes.")

# ------------------------------------------------------------------ PDF
st.markdown("### Engineering report (PDF)")
opt_last = st.session_state.get("optimization_last")
opt_ready = bool(opt_last and opt_last["model"] == model_json)
da0, fos0 = st.session_state.get("advisor_criteria", (model.L / 250, 1.5))

with st.form("reports_form"):
    st.markdown("##### Title page")
    t = st.columns(3)
    title = t[0].text_input("Report title", "SmartBeam 2.0 Engineering Report")
    project = t[1].text_input("Project", model.name or "")
    date = t[2].date_input("Date", dt.date.today())
    t = st.columns(3)
    team = t[0].text_input("Team members", "[Team members]", help="Placeholder until you enter real names.")
    institute = t[1].text_input("Institute", "[Institute]")
    guide = t[2].text_input("Guide", "[Guide]")

    st.markdown("##### Sections")
    s = st.columns(3)
    inc_fem = s[0].checkbox("FEM results and analytical-vs-numerical comparison", True)
    inc_conv = s[0].checkbox("Mesh convergence study (2–32 elements)", True)
    inc_safety = s[1].checkbox("Failure prediction and FOS", True)
    inc_adv = s[1].checkbox("Design advisor recommendations", True)
    inc_opt = s[2].checkbox("Optimization results", opt_ready, disabled=not opt_ready,
                            help="Uses the last optimization run for this exact design." if opt_ready else
                            "Run the Optimization page for the current design first.")
    st.markdown("##### Design criteria (safety and advisor sections)")
    k = st.columns(2)
    delta_mm = k[0].number_input("Allowable deflection (mm)", min_value=1e-4, value=float(f"{da0 * 1e3:.4g}"),
                                 format="%.4g")
    min_fos = k[1].number_input("Minimum FOS", min_value=1.0, value=float(fos0), step=0.1, format="%.2f")
    build = st.form_submit_button("Build PDF report", type="primary", icon=":material/picture_as_pdf:")

if build:
    meta = report.ReportMeta(title=title.strip() or "SmartBeam 2.0 Engineering Report", project=project, team=team,
                             institute=institute, guide=guide, date=date.isoformat())
    try:
        with st.spinner("Rendering figures and typesetting the report…"):
            pdf = report.build_pdf(
                model, analysis=result,
                fem=fem_res if inc_fem else None,
                convergence=fem_convergence(model_json) if inc_conv else None,
                safety=safety.assess(model, result, min_fos) if inc_safety else None,
                optimization=opt_last["result"] if inc_opt and opt_ready else None,
                advisor=run_advise(model_json, delta_mm * 1e-3, min_fos) if inc_adv else None,
                meta=meta)
        included = ["model, loads, supports, section", "reactions, SFD/BMD, stress, strain, deflection"]
        included += [n for n, on in (("FEM", inc_fem), ("convergence", inc_conv), ("safety", inc_safety),
                                     ("advisor", inc_adv), ("optimization", inc_opt and opt_ready)) if on]
        st.session_state["reports_pdf"] = {"model": model_json, "pdf": pdf, "included": included}
    except Exception as exc:  # report a typesetting failure instead of a blank page; engine errors stop earlier
        st.error(f"**Report generation failed.** {type(exc).__name__}: {exc}")

done = st.session_state.get("reports_pdf")
if done and done["model"] == model_json:
    a, b = st.columns([1, 2], vertical_alignment="center")
    a.download_button("Download PDF report", done["pdf"], f"{slug}_report.pdf", "application/pdf", type="primary",
                      icon=":material/download:", width="stretch")
    b.markdown(f"{pill('READY')} &nbsp; {len(done['pdf']) / 1024:.0f} KB · includes {', '.join(done['included'])}",
               unsafe_allow_html=True)
elif done:
    st.info("The last PDF was built for a different design. Build it again for the current one.",
            icon=":material/history:")

with st.expander("What the report contains", icon=":material/list:"):
    st.markdown(
        f"- **Title page**: your title, project, team, institute, guide and date, the key results and contents.\n"
        f"- **Model**: geometry and configuration, material (approximate values flagged), loading table, supports "
        f"and stability check, section properties.\n"
        f"- **Analysis** ({METHOD_LABEL.get(result.method, result.method)}): reactions with equilibrium residuals; "
        f"SFD, BMD, stress (along x and across the depth), strain, slope and deflection, each as a figure with its "
        f"maxima; textbook verification when the beam is one of the four classical cases.\n"
        f"- **Optional**: FEM results with nodal table and analytical-vs-FEM comparison; mesh convergence table and "
        f"plot; yield, von Mises and buckling FOS with the yield-is-not-fracture disclaimer; optimization (initial vs "
        f"optimized, constraint verification, infeasibility reasons); advisor recommendations answering the five "
        f"questions.\n"
        f"- **Assumptions and limitations** plus any solver warnings for this model.")
