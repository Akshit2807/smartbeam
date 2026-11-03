"""report.py: PDF builds (all sections / omitted sections / lab), CSV exports match the engine arrays."""
import io
import re

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from smartbeam import advisor, fem, lab, optimize, report, safety
from smartbeam.analytical import analyze
from smartbeam.metrics import check
from smartbeam.model import BeamModel, DistributedLoad, MomentLoad, PointLoad, Section, Support


def _pages(pdf: bytes) -> int:
    return len(re.findall(rb"/Type /Page\b", pdf))


@pytest.fixture
def frame_model(steel):
    """Indeterminate, inclined load, partial UDL and a moment: exercises the FEM-frame route."""
    return BeamModel(L=3.0, section=Section.hollow_circ(0.06, 0.05), material=steel,
                     supports=[Support("fixed", 0.0), Support("roller", 3.0)],
                     loads=[DistributedLoad(0.5, 2.5, 4000, 1000), PointLoad(1.5, 3000, -60), MomentLoad.cw(2.0, 500)],
                     name="Frame <case> & co")  # markup characters must be escaped, not break the PDF


def test_full_report_and_omitted_sections(cantilever_end_load):
    m = cantilever_end_load
    r = analyze(m)
    full = report.build_pdf(m, r, fem.solve(m, 40), fem.convergence_study(m), safety.assess(m, r),
                            optimize.optimize(m, optimize.OptimizationProblem(delta_allow=m.L / 250)),
                            advisor.advise(m, delta_allow=1e-3),
                            report.ReportMeta(project="Test", team="A, B"))
    bare = report.build_pdf(m)  # no results -> model sections only, nothing invented
    analysis_only = report.build_pdf(m, r)
    for pdf in (full, bare, analysis_only):
        assert pdf.startswith(b"%PDF") and pdf.rstrip().endswith(b"%%EOF")
    assert _pages(bare) < _pages(analysis_only) < _pages(full)


def test_frame_route_and_infeasible_optimization_report(frame_model):
    r = analyze(frame_model)
    assert r.method == "fem-frame"
    opt = optimize.optimize(frame_model, optimize.OptimizationProblem(
        delta_allow=1e-6, bounds={"D": (0.02, 0.03), "d": (0.01, 0.02)}))
    assert opt.status == "INFEASIBLE"
    pdf = report.build_pdf(frame_model, r, r.fem, fem.convergence_study(frame_model), safety.assess(frame_model, r),
                           opt, advisor.advise(frame_model))
    assert pdf.startswith(b"%PDF") and _pages(pdf) >= 8


def test_analysis_csv_matches_result(ss_udl):
    r = analyze(ss_udl)
    df = pd.read_csv(io.StringIO(report.analysis_csv(r)))
    assert list(df.columns) == ["x_m", "V_N", "M_Nm", "N_N", "theta_rad", "v_m", "sigma_top_Pa", "sigma_bottom_Pa",
                                "eps_top", "eps_bottom"]
    assert len(df) == len(r.x)
    np.testing.assert_allclose(df["M_Nm"], r.M, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(df["v_m"], r.v, rtol=1e-12, atol=1e-18)


def test_fem_csv_has_reactions_at_supports(ss_central_load):
    f = fem.solve(ss_central_load, 8)
    df = pd.read_csv(io.StringIO(report.fem_csv(f)))
    assert len(df) == f.mesh.n_nodes
    sup = df.dropna(subset=["Ry_N"])
    assert sorted(sup["x_m"]) == [0.0, ss_central_load.L]
    assert sup["Ry_N"].sum() == pytest.approx(1000.0, rel=1e-9)  # P = 1 kN shared by both supports


def test_summary_csv_values(cantilever_end_load):
    r = analyze(cantilever_end_load)
    df = pd.read_csv(io.StringIO(report.summary_csv(cantilever_end_load, r, safety.assess(cantilever_end_load, r))))
    val = dict(zip(df["quantity"], df["value"]))
    assert float(val["v_max"]) == pytest.approx(r.v_max.value)
    assert float(val["support1_fixed_x0_M"]) == pytest.approx(1000.0 * 2.0)  # +PL, L = 2 m
    assert val["safety_status"] == "SAFE"


def test_format_check_converts_lengths_to_mm():
    assert report.format_check(check("Max deflection |v|max (m)", 0.003, 0.004)) == ("Max deflection |v|max", 3.0, 4.0, "mm")
    assert report.format_check(check("Yield FOS", 2.0, 1.5, ">=")) == ("Yield FOS", 2.0, 1.5, "")


def test_lab_report_builders():
    fig = Figure()
    fig.subplots().plot([1, 2, 3], [1, 8, 27])
    pdf = report.build_lab_report(
        "Height vs stiffness", "Show that delta ~ h^-3", {"L": "1 m", "P": "1 kN"},
        [r"\delta = \dfrac{PL^3}{3EI}", "plain text equation k = P / delta", r"\notacommand{x}"],
        pd.DataFrame({"h_mm": [20, 40], "delta_mm": [8.0, 1.0]}),
        pd.DataFrame({f"col_{i}": np.arange(3.0) for i in range(15)}),  # wide table -> split into chunks
        [fig], ["delta falls as h^-3"])
    assert pdf.startswith(b"%PDF")
    res = lab.run_experiment("height_vs_stiffness")
    assert report.lab_report_pdf(res).startswith(b"%PDF")
