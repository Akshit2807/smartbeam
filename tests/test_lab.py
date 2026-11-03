"""Virtual lab: experiments reproduce known scaling laws, challenges grade honestly, sensitivity is exact."""
import math

import numpy as np
import pytest

from smartbeam import lab
from smartbeam.model import PointLoad, Section


def test_loglog_slope():
    x = np.array([1.0, 2.0, 4.0])
    assert lab.loglog_slope(x, 5 * x**3) == pytest.approx(3.0)
    assert lab.loglog_slope(x, -(x ** -2.0)) == pytest.approx(-2.0)  # sign ignored


def test_scale_parameter_moves_positions_with_span():
    m = lab.build_beam(2.0, Section.rect(0.03, 0.06), lab.default_base_model().material, "overhanging", "point",
                       1000.0, 1.0)
    s = lab.scale_parameter(m, "L", 1.5)
    assert s.L == 3.0 and s.loads[0].x == 1.5 and s.supports[1].x == pytest.approx(0.75 * 3.0)
    assert m.L == 2.0  # original untouched
    assert lab.scale_parameter(m, "load", 2).loads[0].magnitude == 2000.0
    with pytest.raises(ValueError):
        lab.scale_parameter(m, "D", 2)


@pytest.mark.parametrize("key", list(lab.EXPERIMENTS))
def test_every_experiment_runs(key):
    r = lab.run_experiment(key)
    assert len(r.table) >= 2 and r.conclusion and r.equations
    assert r.x in r.table and all(c in r.table for c in r.y)


def test_experiment_scaling_laws():
    assert lab.run_experiment("height_vs_stiffness").findings["slope_delta_h"] == pytest.approx(-3, abs=1e-6)
    assert lab.run_experiment("length_vs_deflection").findings["slope_delta_L"] == pytest.approx(3, abs=1e-6)
    f = lab.run_experiment("steel_vs_aluminium").findings
    assert f["delta_ratio"] == pytest.approx(f["E_ratio"], rel=1e-9)
    f = lab.run_experiment("point_vs_distributed").findings
    assert (f["cant_delta_ratio"], f["ss_delta_ratio"]) == pytest.approx((3 / 8, 5 / 8), rel=1e-6)
    f = lab.run_experiment("cantilever_vs_ss").findings
    assert (f["cant_ss_point"], f["cant_ss_udl"], f["ff_ss_point"]) == pytest.approx((16, 9.6, 0.25), rel=1e-6)
    assert lab.run_experiment("eb_vs_timoshenko").findings["slope"] == pytest.approx(-2, abs=1e-3)
    assert lab.run_experiment("mesh_convergence").findings["order_M_fe"] == pytest.approx(2, abs=0.05)
    f = lab.run_experiment("volume_vs_performance").findings
    assert f["uniform scaling: delta"] == pytest.approx(-2, abs=1e-6)
    assert f["depth only: delta"] == pytest.approx(-3, abs=1e-6)


def test_load_position_matches_closed_form():
    r = lab.run_experiment("load_position")
    assert r.findings["M_rel_error"] < 1e-9
    assert r.table.a_over_L[r.table.M_max_Nm.idxmax()] == pytest.approx(0.5)


def test_challenges_grade_and_reference_passes():
    assert len(lab.CHALLENGES) >= 5
    for ch in lab.CHALLENGES:
        g = lab.grade_challenge(ch, ch.base_model)
        assert 0 <= g.score <= 100
        assert g.passed == all(c.satisfied for c in g.checks)
        ref = g.reference
        assert ref.status == "OPTIMIZED"
        best = lab.grade_challenge(ch, ref.optimized_model)  # the verified optimum must pass with full score
        assert best.passed and best.score == pytest.approx(100, abs=0.1)


def test_challenge_cannot_change_the_problem():
    ch = lab.CHALLENGE_BY_KEY["shelf_bracket"]
    cheat = ch.base_model.copy()
    cheat.loads = [PointLoad(0.25, 1.0, -90.0)]  # submitted with a tiny load: ignored
    assert lab.challenge_design(ch, cheat).loads[0].magnitude == 400.0
    with pytest.raises(ValueError):
        tube = ch.base_model.copy()
        tube.section = Section.rect(0.05, 0.05)
        lab.grade_challenge(lab.CHALLENGE_BY_KEY["hollow_tube"], tube)


def test_failed_constraint_feedback():
    ch = lab.CHALLENGE_BY_KEY["pedestrian_plank"]
    g = lab.grade_challenge(ch, ch.base_model)  # 30 mm plank deflects too much
    assert not g.passed and any("Deflection" in f for f in g.feedback)


def test_tornado_elasticities_are_exponents():
    t = lab.tornado(lab.default_base_model(), 10)
    e = {(r.param, r.output): r.elasticity for r in t.itertuples()}
    assert e[("L", "delta_max")] == pytest.approx(3)
    assert e[("h", "delta_max")] == pytest.approx(-3)
    assert e[("h", "sigma_max")] == pytest.approx(-2)
    assert e[("E", "delta_max")] == pytest.approx(-1)
    assert e[("E", "sigma_max")] == pytest.approx(0, abs=1e-12)  # determinate: stress independent of E


def test_sensitivity_sweep_skips_invalid_hollow():
    m = lab.default_base_model()
    m.section = Section.hollow_circ(0.05, 0.04)
    df = lab.sensitivity_sweep(m, "D", np.array([0.5, 1.0, 2.0]))
    assert list(df.factor) == [1.0, 2.0]  # D = 25 mm < d is skipped
    assert "D" not in set(lab.tornado(m, 50, lab.sensitivity_params(m)).param)


def test_numerical_comparison_reference_only_when_independent():
    df = lab.numerical_comparison(lab.default_base_model(), (1, 2, 4))
    assert (df.err_v_pct < 1e-8).all()
    m = lab.build_beam(1.0, Section.rect(0.03, 0.06), lab.default_base_model().material, "fixed-fixed", "udl", 1000.0)
    assert lab.numerical_comparison(m, (2, 4)).err_v_pct.isna().all()  # FEM route: no independent reference


def test_lab_pdf_and_csv():
    from smartbeam.ui.lab_plots import experiment_figure, lab_pdf
    r = lab.run_experiment("height_vs_stiffness")
    experiment_figure(r)
    assert lab_pdf(r)[:4] == b"%PDF"
    assert "Max deflection" in lab.display_table(r.table).to_csv()
    ch = lab.CHALLENGES[0]
    rep = lab.challenge_report(ch, lab.grade_challenge(ch, ch.base_model))
    assert lab_pdf(rep)[:4] == b"%PDF" and math.isfinite(rep.table.utilisation.iloc[0])
