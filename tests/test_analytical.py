"""Analytical engine: textbook benchmarks, hand-calculated statics, closed-form deflections, routing."""
import numpy as np
import pytest

from smartbeam import sections
from smartbeam.analytical import (analyze, internal_forces, reactions, sample_points, slope_deflection,
                                  textbook_check)
from smartbeam.materials import get_material
from smartbeam.model import (BeamModel, DistributedLoad, ModelError, MomentLoad, PointLoad, Section, Support,
                             UnstableStructureError)

STEEL = get_material("structural_steel")
SEC = Section.rect(0.030, 0.060)
EI = STEEL.E * sections.second_moment(SEC)


def beam(L, supports, loads):
    return BeamModel(L, SEC, STEEL, supports, loads)


@pytest.mark.parametrize("fixture, case", [("cantilever_end_load", "cantilever-end-load"),
                                           ("cantilever_udl", "cantilever-udl"),
                                           ("ss_central_load", "ss-central-load"), ("ss_udl", "ss-udl")])
def test_textbook_benchmarks(request, fixture, case):
    r = analyze(request.getfixturevalue(fixture))
    assert r.method == "analytical" and r.textbook is not None and r.textbook.case == case
    for k, err in r.textbook.rel_error.items():
        assert err < 1e-3, (k, err)  # brief: < 0.1 %
    assert abs(r.equilibrium["sum_Fy"]) < 1e-9 and abs(r.equilibrium["sum_M0"]) < 1e-9


def test_benchmark_signs(cantilever_end_load, ss_central_load):
    c = analyze(cantilever_end_load)
    P, L = 1000.0, 2.0
    assert c.reactions[0].Fy == pytest.approx(P) and c.reactions[0].M == pytest.approx(P * L)
    assert c.M_max.value == pytest.approx(-P * L) and c.M_max.x == 0.0  # hogging at the wall
    assert c.v[-1] == pytest.approx(-P * L**3 / (3 * EI), rel=1e-9)
    assert c.theta[-1] == pytest.approx(-P * L**2 / (2 * EI), rel=1e-9)
    assert c.sigma_top[1] > 0 > c.sigma_bottom[1]  # hogging: top in tension
    s = analyze(ss_central_load)
    assert [rr.Fy for rr in s.reactions] == pytest.approx([P / 2, P / 2])
    assert s.M_max.value == pytest.approx(P * L / 4) and s.M_max.x == pytest.approx(L / 2)
    assert s.v_max.value == pytest.approx(-P * L**3 / (48 * EI), rel=1e-9)
    assert s.sigma_tension_max.where == "bottom"
    assert s.eps_bottom == pytest.approx(s.sigma_bottom / STEEL.E)


def overhang_model():
    """L = 6 m, pin at 1, roller at 5; 2 kN/m UDL on [0, 3]; 3 kN down at 6; 4 kN.m CW at 4."""
    return beam(6.0, [Support("pin", 1.0), Support("roller", 5.0)],
                [DistributedLoad(0.0, 3.0, 2000.0, 2000.0), PointLoad(6.0, 3000.0), MomentLoad.cw(4.0, 4000.0)])


def test_overhang_partial_udl_moment_hand_calc():
    m = overhang_model()
    r = analyze(m)
    # Hand calc, moments about the pin: 4 R_B - 6000*0.5 - 3000*5 - 4000 = 0 -> R_B = 5500, R_A = 9000 - 5500.
    assert [rr.Fy for rr in r.reactions] == pytest.approx([3500.0, 5500.0])
    for k in ("sum_Fx", "sum_Fy", "sum_M0"):
        assert abs(r.equilibrium[k]) < 1e-9
    assert r.equilibrium["dM_dx_minus_V"] < 1e-6 and r.equilibrium["dV_dx_minus_q"] < 1e-6
    V, M, _ = internal_forces(m, r.reactions, np.array([1.0, 1.0, 4.0, 4.0, 5.0, 6.0, 6.0]))
    # left/right of pin: -2000, -2000 + 3500; M(1) = -2000*0.5
    assert V[:2] == pytest.approx([-2000.0, 1500.0]) and M[0] == pytest.approx(-1000.0)
    # CW moment at 4 makes M jump UP by 4000; M(4-) = 3500*3 - 6000*2.5 = -4500
    assert M[2:4] == pytest.approx([-4500.0, -500.0])
    assert M[4] == pytest.approx(-3000.0)  # from the right: -3000 * 1
    assert V[5:] == pytest.approx([3000.0, 0.0]) and M[5:] == pytest.approx([0.0, 0.0], abs=1e-9)  # closure
    # diagrams close at both free ends, deflection zero at both supports
    assert abs(r.V[-1]) < 1e-9 and abs(r.M[-1]) < 1e-9
    assert np.interp([1.0, 5.0], r.x, r.v) == pytest.approx([0.0, 0.0], abs=1e-15)


def test_closed_form_deflections():
    L, P, w0, C = 2.0, 1000.0, 600.0, 250.0
    # SS, off-centre load at a: deflection under the load = -P a^2 b^2 / (3 EI L)
    a = 0.5
    r = analyze(beam(L, [Support("pin", 0.0), Support("roller", L)], [PointLoad(a, P)]))
    assert textbook_check(r.model) is None  # not a textbook case: no forced formula
    assert np.interp(a, r.x, r.v) == pytest.approx(-P * a**2 * (L - a) ** 2 / (3 * EI * L), rel=1e-9)
    # Cantilever, triangular load (w0 at the wall, 0 at tip): tip deflection w0 L^4 / (30 EI)
    r = analyze(beam(L, [Support("fixed", 0.0)], [DistributedLoad(0.0, L, w0, 0.0)]))
    assert r.v[-1] == pytest.approx(-w0 * L**4 / (30 * EI), rel=1e-9)
    # ... with the peak at the tip instead: 11 w0 L^4 / (120 EI)
    r = analyze(beam(L, [Support("fixed", 0.0)], [DistributedLoad(0.0, L, 0.0, w0)]))
    assert r.v[-1] == pytest.approx(-11 * w0 * L**4 / (120 * EI), rel=1e-9)
    # Cantilever, CCW tip moment C: wall reaction -C, so M = +C (sagging) everywhere; v(L) = C L^2 / (2EI) up
    r = analyze(beam(L, [Support("fixed", 0.0)], [MomentLoad.ccw(L, C)]))
    assert r.M[2:-2] == pytest.approx(np.full(len(r.M) - 4, C))
    assert r.v[-1] == pytest.approx(C * L**2 / (2 * EI), rel=1e-9)
    # Overhang: span s, overhang a, tip load P -> tip deflection P a^2 (s + a) / (3EI)
    s, a = 1.5, 0.5
    r = analyze(beam(s + a, [Support("pin", 0.0), Support("roller", s)], [PointLoad(s + a, P)]))
    assert r.v[-1] == pytest.approx(-P * a**2 * (s + a) / (3 * EI), rel=1e-9)


def test_mirrored_cantilever_and_textbook_guard():
    L, P = 2.0, 1000.0
    r = analyze(beam(L, [Support("fixed", L)], [PointLoad(0.0, P)]))
    assert r.textbook is not None and r.textbook.case == "cantilever-end-load"
    assert max(r.textbook.rel_error.values()) < 1e-6
    assert r.v[0] == pytest.approx(-P * L**3 / (3 * EI), rel=1e-9)
    # load not at the free end -> no textbook formula
    assert textbook_check(beam(L, [Support("fixed", 0.0)], [PointLoad(1.0, P)])) is None


def test_sample_points_duplicates_and_reactions_guard():
    m = overhang_model()
    x = sample_points(m, 61)
    assert np.all(np.diff(x) >= 0)
    for k in (1.0, 4.0, 5.0, 6.0):  # concentrated actions: repeated
        assert np.sum(x == k) == 2
    assert np.sum(x == 3.0) == 1  # UDL end: key point only
    with pytest.raises(ValueError):
        reactions(beam(2.0, [Support("fixed", 0.0), Support("roller", 2.0)], [PointLoad(1.0, 1.0)]))
    with pytest.raises(ValueError):
        reactions(beam(2.0, [Support("fixed", 0.0)], [PointLoad(2.0, 1.0, -45.0)]))
    with pytest.raises(ValueError):
        slope_deflection(beam(2.0, [Support("fixed", 0.0), Support("roller", 2.0)], []), x, np.zeros_like(x))


def test_routing_errors():
    with pytest.raises(ModelError):
        analyze(beam(-1.0, [Support("fixed", 0.0)], []))
    with pytest.raises(UnstableStructureError):
        analyze(beam(2.0, [Support("roller", 0.0)], [PointLoad(1.0, 1.0)]))
