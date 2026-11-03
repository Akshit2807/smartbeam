import math

import numpy as np
import pytest

from smartbeam import safety
from smartbeam.analytical import analyze
from smartbeam.materials import get_material
from smartbeam.model import BeamModel, PointLoad, Section, Support

E, SY = 200e9, 250e6
I_MIN = 0.060 * 0.030**3 / 12  # rect 30 x 60 mm, weak axis


def test_yield_fos_and_status(cantilever_end_load):
    r = analyze(cantilever_end_load)
    y = safety.yield_check(r, 1.5)
    sigma = 1000 * 2.0 / (0.030 * 0.060**2 / 6)  # M / Z = 111.1 MPa
    assert y.fos == pytest.approx(SY / sigma, rel=1e-6)
    assert y.first_yield_load_factor == pytest.approx(y.fos)
    assert y.status == safety.STATUS_SAFE
    assert safety.yield_check(r, 3.0).status == safety.STATUS_WARNING
    m = cantilever_end_load.copy()
    m.loads[0].magnitude *= 3
    assert safety.yield_check(analyze(m)).status == safety.STATUS_YIELD


def test_von_mises_slender_beam_equals_fibre_stress(ss_central_load):
    r = analyze(ss_central_load)
    vm = safety.von_mises_check(r)
    assert vm.von_mises == pytest.approx(abs(r.sigma_max.value), rel=1e-9)
    assert vm.tau == pytest.approx(0.0, abs=1e-6)  # extreme fibre: Q = 0
    # critical section is mid-span (|V| = 500 N): at the neutral axis sigma = 0, tau = 3V/2A
    assert vm.x == pytest.approx(1.0)
    j0 = len(vm.y_profile) // 2
    assert vm.y_profile[j0] == pytest.approx(0.0, abs=1e-12)
    assert vm.vm_profile[j0] == pytest.approx(math.sqrt(3) * 1.5 * 500 / (0.03 * 0.06), rel=1e-9)


def test_infer_end_condition(cantilever_end_load, ss_udl):
    assert safety.infer_end_condition(cantilever_end_load) == "fixed-free"
    assert safety.infer_end_condition(ss_udl) == "pinned-pinned"
    m = ss_udl.copy()
    m.supports = [Support("fixed", 0.0), Support("fixed", 2.0)]
    assert safety.infer_end_condition(m) == "fixed-fixed"
    m.supports = [Support("fixed", 0.0), Support("roller", 2.0)]
    assert safety.infer_end_condition(m) == "fixed-pinned"
    m.supports = [Support("pin", 0.0), Support("roller", 1.5)]
    assert safety.infer_end_condition(m) is None


@pytest.mark.parametrize("ec,K", list(safety.BUCKLING_K.items()))
def test_euler_buckling_explicit_load(cantilever_end_load, ec, K):
    b = safety.buckling(cantilever_end_load, end_condition=ec, P_applied=1000.0)
    Pcr = math.pi**2 * E * I_MIN / (K * 2.0) ** 2
    assert b.applicable and b.Pcr == pytest.approx(Pcr, rel=1e-12)
    assert b.fos == pytest.approx(Pcr / 1000.0)
    assert b.slenderness == pytest.approx(K * 2.0 / math.sqrt(I_MIN / (0.03 * 0.06)))
    assert b.slenderness_limit == pytest.approx(math.sqrt(2 * math.pi**2 * E / SY))
    assert np.max(np.abs(b.mode_shape)) == pytest.approx(1.0)
    assert b.mode_shape[0] == pytest.approx(0.0, abs=1e-12)  # every case is restrained at x = 0


def test_fixed_pinned_mode_satisfies_bcs(ss_udl):
    b = safety.buckling(ss_udl, end_condition="fixed-pinned", P_applied=1.0, n_points=2001)
    w, x = b.mode_shape, b.mode_x
    assert w[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.gradient(w, x)[0] == pytest.approx(0.0, abs=1e-2)
    assert safety.BUCKLING_K["fixed-pinned"] == pytest.approx(math.pi / safety.FIXED_PINNED_KL, abs=1e-3)


def test_buckling_from_model_axial_load():
    st = get_material("structural_steel")
    m = BeamModel(2.0, Section.rect(0.03, 0.06), st, [Support("fixed", 0.0)], [PointLoad(2.0, 10000.0, 180.0)])
    b = safety.buckling(m)
    assert b.P_applied == pytest.approx(10000.0, rel=1e-9)
    assert b.Pcr == pytest.approx(math.pi**2 * E * I_MIN / 16.0)
    m.loads = [PointLoad(2.0, 10000.0, 0.0)]  # tension
    assert not safety.buckling(m).applicable
    rep = safety.assess(m, analyze(m))
    assert rep.status == safety.STATUS_SAFE and any("not fracture" in s for s in rep.messages)


def test_stocky_member_warning(cantilever_end_load):
    m = cantilever_end_load.copy()
    m.L = 0.3
    m.loads = [PointLoad(0.3, 1000.0, 180.0)]
    b = safety.buckling(m)
    assert b.slenderness < b.slenderness_limit
    assert any("UNCONSERVATIVE" in w for w in b.warnings)


def test_fatigue_hand_calc():
    f = safety.fatigue(200e6, -50e6, 400e6, sigma_f_prime=900e6, b=-0.1, endurance_limit=100e6)
    assert (f.sigma_a, f.sigma_m) == (125e6, 75e6)
    assert f.sigma_ar == pytest.approx(125e6 / (1 - 75 / 400))
    assert f.sigma_ar_gerber == pytest.approx(125e6 / (1 - (75 / 400) ** 2))
    assert f.cycles_to_failure == pytest.approx(0.5 * (f.sigma_ar / 900e6) ** (1 / -0.1))
    assert f.goodman_fos == pytest.approx(1 / (125 / 100 + 75 / 400))
    assert f.infinite_life is False and len(f.sn_N) == len(f.sn_S) > 0


def test_fatigue_without_data_is_explicit():
    f = safety.fatigue(100e6, 0.0, 400e6)
    assert f.cycles_to_failure is None and f.goodman_fos is None and f.infinite_life is None
    assert f.sn_N.size == 0
    assert any("Insufficient data" in w for w in f.warnings)
    with pytest.raises(ValueError):
        safety.fatigue(100e6, 0.0, 400e6, sigma_f_prime=900e6, b=0.1)
