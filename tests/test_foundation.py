"""Tests for the foundation modules: model, sections, materials, units, metrics.check."""
import math

import numpy as np
import pytest

from smartbeam import sections, units
from smartbeam.materials import MATERIALS, custom_material, get_material
from smartbeam.metrics import check
from smartbeam.model import (BeamModel, DistributedLoad, MomentLoad, PointLoad, Section, Support,
                             check_stability, classify, is_transverse_only, validate)


# ---------------------------------------------------------------- sections
def test_rect_properties():
    s = Section.rect(0.03, 0.06)
    p = sections.properties(s)
    assert p.A == pytest.approx(0.03 * 0.06)
    assert p.I == pytest.approx(0.03 * 0.06**3 / 12)
    assert p.Z == pytest.approx(0.03 * 0.06**2 / 6)
    assert p.r_g == pytest.approx(0.06 / math.sqrt(12))
    assert (p.y_top, p.y_bottom, p.centroid) == pytest.approx((0.03, -0.03, 0.03))
    assert p.kappa == pytest.approx(5 / 6)
    # tau_max = 3V/(2A) at the neutral axis, zero at the fibres
    assert sections.shear_stress(s, 1000.0, 0.0) == pytest.approx(1.5 * 1000 / p.A)
    assert sections.shear_stress(s, 1000.0, 0.03) == pytest.approx(0.0)


def test_circular_properties():
    d = 0.05
    s = Section.solid_circ(d)
    assert sections.area(s) == pytest.approx(math.pi * d**2 / 4)
    assert sections.second_moment(s) == pytest.approx(math.pi * d**4 / 64)
    assert sections.section_modulus(s) == pytest.approx(math.pi * d**3 / 32)
    assert sections.width_at(s, 0.0) == pytest.approx(d)
    assert sections.shear_stress(s, 1000.0, 0.0) == pytest.approx(4 / 3 * 1000 / sections.area(s))

    D, di = 0.06, 0.05
    h = Section.hollow_circ(D, di)
    assert sections.area(h) == pytest.approx(math.pi * (D**2 - di**2) / 4)
    assert sections.second_moment(h) == pytest.approx(math.pi * (D**4 - di**4) / 64)
    assert sections.width_at(h, 0.0) == pytest.approx(D - di)
    # Q at the NA equals (D^3 - d^3)/12 for a tube
    assert sections.first_moment(h, 0.0) == pytest.approx((D**3 - di**3) / 12)
    assert 0.5 < sections.shear_correction_factor(Section.hollow_circ(0.1, 0.098)) < 0.56


def test_first_moment_matches_numerical_integral():
    for s in (Section.rect(0.02, 0.05), Section.solid_circ(0.04), Section.hollow_circ(0.05, 0.03)):
        y0 = 0.3 * s.depth / 2
        ys = np.linspace(y0, s.depth / 2, 20001)
        q_num = np.trapezoid(sections.width_at(s, ys) * ys, ys)
        assert sections.first_moment(s, y0) == pytest.approx(q_num, rel=1e-4)


# ---------------------------------------------------------------- model
def _full_model():
    return BeamModel(
        L=3.0, section=Section.hollow_circ(0.08, 0.06), material=get_material("aluminium_6061_t6"),
        supports=[Support("pin", 0.0), Support("roller", 2.5)],
        loads=[PointLoad(1.0, 2000.0, -60.0), DistributedLoad(0.5, 2.0, 0.0, 800.0, "down"),
               MomentLoad.cw(2.8, 150.0)],
        name="round-trip",
    )


def test_json_round_trip():
    m = _full_model()
    m2 = BeamModel.from_json(m.to_json())
    assert m2 == m
    assert m2.to_dict() == m.to_dict()


def test_load_components_and_conventions():
    down = PointLoad(1.0, 1000.0)
    assert (down.fx, down.fy) == pytest.approx((0.0, -1000.0))
    assert down.is_transverse
    inc = PointLoad(1.0, 1000.0, -30.0)
    assert inc.fx == pytest.approx(1000 * math.cos(math.radians(30)))
    assert not inc.is_transverse
    tri = DistributedLoad(0.0, 3.0, 0.0, 600.0)  # triangular, downward
    assert tri.resultant == pytest.approx(-900.0)
    assert tri.centroid == pytest.approx(2.0)
    assert tri.q(1.5) == pytest.approx(-300.0) and tri.q(3.5) == 0.0
    assert MomentLoad.cw(1, 5).M == -5 and MomentLoad.ccw(1, -5).M == 5


def test_validation():
    assert validate(_full_model()) == []
    bad = _full_model()
    bad.L = -1
    assert validate(bad)
    bad = _full_model()
    bad.section = Section.hollow_circ(0.05, 0.06)
    bad.loads.append(DistributedLoad(2.0, 1.0, 10, 10))
    bad.supports.append(Support("hinge", 1.0))
    errs = validate(bad)
    assert len(errs) == 3, errs


def test_classify_and_transverse(cantilever_end_load, ss_udl):
    assert classify(cantilever_end_load) == "cantilever"
    assert classify(ss_udl) == "simply-supported"
    assert is_transverse_only(ss_udl)
    assert not is_transverse_only(_full_model())
    m = _full_model()
    assert classify(m) == "general"  # overhang


@pytest.mark.parametrize("supports, transverse, stable, degree", [
    ([("fixed", 0)], True, True, 0),
    ([("pin", 0), ("roller", 2)], True, True, 0),
    ([("pin", 0), ("pin", 2)], True, True, 0),  # axial redundancy irrelevant in transverse mode
    ([("fixed", 0), ("roller", 2)], True, True, 1),
    ([("fixed", 0), ("fixed", 2)], True, True, 2),
    ([("pin", 0)], True, False, 0),
    ([("roller", 0), ("roller", 2)], False, False, 0),  # cannot resist axial load
    ([("pin", 0), ("roller", 2)], False, True, 0),
    ([("fixed", 0), ("fixed", 2)], False, True, 3),
])
def test_stability(supports, transverse, stable, degree):
    loads = [PointLoad(1.0, 100.0, -90.0 if transverse else -45.0)]
    m = BeamModel(2.0, Section.rect(0.02, 0.04), get_material("mild_steel"),
                  [Support(t, x) for t, x in supports], loads)
    st = check_stability(m)
    assert st.mode == ("transverse" if transverse else "planar")
    assert st.stable is stable
    if stable:
        assert st.degree == degree and st.determinate is (degree == 0)


# ---------------------------------------------------------------- materials, units, metrics
def test_materials():
    assert len(MATERIALS) == 5
    for m in MATERIALS.values():
        assert m.approximate and m.sigma_f_prime is None and m.endurance_limit is None
        assert m.uts >= m.yield_strength
    s = get_material("structural_steel")
    s.E = 1
    assert MATERIALS["structural_steel"].E == 200e9  # library not mutated
    assert get_material("mild_steel").G == pytest.approx(200e9 / 2.58)
    with pytest.raises(ValueError):
        custom_material("x", E=-1, yield_strength=1, nu=0.3, density=1)


def test_units():
    assert units.to_si(5, "kN") == 5000
    assert units.from_si(0.002, "mm") == pytest.approx(2)
    assert units.convert(1, "kN*m", "N*mm") == pytest.approx(1e6)
    assert units.fmt(2.5e8, "MPa") == "250 MPa"


def test_constraint_check():
    assert check("d", 4.0, 5.0, "<=").satisfied
    assert check("fos", 1.2, 1.5, ">=").margin_pct == pytest.approx(-20)
