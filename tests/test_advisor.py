import math

import pytest

from smartbeam.advisor import ROUND, advise
from smartbeam.materials import get_material
from smartbeam.model import BeamModel, PointLoad, Section, Support


def test_deflection_fix_uses_cube_root_and_passes(cantilever_end_load):
    rep = advise(cantilever_end_load, delta_allow=0.008)
    assert rep.limiting == "deflection" and rep.utilisation["deflection"] > 1
    rec = rep.recommendations[0]
    ratio = rec.current / rec.limit
    h_expected = math.ceil(0.060 * ratio ** (1 / 3) / ROUND - 1e-9) * ROUND
    assert rec.modified_model.section.h == pytest.approx(h_expected)
    assert rec.after.delta_max <= 0.008 and rec.satisfies_all
    assert all(r.satisfies_all for r in rep.recommendations)


def test_material_upgrade_for_stiffness():
    m = BeamModel(2.0, Section.rect(0.05, 0.08), get_material("aluminium_6061_t6"), [Support("fixed", 0.0)],
                  [PointLoad(2.0, 1000.0)])
    rep = advise(m)  # delta_allow = L/250 = 8 mm
    mats = [r for r in rep.recommendations if r.action.startswith("Change material")]
    assert mats and mats[0].modified_model.material.E > m.material.E and mats[0].satisfies_all


def test_buckling_limited():
    m = BeamModel(2.0, Section.rect(0.03, 0.06), get_material("structural_steel"), [Support("fixed", 0.0)],
                  [PointLoad(2.0, 20000.0, 180.0)])
    rep = advise(m)
    assert rep.limiting == "buckling"
    assert rep.recommendations[0].modified_model.section.b > 0.03  # weak axis is b
    assert all(r.satisfies_all for r in rep.recommendations)


def test_material_excess_and_none(cantilever_end_load):
    big = cantilever_end_load.copy()
    big.section = Section.rect(0.1, 0.2)
    rep = advise(big)
    assert rep.limiting == "material excess"
    rec = rep.recommendations[0]
    assert rec.after.mass < rec.before.mass and rec.satisfies_all
    ok = cantilever_end_load.copy()
    ok.section = Section.rect(0.03, 0.09)  # delta 7.3 mm < 8 mm, FOS 5.1
    assert advise(ok).limiting == "none"
