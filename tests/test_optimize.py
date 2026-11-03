import math

import numpy as np
import pytest

from smartbeam.materials import get_material
from smartbeam.model import BeamModel, PointLoad, Section, Support
from smartbeam.optimize import OptimizationProblem, optimize, pareto_front


def test_rect_optimum_matches_closed_form(ss_central_load):
    prob = OptimizationProblem(delta_allow=0.004, fos_min=2.0)
    r = optimize(ss_central_load, prob)
    assert r.status == "OPTIMIZED" and r.feasible
    assert all(c.satisfied for c in r.checks)
    # deflection governs; volume ~ b h with I ~ b h^3 -> b at its lower bound (0.2 x 30 mm)
    b = r.optimal_dims["b"]
    assert b == pytest.approx(0.006, rel=1e-6)
    I_req = 1000 * 2.0**3 / (48 * 200e9 * 0.004)
    assert r.optimal_dims["h"] == pytest.approx((12 * I_req / b) ** (1 / 3), rel=1e-4)
    assert r.reduction_pct > 0


def test_hollow_respects_wall_and_buckling_constraint():
    st = get_material("structural_steel")
    m = BeamModel(2.0, Section.hollow_circ(0.06, 0.05), st, [Support("fixed", 0.0)], [PointLoad(2.0, 1000.0)])
    r = optimize(m, OptimizationProblem(delta_allow=0.004, fos_min=2.0, t_min=0.002))
    assert r.status == "OPTIMIZED"
    D, d = r.optimal_dims["D"], r.optimal_dims["d"]
    assert (D - d) / 2 >= 0.002 - 1e-12
    # axial compression -> Euler constraint active, d = (FOS P 64 (KL)^2 / (pi^3 E))^(1/4)
    m2 = BeamModel(2.0, Section.solid_circ(0.04), st, [Support("fixed", 0.0)], [PointLoad(2.0, 20000.0, 180.0)])
    r2 = optimize(m2, OptimizationProblem(delta_allow=0.01, fos_min=2.0))
    assert r2.status == "OPTIMIZED"
    assert r2.optimal_dims["d"] == pytest.approx((2 * 20000 * 16 * 64 / (math.pi**3 * 200e9)) ** 0.25, rel=1e-4)
    assert any(c.name.startswith("Buckling") and c.satisfied for c in r2.checks)


def test_infeasible_is_reported(cantilever_end_load):
    m = cantilever_end_load.copy()
    m.loads[0].magnitude = 20000.0
    r = optimize(m, OptimizationProblem(delta_allow=0.002, bounds={"b": (0.01, 0.04), "h": (0.02, 0.08)}))
    assert r.status == "INFEASIBLE" and not r.feasible and r.optimized_model is None
    assert any("required I" in s for s in r.infeasibility_reasons)


def test_bad_problem_raises(cantilever_end_load):
    with pytest.raises(ValueError):
        optimize(cantilever_end_load, OptimizationProblem(fos_min=0.5))
    with pytest.raises(ValueError):
        optimize(cantilever_end_load, OptimizationProblem(bounds={"b": (0.05, 0.01)}))


def test_pareto_front_is_non_dominated(cantilever_end_load):
    df = pareto_front(cantilever_end_load, OptimizationProblem(objective="mass", delta_allow=0.01), n_points=10)
    assert len(df) == 50 and df.pareto.any()
    front = df[df.pareto][["mass", "stiffness", "fos"]].to_numpy()
    allp = df[["mass", "stiffness", "fos"]].to_numpy()
    for p in front:
        dom = (allp[:, 0] <= p[0]) & (allp[:, 1] >= p[1]) & (allp[:, 2] >= p[2]) & np.any(
            (allp[:, 0] < p[0], allp[:, 1] > p[1], allp[:, 2] > p[2]), axis=0)
        assert not dom.any()
