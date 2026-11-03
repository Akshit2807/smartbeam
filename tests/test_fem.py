"""FEM engine checks against closed-form solutions (fem.py does not use analytical.py)."""
import math

import numpy as np
import pytest

from smartbeam import fem, sections
from smartbeam.materials import get_material
from smartbeam.model import (BeamModel, DistributedLoad, ModelError, MomentLoad, PointLoad, Section, Support,
                             UnstableStructureError)

STEEL = get_material("structural_steel")
SEC = Section.rect(0.030, 0.060)
E, I, A = STEEL.E, sections.second_moment(SEC), sections.area(SEC)
EI = E * I
L, P, W = 2.0, 1000.0, 500.0


def model(supports, loads, L=L, section=SEC):
    return BeamModel(L, section, STEEL, supports, loads)


def rel(a, b):
    return abs(a - b) / abs(b)


# --------------------------------------------------------------------------- element matrices
def test_eb_element_symmetric_rank2():
    k = fem.eb_element_stiffness(E, I, 0.5)
    assert np.allclose(k, k.T)
    assert np.linalg.matrix_rank(k) == 2  # two rigid-body modes: translation, rotation
    assert np.allclose(k @ [1, 0, 1, 0], 0)  # rigid translation
    assert np.allclose(k @ [0, 1, 0.5, 1], 0, atol=1e-6 * k.max())  # rigid rotation v = x
    assert k[0, 0] == pytest.approx(12 * EI / 0.5**3)


def test_timoshenko_element_reduces_to_eb_and_is_singular():
    kt = fem.timoshenko_element_stiffness(E, I, 0.5, STEEL.G, A, 5 / 6)
    assert np.allclose(kt, kt.T) and np.linalg.matrix_rank(kt) == 2
    assert not np.allclose(kt, fem.eb_element_stiffness(E, I, 0.5))
    k_inf_shear = fem.timoshenko_element_stiffness(E, I, 0.5, STEEL.G * 1e9, A, 5 / 6)
    assert np.allclose(k_inf_shear, fem.eb_element_stiffness(E, I, 0.5), rtol=1e-6)


def test_frame_element():
    k = fem.frame_element_stiffness(E, A, I, 0.5)
    assert np.allclose(k, k.T) and np.linalg.matrix_rank(k) == 3
    assert k[0, 0] == pytest.approx(E * A / 0.5) and k[0, 1] == 0
    assert np.allclose(k[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])], fem.eb_element_stiffness(E, I, 0.5))


def test_consistent_load_formula():
    q1, q2, Le = -300.0, -800.0, 0.4
    expected = Le * np.array([(7 * q1 + 3 * q2) / 20, Le * (3 * q1 + 2 * q2) / 60,
                              (3 * q1 + 7 * q2) / 20, -Le * (2 * q1 + 3 * q2) / 60])
    assert np.allclose(fem.consistent_load(q1, q2, Le), expected)
    f = fem.consistent_load(q1, q2, Le, phi=0.7)
    assert f[0] + f[2] == pytest.approx(0.5 * (q1 + q2) * Le)  # resultant preserved for any phi
    assert np.allclose(fem.consistent_load(-1, -1, Le, 0.7), fem.consistent_load(-1, -1, Le))  # UDL: phi-free


def test_mesh_has_nodes_at_key_points():
    m = model([Support("pin", 0.0), Support("roller", 1.7)],
              [PointLoad(0.33, P), DistributedLoad(0.5, 1.25, W, 0), MomentLoad(1.9, 10.0)])
    mesh = fem.build_mesh(m, 8)
    assert mesh.n_elements >= 8 and np.all(np.diff(mesh.nodes) > 0)
    for x in (0.0, 0.33, 0.5, 1.25, 1.7, 1.9, 2.0):
        assert np.min(np.abs(mesh.nodes - x)) < 1e-12


def test_assembly_and_boundary_conditions():
    m = model([Support("fixed", 0.0), Support("roller", L)], [DistributedLoad(0, L, W, W)])
    mesh = fem.build_mesh(m, 4)
    K, F = fem.assemble(m, mesh)
    assert K.shape == (10, 10) and np.allclose(K, K.T)
    assert np.linalg.matrix_rank(K) == 10 - 2  # free-free beam: 2 rigid modes
    assert F[0::2].sum() == pytest.approx(-W * L)
    assert list(fem.restrained_dofs(m, mesh)) == [0, 1, 8]
    assert list(fem.restrained_dofs(m, mesh, frame=True)) == [0, 1, 2, 13]


# --------------------------------------------------------------------------- Euler-Bernoulli benchmarks
def test_cantilever_end_load_exact_with_one_element():
    r = fem.solve(model([Support("fixed", 0.0)], [PointLoad(L, P)]), n_elements=1)
    assert r.v_nodes[-1] == pytest.approx(-P * L**3 / (3 * EI), rel=1e-12)
    assert r.theta_nodes[-1] == pytest.approx(-P * L**2 / (2 * EI), rel=1e-12)
    R = r.reactions[0]
    assert R.Fy == pytest.approx(P) and R.M == pytest.approx(P * L)  # CCW reaction moment
    assert r.M[0] == pytest.approx(-P * L)  # hogging at the wall
    assert r.sigma_top[0] > 0  # hogging: top in tension
    assert abs(r.sigma_max.value) == pytest.approx(P * L * (SEC.h / 2) / I)


def test_ss_udl_nodal_values_exact():
    m = model([Support("pin", 0.0), Support("roller", L)], [DistributedLoad(0, L, W, W)])
    for n in (2, 4):
        r = fem.solve(m, n_elements=n)
        x = r.mesh.nodes
        exact = -W * x * (L**3 - 2 * L * x**2 + x**3) / (24 * EI)
        assert np.allclose(r.v_nodes, exact, rtol=1e-10, atol=1e-15)
        assert r.M_max.value == pytest.approx(W * L**2 / 8)
        assert [x.Fy for x in r.reactions] == pytest.approx([W * L / 2] * 2)


def test_ss_central_load():
    r = fem.solve(model([Support("pin", 0.0), Support("roller", L)], [PointLoad(L / 2, P)]), n_elements=2)
    assert [x.Fy for x in r.reactions] == pytest.approx([P / 2, P / 2])
    assert r.M_max.value == pytest.approx(P * L / 4) and r.M_max.x == pytest.approx(L / 2)
    assert r.v_max.value == pytest.approx(-P * L**3 / (48 * EI))
    assert r.sigma_bottom.max() > 0 and r.sigma_max.where == "top"  # |top| == |bottom|; top found first
    i = np.where(np.isclose(r.x, L / 2))[0]
    assert r.V[i[0]] == pytest.approx(P / 2) and r.V[i[-1]] == pytest.approx(-P / 2)  # jump drawn at x = L/2


def test_propped_cantilever_udl():
    r = fem.solve(model([Support("fixed", 0.0), Support("roller", L)], [DistributedLoad(0, L, W, W)]), 8)
    fixed, roller = r.reactions
    assert roller.Fy == pytest.approx(3 * W * L / 8)
    assert fixed.Fy == pytest.approx(5 * W * L / 8)
    assert fixed.M == pytest.approx(W * L**2 / 8)
    assert r.M[0] == pytest.approx(-W * L**2 / 8)
    assert r.M.max() == pytest.approx(9 * W * L**2 / 128, rel=1e-4)  # at x = 5L/8 (sampled)


def test_fixed_fixed_central_load():
    r = fem.solve(model([Support("fixed", 0.0), Support("fixed", L)], [PointLoad(L / 2, P)]), n_elements=2)
    assert r.M[0] == pytest.approx(-P * L / 8) and r.M[-1] == pytest.approx(-P * L / 8)
    assert r.M.max() == pytest.approx(P * L / 8)
    assert r.v_nodes[1] == pytest.approx(-P * L**3 / (192 * EI))


def test_continuous_two_span_udl():
    r = fem.solve(model([Support("pin", 0.0), Support("roller", L / 2), Support("roller", L)],
                        [DistributedLoad(0, L, W, W)]), 10)
    l = L / 2
    assert [x.Fy for x in r.reactions] == pytest.approx([3 * W * l / 8, 10 * W * l / 8, 3 * W * l / 8])
    assert r.M_max.value == pytest.approx(-W * l**2 / 8)  # hogging over the middle support


def test_triangular_load_and_moment_recovery():
    # SS beam, load rising linearly 0 -> W: R_A = WL/6, R_B = WL/3, M_max = WL^2/(9 sqrt 3) at x = L/sqrt 3
    r = fem.solve(model([Support("pin", 0.0), Support("roller", L)], [DistributedLoad(0, L, 0.0, W)]), 3,
                  n_per_element=200)
    assert [x.Fy for x in r.reactions] == pytest.approx([W * L / 6, W * L / 3])
    assert r.M_max.value == pytest.approx(W * L**2 / (9 * math.sqrt(3)), rel=1e-5)  # sampled peak
    assert r.M_max.x == pytest.approx(L / math.sqrt(3), abs=2e-3)
    assert np.allclose(r.M[[0, -1]], 0, atol=1e-9)


def test_applied_moment_jump():
    C = 300.0
    r = fem.solve(model([Support("pin", 0.0), Support("roller", L)], [MomentLoad.ccw(L / 2, C)]), 4)
    assert [x.Fy for x in r.reactions] == pytest.approx([C / L, -C / L])
    i = np.where(np.isclose(r.x, L / 2))[0]
    assert r.M[i[0]] - r.M[i[-1]] == pytest.approx(C)  # CCW moment: M jumps down by C


def test_partial_lvl_equilibrium():
    loads = [PointLoad(0.4, 700), DistributedLoad(0.6, 1.5, 200, 900), MomentLoad.cw(1.8, 150)]
    r = fem.solve(model([Support("fixed", 0.0), Support("roller", 1.2), Support("pin", L)], loads), 12)
    fy = sum(x.Fy for x in r.reactions) + sum(ld.fy for ld in loads[:1]) + loads[1].resultant
    m0 = sum(x.Fy * x.x + x.M for x in r.reactions) + loads[0].fy * 0.4 \
        + loads[1].resultant * loads[1].centroid + loads[2].M
    assert abs(fy) < 1e-8 and abs(m0) < 1e-8


# --------------------------------------------------------------------------- Timoshenko
def test_timoshenko_deep_cantilever_formula():
    deep = Section.rect(0.05, 0.5)
    Id, Ad, kappa = sections.second_moment(deep), sections.area(deep), sections.shear_correction_factor(deep)
    r = fem.solve(model([Support("fixed", 0.0)], [PointLoad(1.0, P)], L=1.0, section=deep), 1, theory="timoshenko")
    assert r.v_nodes[-1] == pytest.approx(-(P / (3 * E * Id) + P / (kappa * STEEL.G * Ad)), rel=1e-12)
    assert r.theta_nodes[-1] == pytest.approx(-P / (2 * E * Id), rel=1e-12)  # section rotation unaffected


def test_timoshenko_ss_udl_coarse_mesh_exact():
    deep = Section.rect(0.05, 0.4)
    Id, Ad, kappa = sections.second_moment(deep), sections.area(deep), sections.shear_correction_factor(deep)
    r = fem.solve(model([Support("pin", 0.0), Support("roller", 1.0)], [DistributedLoad(0, 1.0, W, W)],
                        L=1.0, section=deep), 2, theory="timoshenko")
    expected = 5 * W / (384 * E * Id) + W / (8 * kappa * STEEL.G * Ad)
    assert r.v_max.value == pytest.approx(-expected, rel=1e-10)


def test_timoshenko_reconstruction_matches_nodes():
    deep = Section.rect(0.05, 0.4)
    r = fem.solve(model([Support("fixed", 0.0)], [DistributedLoad(0, 1.0, W, 0)], L=1.0, section=deep), 5,
                  theory="timoshenko")
    for xn, vn, tn in zip(r.mesh.nodes, r.v_nodes, r.theta_nodes):
        at = np.isclose(r.x, xn)
        assert np.allclose(r.v[at], vn, rtol=1e-12, atol=1e-18)
        assert np.allclose(r.theta[at], tn, rtol=1e-12, atol=1e-18)


def test_timoshenko_tends_to_eb_as_slenderness_grows():
    diffs = []
    for Lb in (0.3, 1.0, 3.0, 10.0):
        c = fem.compare_theories(model([Support("fixed", 0.0)], [PointLoad(Lb, P)], L=Lb), 10)
        assert c.v_timoshenko > c.v_eb
        diffs.append(c.diff_pct)
    assert all(a > b for a, b in zip(diffs, diffs[1:]))
    assert diffs[-1] < 0.05  # L/h = 167: shear deflection is negligible


# --------------------------------------------------------------------------- frame
def test_frame_axial_bar():
    r = fem.solve(model([Support("fixed", 0.0)], [PointLoad(L, P, 0.0)]))
    assert r.element == "frame"
    assert r.u_nodes[-1] == pytest.approx(P * L / (E * A))
    assert np.allclose(r.N, P) and r.reactions[0].Fx == pytest.approx(-P)
    assert np.allclose(r.v, 0) and np.allclose(r.sigma_top, P / A)


def test_frame_inclined_load_superposition():
    r = fem.solve(model([Support("fixed", 0.0)], [PointLoad(L, P, -135.0)]))  # down and to the left
    c = math.sqrt(0.5)
    assert r.u_nodes[-1] == pytest.approx(-P * c * L / (E * A))
    assert r.v_nodes[-1] == pytest.approx(-P * c * L**3 / (3 * EI))
    assert np.allclose(r.N, -P * c)  # compression
    assert r.sigma_bottom[0] == pytest.approx(-P * c / A - P * c * L * SEC.h / 2 / I)


def test_frame_on_transverse_model_without_axial_restraint():
    r = fem.solve(model([Support("roller", 0.0), Support("roller", L)], [PointLoad(1.0, P)]), 4, frame=True)
    assert r.warnings and abs(r.reactions[0].Fx) < 1e-9
    assert r.v_max.value == pytest.approx(-P * L**3 / (48 * EI))


# --------------------------------------------------------------------------- errors & studies
@pytest.mark.parametrize("supports, loads", [
    ([Support("roller", 0.0), Support("roller", L)], [PointLoad(1.0, P, -45.0)]),  # free axially
    ([Support("pin", 1.0)], [PointLoad(1.5, P)]),  # free to rotate
    ([Support("roller", 0.5)], [PointLoad(1.5, P)]),
])
def test_mechanism_detected(supports, loads):
    with pytest.raises(UnstableStructureError, match="insufficient supports to prevent rigid-body motion"):
        fem.solve(model(supports, loads))


def test_invalid_model():
    with pytest.raises(ModelError):
        fem.solve(model([Support("fixed", 0.0)], [PointLoad(3.0, P)]))
    with pytest.raises(ValueError):
        fem.solve(model([Support("fixed", 0.0)], [PointLoad(L, P)]), theory="reddy")


def test_close_load_snapped_keeps_reactions_accurate():
    r = fem.solve(model([Support("pin", 0.0), Support("roller", L)], [PointLoad(1e-8, P), PointLoad(1.0, 10)]), 40)
    assert [x.Fy for x in r.reactions] == pytest.approx([P + 5, 5], rel=1e-6)
    assert r.warnings


def test_convergence_study(cantilever_udl, ss_central_load):
    c = fem.convergence_study(cantilever_udl)
    assert c.n_elements == [2, 4, 8, 16, 32] and c.reference is not None
    for k in ("v_max", "M_max", "sigma_max"):
        assert c.error_pct[k].max() < 1e-6  # exact nodal values / equilibrium-recovered moments
    # cubic slope inside the last element overshoots the exact tip slope under UDL; converges ~h^3
    assert np.all(np.diff(c.error_pct["theta_max"]) < 0) and c.error_pct["theta_max"][-1] < 1e-3
    assert c.reference["v_max"] == pytest.approx(W * L**4 / (8 * EI))
    assert np.all(np.diff(c.error_pct["M_fe"]) < 0) and c.error_pct["M_fe"][0] > 1  # EI v'' converges
    c2 = fem.convergence_study(ss_central_load, theory="timoshenko")
    assert c2.error_pct["v_max"].max() < 1e-6
    general = model([Support("fixed", 0.0), Support("roller", L)], [PointLoad(0.7, P)])
    c3 = fem.convergence_study(general)
    assert c3.reference is None and c3.error_pct is None and len(c3.v_max) == 5 and c3.note
