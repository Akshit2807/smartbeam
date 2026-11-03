"""Independent validation suite (Test & Validation Engineer).

Every expected value below is derived here from first principles (closed-form beam theory, hand-assembled
matrices, SymPy integration), NOT read back from the engine's own textbook table. Tolerance: rel 1e-3 unless
stated (the analytical route samples ~500 points; the FEM nodal values of EB elements are exact).
"""
import math

import numpy as np
import pytest
import sympy as sp

from smartbeam import advisor, fem, lab, safety, sections
from smartbeam.analytical import analyze
from smartbeam.materials import get_material
from smartbeam.model import (BeamModel, DistributedLoad, ModelError, MomentLoad, PointLoad, Section, Support,
                             UnstableStructureError)
from smartbeam.optimize import OptimizationProblem, optimize
from smartbeam.playground import convert
from smartbeam.report import build_pdf, lab_report_pdf

L, P, W = 2.0, 1000.0, 500.0  # same as conftest fixtures
B, H = 0.030, 0.060
E = 200e9
I_RECT = B * H**3 / 12  # 5.4e-7 m^4
Z_RECT = B * H**2 / 6  # 1.8e-5 m^3
EI = E * I_RECT
RTOL = 1e-3


def close(a, b, rtol=RTOL):
    return math.isclose(abs(a), abs(b), rel_tol=rtol, abs_tol=1e-12)


def beam(supports, loads, L=L, section=None, material="structural_steel"):
    return BeamModel(L=L, section=section or Section.rect(B, H), material=get_material(material),
                     supports=supports, loads=loads)


def ss():
    return [Support("pin", 0.0), Support("roller", L)]


def fy_at(reacts, x):
    return sum(r.Fy for r in reacts if abs(r.x - x) < 1e-9)


# =========================================================================== 1. four textbook benchmarks
# expected: R (sum of support Fy), M_reaction (cantilever wall), V_max, M_max, sigma_max, theta_max, delta_max
BENCH = {
    "cantilever_end_load": dict(R=P, Mr=P * L, V=P, M=P * L, sigma=P * L / Z_RECT, theta=P * L**2 / (2 * EI),
                                delta=P * L**3 / (3 * EI)),
    "cantilever_udl": dict(R=W * L, Mr=W * L**2 / 2, V=W * L, M=W * L**2 / 2, sigma=W * L**2 / 2 / Z_RECT,
                           theta=W * L**3 / (6 * EI), delta=W * L**4 / (8 * EI)),
    "ss_central_load": dict(R=P / 2, V=P / 2, M=P * L / 4, sigma=P * L / 4 / Z_RECT, theta=P * L**2 / (16 * EI),
                            delta=P * L**3 / (48 * EI)),
    "ss_udl": dict(R=W * L / 2, V=W * L / 2, M=W * L**2 / 8, sigma=W * L**2 / 8 / Z_RECT,
                   theta=W * L**3 / (24 * EI), delta=5 * W * L**4 / (384 * EI)),
}


def _check_benchmark(name, res, request):
    exp = BENCH[name]
    model = request.getfixturevalue(name)
    if name.startswith("cantilever"):
        (r,) = res.reactions
        assert close(r.Fy, exp["R"]) and r.Fy > 0
        assert close(r.M, exp["Mr"]) and r.M > 0  # CCW wall moment
    else:
        assert close(fy_at(res.reactions, 0.0), exp["R"]) and close(fy_at(res.reactions, L), exp["R"])
    assert close(res.V_max.value, exp["V"])
    assert close(res.M_max.value, exp["M"])
    assert close(res.sigma_max.value, exp["sigma"])
    assert close(res.theta_max.value, exp["theta"])
    assert close(res.v_max.value, exp["delta"])
    assert res.v_max.value < 0  # downward
    # sign conventions: cantilever hogging, SS sagging
    assert (res.M_max.value < 0) == name.startswith("cantilever")
    # total vertical equilibrium
    total = P if "load" in name else W * L
    assert math.isclose(sum(r.Fy for r in res.reactions), total, rel_tol=1e-9)
    assert model.L == L


@pytest.mark.parametrize("name", list(BENCH))
def test_benchmark_analytical(name, request):
    res = analyze(request.getfixturevalue(name))
    assert res.method == "analytical"
    _check_benchmark(name, res, request)


@pytest.mark.parametrize("name", list(BENCH))
@pytest.mark.parametrize("theory", ["euler"])
def test_benchmark_fem(name, theory, request):
    res = fem.solve(request.getfixturevalue(name), n_elements=20, theory=theory)
    _check_benchmark(name, res, request)


@pytest.mark.parametrize("name", list(BENCH))
def test_benchmark_fem_matches_analytical_everywhere(name, request):
    """Not only the maxima: the whole deflection curve agrees (FEM sampled at the analytical x)."""
    model = request.getfixturevalue(name)
    a, f = analyze(model), fem.solve(model, n_elements=16)
    v_f = np.interp(a.x, f.x, f.v)
    assert np.max(np.abs(v_f - a.v)) <= 1e-3 * abs(a.v_max.value)
    M_f = np.interp(f.mesh.nodes, a.x, a.M)  # nodal moments
    M_fem_nodes = np.interp(f.mesh.nodes, f.x, f.M)
    assert np.max(np.abs(M_f - M_fem_nodes)) <= 1e-3 * abs(a.M_max.value) + 1e-6


def test_benchmark_sigma_bottom_tension_for_sagging(ss_central_load):
    r = analyze(ss_central_load)
    i = np.argmin(np.abs(r.x - L / 2))
    assert r.sigma_bottom[i] > 0 > r.sigma_top[i]
    assert close(r.sigma_bottom[i], P * L / 4 / Z_RECT)


# =========================================================================== 2. SymPy symbolic verification
def _sympy_deflection(M_expr, x, bcs):
    """Solve EI v'' = M(x) symbolically with the given (point, derivative order, value) BCs."""
    EIs = sp.Symbol("EI", positive=True)
    C1, C2 = sp.symbols("C1 C2")
    v = sp.integrate(sp.integrate(M_expr / EIs, x), x) + C1 * x + C2
    eqs = [sp.diff(v, x, k).subs(x, xp) - val for xp, k, val in bcs]
    sol = sp.solve(eqs, [C1, C2], dict=True)[0]
    return sp.simplify(v.subs(sol)), EIs


def test_sympy_cantilever_end_load_deflection(cantilever_end_load):
    x, Ls, Ps = sp.symbols("x L P", positive=True)
    v, EIs = _sympy_deflection(-Ps * (Ls - x), x, [(0, 0, 0), (0, 1, 0)])  # hogging M, v(0)=v'(0)=0
    assert sp.simplify(v.subs(x, Ls) + Ps * Ls**3 / (3 * EIs)) == 0  # v(L) = -PL^3/3EI
    assert sp.simplify(sp.diff(v, x).subs(x, Ls) + Ps * Ls**2 / (2 * EIs)) == 0  # theta(L) = -PL^2/2EI
    # the engine's whole curve matches the symbolic one
    f = sp.lambdify(x, v.subs({Ls: L, Ps: P, EIs: EI}), "numpy")
    r = analyze(cantilever_end_load)
    assert np.allclose(r.v, f(r.x), rtol=0, atol=1e-6 * abs(r.v_max.value))


def test_sympy_ss_udl_deflection(ss_udl):
    x, Ls, ws = sp.symbols("x L w", positive=True)
    v, EIs = _sympy_deflection(ws * x * (Ls - x) / 2, x, [(0, 0, 0), (Ls, 0, 0)])
    assert sp.simplify(v.subs(x, Ls / 2) + 5 * ws * Ls**4 / (384 * EIs)) == 0
    f = sp.lambdify(x, v.subs({Ls: L, ws: W, EIs: EI}), "numpy")
    a = analyze(ss_udl)
    assert np.allclose(a.v, f(a.x), rtol=0, atol=1e-6 * abs(a.v_max.value))
    # EB FEM: nodal values exact; between nodes the Hermite cubic cannot represent the quartic exact curve,
    # so only interpolation-level agreement (1e-4 of v_max for 8 elements) is expected there.
    r = fem.solve(ss_udl, n_elements=8)
    assert np.allclose(r.v_nodes, f(r.mesh.nodes), rtol=0, atol=1e-12)
    assert np.allclose(r.v, f(r.x), rtol=0, atol=1e-4 * abs(r.v_max.value))


# =========================================================================== 3. section properties
def test_section_rect():
    s = Section.rect(B, H)
    p = sections.properties(s)
    assert math.isclose(p.A, B * H) and math.isclose(p.I, I_RECT) and math.isclose(p.Z, Z_RECT)
    assert math.isclose(p.r_g, H / math.sqrt(12))
    assert p.y_top == H / 2 and p.y_bottom == -H / 2 and math.isclose(p.kappa, 5 / 6)
    assert math.isclose(p.Q_max, B * H**2 / 8)
    V = 1000.0  # tau_max = 1.5 V/A at the neutral axis, 0 at the fibres
    assert math.isclose(sections.shear_stress(s, V, 0.0), 1.5 * V / (B * H))
    assert sections.shear_stress(s, V, H / 2) == pytest.approx(0.0, abs=1e-9)


def test_section_solid_circle():
    d = 0.05
    s = Section.solid_circ(d)
    assert math.isclose(sections.area(s), math.pi * d**2 / 4)
    assert math.isclose(sections.second_moment(s), math.pi * d**4 / 64)
    assert math.isclose(sections.section_modulus(s), math.pi * d**3 / 32)
    assert math.isclose(sections.radius_of_gyration(s), d / 4)
    V = 1000.0  # tau_max = 4V/3A
    assert math.isclose(sections.shear_stress(s, V, 0.0), 4 * V / (3 * sections.area(s)))


def test_section_hollow_circle():
    D, d = 0.06, 0.05
    s = Section.hollow_circ(D, d)
    assert math.isclose(sections.area(s), math.pi * (D**2 - d**2) / 4)
    assert math.isclose(sections.second_moment(s), math.pi * (D**4 - d**4) / 64)
    assert math.isclose(sections.section_modulus(s), math.pi * (D**4 - d**4) / (32 * D))
    assert math.isclose(sections.radius_of_gyration(s), math.sqrt(D**2 + d**2) / 4)
    # Q at the neutral axis = (D^3 - d^3)/12 ; b = D - d
    assert math.isclose(sections.first_moment(s, 0.0), (D**3 - d**3) / 12)
    assert 0.5 < sections.shear_correction_factor(s) < 0.9


# =========================================================================== 4. element stiffness
def test_eb_element_values_symmetry_rank():
    Le = 0.7
    k = fem.eb_element_stiffness(E, I_RECT, Le)
    c = EI / Le**3
    exp = c * np.array([[12, 6 * Le, -12, 6 * Le], [6 * Le, 4 * Le**2, -6 * Le, 2 * Le**2],
                        [-12, -6 * Le, 12, -6 * Le], [6 * Le, 2 * Le**2, -6 * Le, 4 * Le**2]])
    assert np.allclose(k, exp, rtol=1e-14)
    assert np.allclose(k, k.T)
    assert np.linalg.matrix_rank(k) == 2
    assert np.all(np.linalg.eigvalsh(k) > -1e-6 * c)  # positive semidefinite
    for rigid in ([1, 0, 1, 0], [0, 1, Le, 1]):  # translation, rotation
        assert np.allclose(k @ rigid, 0, atol=1e-9 * c)


def test_timoshenko_element_and_eb_limit():
    Le, G, A, kap = 0.5, 77e9, B * H, 5 / 6
    phi = 12 * EI / (kap * G * A * Le**2)
    k = fem.timoshenko_element_stiffness(E, I_RECT, Le, G, A, kap)
    assert math.isclose(k[1, 1], EI / ((1 + phi) * Le**3) * (4 + phi) * Le**2)
    assert math.isclose(k[1, 3], EI / ((1 + phi) * Le**3) * (2 - phi) * Le**2)
    assert np.allclose(k, k.T) and np.linalg.matrix_rank(k) == 2
    k_inf = fem.timoshenko_element_stiffness(E, I_RECT, Le, 1e30, A, kap)
    assert np.allclose(k_inf, fem.eb_element_stiffness(E, I_RECT, Le), rtol=1e-12)


def test_frame_element():
    Le, A = 0.5, B * H
    k = fem.frame_element_stiffness(E, A, I_RECT, Le)
    assert np.allclose(k, k.T) and np.linalg.matrix_rank(k) == 3
    assert math.isclose(k[0, 0], E * A / Le) and math.isclose(k[0, 3], -E * A / Le)
    assert np.allclose(k[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])], fem.eb_element_stiffness(E, I_RECT, Le))
    assert np.all(k[np.ix_([0, 3], [1, 2, 4, 5])] == 0)  # axial / bending uncoupled
    assert np.allclose(k @ [1, 0, 0, 1, 0, 0], 0)  # rigid axial translation


def test_consistent_load_vector():
    Le, q = 0.8, -500.0
    assert np.allclose(fem.consistent_load(q, q, Le), [q * Le / 2, q * Le**2 / 12, q * Le / 2, -q * Le**2 / 12])
    # triangular 0 -> q: [3qL/20, qL^2/30, 7qL/20, -qL^2/20]
    assert np.allclose(fem.consistent_load(0, q, Le),
                       [3 * q * Le / 20, q * Le**2 / 30, 7 * q * Le / 20, -q * Le**2 / 20])


# =========================================================================== 5. global assembly (2 elements by hand)
def test_global_assembly_two_elements(cantilever_end_load):
    mesh = fem.build_mesh(cantilever_end_load, 2)
    assert np.allclose(mesh.nodes, [0, 1, 2])
    K, F = fem.assemble(cantilever_end_load, mesh)
    Le = 1.0
    c = EI / Le**3
    a, b2, b4 = 6 * Le, 2 * Le**2, 4 * Le**2
    exp = c * np.array([
        [12, a, -12, a, 0, 0],
        [a, b4, -a, b2, 0, 0],
        [-12, -a, 24, 0, -12, a],
        [a, b2, 0, 2 * b4, -a, b2],
        [0, 0, -12, -a, 12, -a],
        [0, 0, a, b2, -a, b4],
    ])
    assert np.allclose(K, exp, rtol=1e-13)
    assert np.allclose(F, [0, 0, 0, 0, -P, 0])
    # hand solve of the reduced 4x4 system gives the exact tip values
    U, R, _ = fem.solve_linear(K, F, fem.restrained_dofs(cantilever_end_load, mesh))
    assert close(U[4], -P * L**3 / (3 * EI), 1e-12) and close(U[5], -P * L**2 / (2 * EI), 1e-12)
    assert close(R[0], P, 1e-12) and close(R[1], P * L, 1e-12)


def test_global_assembly_udl_load_vector(ss_udl):
    mesh = fem.build_mesh(ss_udl, 2)
    _, F = fem.assemble(ss_udl, mesh)
    Le = 1.0
    q = -W
    assert np.allclose(F, [q * Le / 2, q * Le**2 / 12, q * Le, 0, q * Le / 2, -q * Le**2 / 12])


# =========================================================================== 6. boundary conditions
def test_restrained_dofs_beam_and_frame():
    m = beam([Support("fixed", 0.0), Support("pin", 1.0), Support("roller", L)], [PointLoad(0.5, P)])
    mesh = fem.build_mesh(m, 2)
    assert np.allclose(mesh.nodes, [0, 0.5, 1, 2])  # node forced at the load
    assert list(fem.restrained_dofs(m, mesh)) == [0, 1, 4, 6]  # v0, th0 | v(node 2) | v(node 3)
    assert list(fem.restrained_dofs(m, mesh, frame=True)) == [0, 1, 2, 6, 7, 10]  # u0 v0 th0 | u2 v2 | v3


@pytest.mark.parametrize("name", ["cantilever_end_load", "ss_udl"])
def test_bc_displacements_zero_and_reactions_only_at_supports(name, request):
    m = request.getfixturevalue(name)
    r = fem.solve(m, n_elements=10)
    fixed = fem.restrained_dofs(m, r.mesh)
    assert np.allclose(r.U[fixed], 0)
    sup_x = {s.x for s in m.supports}
    assert {rx.x for rx in r.reactions} == sup_x
    if name == "ss_udl":  # pinned ends rotate: theta(0) = -wL^3/24EI
        assert close(r.theta_nodes[0], W * L**3 / (24 * EI)) and r.theta_nodes[0] < 0


def test_roller_does_not_resist_axial():
    """Pin + roller under an axial load: the pin alone takes Fx, the roller Fx is zero."""
    m = beam(ss(), [PointLoad(L / 2, P, 0.0), PointLoad(L / 2, P)])
    r = analyze(m)
    pin, roller = sorted(r.reactions, key=lambda x: x.x)
    assert math.isclose(pin.Fx, -P, rel_tol=1e-9) and abs(roller.Fx) < 1e-9 * P


# =========================================================================== 7. mechanism detection
@pytest.mark.parametrize("supports, loads", [
    ([Support("roller", 1.0)], [PointLoad(L, P)]),  # one roller: rotation free
    ([Support("pin", 0.0)], [PointLoad(L, P)]),  # pin alone: rotation free
    ([Support("roller", 0.0), Support("roller", L)], [PointLoad(L / 2, P, -45.0)]),  # no axial restraint
])
def test_mechanism_detected(supports, loads):
    m = beam(supports, loads)
    with pytest.raises(UnstableStructureError):
        analyze(m)


def test_mechanism_detected_by_fem_solver():
    m = beam([Support("roller", 1.0)], [PointLoad(L, P)])
    with pytest.raises(UnstableStructureError, match="Mechanism"):
        fem.solve(m)


@pytest.mark.parametrize("supports", [[], [Support("pin", 1.0), Support("roller", 1.0)]])
def test_no_support_or_coincident_supports_rejected(supports):
    # rejected by validation before stability is even evaluated
    with pytest.raises(ModelError):
        analyze(beam(supports, [PointLoad(L, P)]))


# =========================================================================== 8. indeterminate beams
def test_propped_cantilever_udl():
    m = beam([Support("fixed", 0.0), Support("roller", L)], [DistributedLoad(0, L, W, W)])
    for r in (analyze(m), fem.solve(m, n_elements=40)):
        assert close(fy_at(r.reactions, L), 3 * W * L / 8)
        assert close(fy_at(r.reactions, 0.0), 5 * W * L / 8)
        wall = next(x for x in r.reactions if x.type == "fixed")
        assert close(wall.M, W * L**2 / 8)
        assert close(r.M_max.value, W * L**2 / 8)
        # max sagging 9wL^2/128 at x = 5L/8
        assert close(np.max(r.M), 9 * W * L**2 / 128)
        # exact max deflection: wL^4/EI * (39 + 55 sqrt 33)/65536 at x = (15 - sqrt33)/16 L
        assert close(r.v_max.value, W * L**4 / EI * (39 + 55 * math.sqrt(33)) / 65536)
        assert math.isclose(r.v_max.x, (15 - math.sqrt(33)) / 16 * L, abs_tol=0.01)
    assert analyze(m).method == "fem-euler"


def test_propped_cantilever_central_point():
    m = beam([Support("fixed", 0.0), Support("roller", L)], [PointLoad(L / 2, P)])
    r = analyze(m)
    assert close(fy_at(r.reactions, L), 5 * P / 16)
    assert close(next(x for x in r.reactions if x.type == "fixed").M, 3 * P * L / 16)
    assert close(r.v_max.value, P * L**3 / (48 * math.sqrt(5) * EI))  # at x = L/sqrt5 from the roller


def test_fixed_fixed_beams():
    m = beam([Support("fixed", 0.0), Support("fixed", L)], [PointLoad(L / 2, P)])
    r = analyze(m)
    assert close(fy_at(r.reactions, 0), P / 2) and close(fy_at(r.reactions, L), P / 2)
    assert close(r.M_max.value, P * L / 8)
    assert close(r.v_max.value, P * L**3 / (192 * EI))
    walls = sorted(r.reactions, key=lambda x: x.x)
    assert close(walls[0].M, P * L / 8) and walls[0].M > 0 > walls[1].M  # equal and opposite
    m = beam([Support("fixed", 0.0), Support("fixed", L)], [DistributedLoad(0, L, W, W)])
    r = analyze(m)
    assert close(r.M_max.value, W * L**2 / 12) and r.M_max.value < 0  # hogging at the walls
    assert close(np.max(r.M), W * L**2 / 24)
    assert close(r.v_max.value, W * L**4 / (384 * EI))


def test_continuous_two_span():
    m = beam([Support("pin", 0.0), Support("roller", L / 2), Support("roller", L)], [DistributedLoad(0, L, W, W)])
    r = analyze(m)
    s = L / 2
    assert close(fy_at(r.reactions, 0), 3 * W * s / 8) and close(fy_at(r.reactions, s), 10 * W * s / 8)
    assert close(r.M_max.value, W * s**2 / 8)


# =========================================================================== 9. frame element: axial
def test_frame_axial_tension_and_compression():
    A = B * H
    for angle, sign in ((0.0, +1), (180.0, -1)):
        m = beam([Support("fixed", 0.0)], [PointLoad(L, P, angle)])
        r = analyze(m)
        assert r.method == "fem-frame"
        assert close(r.fem.u_nodes[-1], sign * P * L / (E * A), 1e-9)
        assert np.allclose(r.N, sign * P, rtol=1e-9)
        assert close(r.reactions[0].Fx, -sign * P, 1e-9)
        assert np.allclose(r.v, 0, atol=1e-15)  # no bending
        assert close(r.sigma_max.value, P / A, 1e-9)


def test_frame_inclined_load_superposition():
    A = B * H
    th = math.radians(-30.0)
    m = beam([Support("fixed", 0.0)], [PointLoad(L, P, -30.0)])
    r = fem.solve(m, n_elements=8)
    Px, Py = P * math.cos(th), P * math.sin(th)
    assert close(r.u_nodes[-1], Px * L / (E * A), 1e-9)
    assert close(r.v_nodes[-1], Py * L**3 / (3 * EI), 1e-9)
    # combined fibre stress at the wall: N/A -/+ M c / I with M(0) = Py L (hogging -> top in tension)
    assert close(r.sigma_top[0], Px / A + abs(Py) * L / Z_RECT, 1e-9)


# =========================================================================== 10. Timoshenko
def _shear_term(m, L_):
    s, mat = m.section, m.material
    return P * L_ / (sections.shear_correction_factor(s, mat.nu) * mat.G * sections.area(s))


def test_timoshenko_cantilever_eb_plus_shear():
    for L_ in (0.1, 0.3, 2.0):
        m = beam([Support("fixed", 0.0)], [PointLoad(L_, P)], L=L_)
        r = fem.solve(m, n_elements=4, theory="timoshenko")
        assert close(r.v_max.value, P * L_**3 / (3 * EI) + _shear_term(m, L_), 1e-9)


def test_timoshenko_converges_to_eb_as_slenderness_grows():
    diffs = []
    for ratio in (2, 5, 10, 20, 50, 100):
        L_ = ratio * H
        m = beam([Support("fixed", 0.0)], [PointLoad(L_, P)], L=L_)
        eb = fem.solve(m, n_elements=4, theory="euler").v_max.value
        tm = fem.solve(m, n_elements=4, theory="timoshenko").v_max.value
        d = (tm - eb) / eb
        assert close(d, _shear_term(m, L_) / (P * L_**3 / (3 * EI)), 1e-6)  # = 3EI/(kGAL^2)
        diffs.append(d)
    assert all(a > b > 0 for a, b in zip(diffs, diffs[1:]))
    assert diffs[-1] < 1e-3
    # log-log slope of the shear share vs L/h is -2
    assert lab.loglog_slope([2, 5, 10, 20, 50, 100], diffs) == pytest.approx(-2.0, abs=1e-6)


def test_timoshenko_ss_udl_exact():
    """SS UDL Timoshenko midspan: 5wL^4/384EI + wL^2/(8 kGA)."""
    L_ = 0.3
    m = beam(ss()[:1] + [Support("roller", L_)], [DistributedLoad(0, L_, W, W)], L=L_)
    s, mat = m.section, m.material
    exp = 5 * W * L_**4 / (384 * EI) + W * L_**2 / (8 * 5 / 6 * mat.G * sections.area(s))
    assert close(fem.solve(m, n_elements=2, theory="timoshenko").v_max.value, exp, 1e-9)


# =========================================================================== 11. buckling
@pytest.mark.parametrize("ec, K", [("fixed-free", 2.0), ("pinned-pinned", 1.0), ("fixed-pinned", 0.699),
                                    ("fixed-fixed", 0.5)])
def test_buckling_pcr(ec, K):
    m = beam([Support("fixed", 0.0)], [PointLoad(L, 5000.0, 180.0)])
    r = safety.buckling(m, end_condition=ec, P_applied=5000.0)
    I_min = H * B**3 / 12  # weak axis
    pcr = math.pi**2 * E * I_min / (K * L)**2
    assert r.K == K and close(r.L_eff, K * L, 1e-12) and close(r.Pcr, pcr, 1e-12)
    assert close(r.fos, pcr / 5000.0, 1e-12)
    assert close(r.slenderness, K * L / math.sqrt(I_min / (B * H)), 1e-12)


def test_buckling_fixed_pinned_K_is_physical():
    """K = pi / (kL), tan(kL) = kL."""
    kL = 4.493409457909064
    assert math.isclose(math.tan(kL), kL, rel_tol=1e-9)
    assert safety.BUCKLING_K["fixed-pinned"] == pytest.approx(math.pi / kL, abs=5e-4)


@pytest.mark.parametrize("supports, ec", [
    ([Support("fixed", 0.0)], "fixed-free"),
    ([Support("fixed", L)], "fixed-free"),
    ([Support("pin", 0.0), Support("roller", L)], "pinned-pinned"),
    ([Support("fixed", 0.0), Support("roller", L)], "fixed-pinned"),
    ([Support("fixed", 0.0), Support("fixed", L)], "fixed-fixed"),
    ([Support("pin", 0.5), Support("roller", L)], None),
])
def test_buckling_end_condition_inference(supports, ec):
    assert safety.infer_end_condition(beam(supports, [PointLoad(1.0, P)])) == ec


def test_buckling_compression_read_from_model_and_status():
    m = beam([Support("fixed", 0.0)], [PointLoad(L, 20000.0, 180.0)])
    r = safety.buckling(m)
    assert close(r.P_applied, 20000.0, 1e-9) and r.applicable and r.end_condition == "fixed-free"
    pcr = math.pi**2 * E * (H * B**3 / 12) / (2 * L)**2  # 16.65 kN < 20 kN
    assert close(r.Pcr, pcr) and r.fos < 1
    rep = safety.assess(m, analyze(m))
    assert rep.status == safety.STATUS_BUCKLING  # instability, not "yield"
    tension = beam([Support("fixed", 0.0)], [PointLoad(L, 2000.0, 0.0)])
    assert safety.buckling(tension).applicable is False


# =========================================================================== 12. fatigue
def test_fatigue_refuses_without_data():
    f = safety.fatigue(200e6, 0.0, uts=None)
    assert f.cycles_to_failure is None and f.goodman_fos is None and f.infinite_life is None
    assert f.sigma_ar is None and len(f.sn_N) == 0
    assert sum("Insufficient data" in w for w in f.warnings) >= 2
    f = safety.fatigue(200e6, 0.0, uts=400e6)  # mean correction possible, life still not
    assert f.cycles_to_failure is None and close(f.sigma_ar, 100e6 / (1 - 100e6 / 400e6))
    assert any("Insufficient data" in w for w in f.warnings)


def test_fatigue_with_user_data():
    sf, b, Se, uts = 900e6, -0.1, 100e6, 400e6  # sigma_ar 133 MPa > Se: finite life
    f = safety.fatigue(200e6, 0.0, uts, sigma_f_prime=sf, b=b, endurance_limit=Se)
    sar = 100e6 / (1 - 0.25)
    assert close(f.sigma_ar, sar, 1e-12)
    assert close(f.cycles_to_failure, 0.5 * (sar / sf) ** (1 / b), 1e-12)
    assert close(f.goodman_fos, 1 / (100e6 / Se + 100e6 / uts), 1e-12)
    assert f.infinite_life is False
    with pytest.raises(ValueError):
        safety.fatigue(100e6, 0.0, uts, sigma_f_prime=sf, b=0.1)  # b must be negative


# =========================================================================== 13. safety status thresholds
@pytest.mark.parametrize("fos, status", [(0.5, "YIELD PREDICTED"), (0.999, "YIELD PREDICTED"), (1.0, "WARNING"),
                                         (1.49, "WARNING"), (1.5, "SAFE"), (10.0, "SAFE")])
def test_status_thresholds(fos, status):
    assert safety.status_for(fos, 1.5) == status


@pytest.mark.parametrize("target_fos, status", [(0.8, "YIELD PREDICTED"), (1.2, "WARNING"), (3.0, "SAFE")])
def test_yield_check_on_real_beam(target_fos, status):
    sy = get_material("structural_steel").yield_strength
    Pt = sy * Z_RECT / (L * target_fos)  # sigma = PL/Z
    m = beam([Support("fixed", 0.0)], [PointLoad(L, Pt)])
    y = safety.yield_check(analyze(m), min_fos=1.5)
    assert close(y.fos, target_fos) and y.status == status
    assert close(y.first_yield_load_factor, target_fos)
    with pytest.raises(ValueError):
        safety.yield_check(analyze(m), min_fos=0.9)


def test_von_mises_critical_point(ss_central_load):
    c = safety.von_mises_check(analyze(ss_central_load))
    # max at mid-span fibre (tau = 0 there): sigma_vm = PL/4Z
    assert close(np.max(c.von_mises), P * L / 4 / Z_RECT)


# =========================================================================== 14. optimization
def _ok(model, prob):
    r = analyze(model)
    fos = model.material.yield_strength / abs(r.sigma_max.value)
    return abs(r.v_max.value) <= prob.delta_allow * (1 + 1e-9) and fos >= prob.fos_min * (1 - 1e-9)


def test_optimization_closed_form_cantilever(cantilever_end_load):
    prob = OptimizationProblem(objective="volume", delta_allow=0.004, fos_min=2.0,
                               bounds={"b": (0.02, 0.10), "h": (0.02, 0.20)})
    res = optimize(cantilever_end_load, prob)
    assert res.status == "OPTIMIZED" and res.feasible
    b = res.optimal_dims["b"]
    assert b == pytest.approx(0.02, rel=1e-3)  # volume always falls with b -> lower bound
    sy = 250e6
    h_def = (4 * P * L**3 / (E * b * prob.delta_allow)) ** (1 / 3)
    h_str = math.sqrt(6 * P * L * prob.fos_min / (b * sy))
    assert res.optimal_dims["h"] == pytest.approx(max(h_def, h_str), rel=1e-3)
    # independent re-verification + every reported check satisfied + bounds respected
    assert _ok(res.optimized_model, prob) and all(c.satisfied for c in res.checks)
    for k, (lo, hi) in prob.bounds.items():
        assert lo * (1 - 1e-9) <= res.optimal_dims[k] <= hi * (1 + 1e-9)


@pytest.mark.parametrize("section, bounds", [
    (Section.solid_circ(0.05), {"d": (0.01, 0.2)}),
    (Section.hollow_circ(0.06, 0.05), {"D": (0.02, 0.2), "d": (0.005, 0.19)}),
])
def test_optimization_other_shapes_feasible(section, bounds):
    m = beam(ss(), [PointLoad(L / 2, 3000.0)], section=section)
    prob = OptimizationProblem(objective="mass", delta_allow=L / 360, fos_min=1.5, bounds=bounds)
    res = optimize(m, prob)
    assert res.status == "OPTIMIZED"
    assert _ok(res.optimized_model, prob)
    if section.shape == "hollow_circ":
        s = res.optimized_model.section
        assert (s.D - s.d) / 2 >= prob.t_min * (1 - 1e-9)
    if section.shape == "solid_circ":  # closed form: d from max(deflection, stress)
        d_def = (64 * 3000 * L**3 / (48 * E * math.pi * prob.delta_allow)) ** 0.25
        d_str = (32 * 3000 * L / 4 * 1.5 / (math.pi * 250e6)) ** (1 / 3)
        assert res.optimal_dims["d"] == pytest.approx(max(d_def, d_str), rel=1e-3)


def test_optimization_infeasible_detected(cantilever_end_load):
    prob = OptimizationProblem(delta_allow=1e-5, fos_min=1.5, bounds={"b": (0.01, 0.04), "h": (0.01, 0.08)})
    res = optimize(cantilever_end_load, prob)
    assert res.status == "INFEASIBLE" and not res.feasible
    assert res.infeasibility_reasons
    # truly infeasible: the largest allowed section still fails
    big = cantilever_end_load.copy()
    big.section = Section.rect(0.04, 0.08)
    assert not _ok(big, prob)


# =========================================================================== 15. advisor
def _verify_rec(rec, delta_allow, min_fos):
    m = rec.modified_model
    r = analyze(m)
    assert abs(r.v_max.value) <= delta_allow * (1 + 1e-9)
    assert m.material.yield_strength / abs(r.sigma_max.value) >= min_fos * (1 - 1e-9)
    bk = safety.buckling(m)
    if bk.applicable:
        assert bk.fos >= 1.0


@pytest.mark.parametrize("load, delta_allow, limiting", [
    (1000.0, 0.001, "deflection"),  # default 30x60 cantilever: delta 3.09 mm >> 1 mm
    (6000.0, 1.0, "stress"),  # sigma = 6000*2/1.8e-5 = 667 MPa > 250
])
def test_advisor_fixes_pass_after_reanalysis(load, delta_allow, limiting):
    m = beam([Support("fixed", 0.0)], [PointLoad(L, load)])
    rep = advisor.advise(m, delta_allow=delta_allow, min_fos=1.5)
    assert rep.limiting == limiting
    assert rep.utilisation[limiting] > 1
    assert rep.recommendations
    for rec in rep.recommendations:
        assert rec.satisfies_all
        _verify_rec(rec, delta_allow, 1.5)


def test_advisor_buckling_fix():
    m = beam([Support("fixed", 0.0)], [PointLoad(L, 20000.0, 180.0)])  # Pcr = 16.65 kN
    rep = advisor.advise(m, min_fos=1.5)
    assert rep.limiting == "buckling"
    for rec in rep.recommendations:
        assert rec.satisfies_all
        _verify_rec(rec, L / advisor.DEFAULT_SPAN_RATIO, 1.5)
        assert safety.buckling(rec.modified_model).fos >= 1.5 * (1 - 1e-9)


def test_advisor_safe_design_no_violation():
    m = beam([Support("fixed", 0.0)], [PointLoad(L, 100.0)], section=Section.rect(0.03, 0.08))
    rep = advisor.advise(m, delta_allow=L / 250)
    assert rep.limiting in ("none", "material excess")
    assert all(u <= 1 for u in rep.utilisation.values())


# =========================================================================== 16. sensitivity exponents
def test_sensitivity_exponents(cantilever_end_load):
    t = lab.tornado(cantilever_end_load, pct=10.0)
    el = {(r.param, r.output): r.elasticity for r in t.itertuples()}
    expected = {("L", "delta_max"): 3, ("h", "delta_max"): -3, ("b", "delta_max"): -1, ("E", "delta_max"): -1,
                ("load", "delta_max"): 1, ("L", "sigma_max"): 1, ("h", "sigma_max"): -2, ("E", "sigma_max"): 0}
    for k, v in expected.items():
        assert el[k] == pytest.approx(v, abs=1e-6), k


@pytest.mark.parametrize("param, slope", [("L", 3.0), ("h", -3.0)])
def test_sensitivity_sweep_loglog(cantilever_end_load, param, slope):
    df = lab.sensitivity_sweep(cantilever_end_load, param)
    assert len(df) == 16
    assert lab.loglog_slope(df["value"], df["delta_max"]) == pytest.approx(slope, abs=1e-6)


def test_udl_deflection_scales_L4(cantilever_udl):
    df = lab.sensitivity_sweep(cantilever_udl, "L")
    assert lab.loglog_slope(df["value"], df["delta_max"]) == pytest.approx(4.0, abs=1e-6)


# =========================================================================== 17. lab challenge grading
@pytest.mark.parametrize("ch", lab.CHALLENGES, ids=lambda c: c.key)
def test_challenge_reference_passes_and_tiny_fails(ch):
    shape = ch.shapes[0]
    ref = lab.reference_optimum(ch, shape, ch.base_model.material)
    assert ref.status == "OPTIMIZED"
    g = lab.grade_challenge(ch, ref.optimized_model)
    assert g.passed and g.score >= 99.0
    # independent check of the graded numbers
    r = analyze(lab.challenge_design(ch, ref.optimized_model))
    if ch.delta_allow:
        assert abs(r.v_max.value) <= ch.delta_allow * (1 + 1e-9)
    # a clearly undersized section fails with feedback and a low score
    lo = {k: v[0] for k, v in lab.challenge_bounds(ch, shape).items()}
    if shape == "hollow_circ":
        lo["d"] = lo["D"] - 2 * (ch.min_wall or 1e-3)
    bad = ch.base_model.copy()
    bad.section = Section(shape, **lo)
    gb = lab.grade_challenge(ch, bad)
    assert not gb.passed and gb.score < 50 and gb.feedback


def test_challenge_cannot_change_problem():
    ch = lab.CHALLENGES[0]
    cheat = ch.base_model.copy()
    cheat.L = 0.1
    cheat.loads = []
    d = lab.challenge_design(ch, cheat)
    assert d.L == ch.base_model.L and len(d.loads) == len(ch.base_model.loads)


# =========================================================================== 18. playground converter
PL, PB, PH = 2.0, 0.040, 0.080
P_EI = E * PB * PH**3 / 12
P_Z = PB * PH**2 / 6


def cfg(key):
    return convert.preset_configs()[key]["config"]


def run(c):
    return analyze(convert.config_to_model(c))


PRESET_EXPECT = {  # key: (R_left, R_right, |M_max|, |delta_max|)
    "cantilever_end": (1000.0, None, 1000.0 * PL, 1000.0 * PL**3 / (3 * P_EI)),
    "cantilever_udl": (1000.0 * PL, None, 1000.0 * PL**2 / 2, 1000.0 * PL**4 / (8 * P_EI)),
    "ss_central": (1000.0, 1000.0, 2000.0 * PL / 4, 2000.0 * PL**3 / (48 * P_EI)),
    "ss_udl": (2000.0, 2000.0, 2000.0 * PL**2 / 8, 5 * 2000.0 * PL**4 / (384 * P_EI)),
}


@pytest.mark.parametrize("key", list(PRESET_EXPECT))
def test_playground_textbook_presets(key):
    Rl, Rr, M, d = PRESET_EXPECT[key]
    r = run(cfg(key))
    assert close(fy_at(r.reactions, 0.0), Rl)
    if Rr is not None:
        assert close(fy_at(r.reactions, PL), Rr)
    assert close(r.M_max.value, M) and close(r.v_max.value, d)
    assert close(r.sigma_max.value, M / P_Z)


def test_playground_partial_udl_preset():
    # 3 kN/m over 0.5..1.25: F = 2250 N at 0.875 m
    F, xc = 3000.0 * 0.75, 0.875
    r = run(cfg("ss_partial_udl"))
    Rb = F * xc / PL
    assert close(fy_at(r.reactions, PL), Rb) and close(fy_at(r.reactions, 0), F - Rb)
    # M max where V = 0: x = 0.5 + Ra/w
    Ra = F - Rb
    xm = 0.5 + Ra / 3000.0
    assert close(r.M_max.value, Ra * xm - 3000.0 * (xm - 0.5) ** 2 / 2)


def _ss_point_deflection(Pv, a, x):
    """SS beam, point load Pv at a, deflection at x (positive down), any x."""
    b = PL - a
    if x <= a:
        return Pv * b * x * (PL**2 - b**2 - x**2) / (6 * PL * P_EI)
    return _ss_point_deflection(Pv, PL - a, PL - x)


def test_playground_multi_point_superposition():
    r = run(cfg("ss_multi_point"))
    loads = [(1000.0, 0.4), (2000.0, 1.0), (1500.0, 1.6)]
    for xq in (0.3, 0.8, 1.0, 1.7):
        exp = -sum(_ss_point_deflection(p, a, xq) for p, a in loads)
        assert np.interp(xq, r.x, r.v) == pytest.approx(exp, rel=1e-3)
    assert close(fy_at(r.reactions, 0), sum(p * (PL - a) / PL for p, a in loads))


def test_playground_position_and_magnitude_editing():
    c = cfg("ss_central")
    c["loads"][0]["x"] = 0.5
    c["loads"][0]["magnitude"] = 4000.0
    r = run(c)
    assert close(fy_at(r.reactions, 0), 4000.0 * 1.5 / 2) and close(fy_at(r.reactions, PL), 4000.0 * 0.5 / 2)
    assert close(r.M_max.value, 4000.0 * 0.5 * 1.5 / PL)
    assert close(r.M_max.x, 0.5, 1e-9)


def test_playground_direction_flips():
    down = run(cfg("ss_central"))
    c = cfg("ss_central")
    c["loads"][0]["angle_deg"] = 90.0
    up = run(c)
    assert np.allclose(up.v, -down.v, atol=1e-15) and np.allclose(up.M, -down.M, atol=1e-9)
    assert close(fy_at(up.reactions, 0), -1000.0) and fy_at(up.reactions, 0) < 0
    c = cfg("ss_udl")
    c["loads"][0]["direction"] = "up"
    assert run(c).v_max.value > 0
    c = cfg("cantilever_end")  # moment: CW vs CCW flips the response
    c["loads"] = [{"kind": "moment", "x": PL, "M": 500.0}]
    r_ccw = run(c)
    c["loads"][0]["M"] = -500.0
    r_cw = run(c)
    assert close(r_ccw.v_max.value, 500.0 * PL**2 / (2 * P_EI)) and r_ccw.v_max.value > 0
    assert np.allclose(r_cw.v, -r_ccw.v, atol=1e-15)


def test_playground_multiple_loads_superpose():
    c1, c2 = cfg("ss_central"), cfg("ss_udl")
    both = cfg("ss_central")
    both["loads"] = c1["loads"] + c2["loads"]
    r1, r2, r12 = run(c1), run(c2), run(both)
    xs = np.linspace(0, PL, 9)
    for f in ("v", "theta"):
        a = np.interp(xs, r1.x, getattr(r1, f)) + np.interp(xs, r2.x, getattr(r2, f))
        assert np.allclose(np.interp(xs, r12.x, getattr(r12, f)), a, rtol=1e-6, atol=1e-12)


def test_playground_triangular_and_partial_distributed():
    c = cfg("ss_udl")
    c["loads"] = [{"kind": "distributed", "x1": 0.0, "x2": PL, "w1": 0.0, "w2": 3000.0, "direction": "down"}]
    r = run(c)
    wL = 3000.0 * PL
    assert close(fy_at(r.reactions, 0), wL / 6) and close(fy_at(r.reactions, PL), wL / 3)
    assert close(r.M_max.value, 3000.0 * PL**2 / (9 * math.sqrt(3)))
    assert r.M_max.x == pytest.approx(PL / math.sqrt(3), abs=PL / 400)  # located on the ~L/500 grid


def test_playground_support_movement_overhang():
    c = cfg("ss_central")
    c["supports"][1]["x"] = 1.5
    c["loads"] = [{"kind": "point", "x": PL, "magnitude": 1000.0, "angle_deg": -90.0}]
    r = run(c)
    # moments about the pin: Rb*1.5 = 1000*2
    assert close(fy_at(r.reactions, 1.5), 1000.0 * 2 / 1.5) and close(fy_at(r.reactions, 0), -1000.0 * 0.5 / 1.5)
    assert close(r.M_max.value, 1000.0 * 0.5) and r.M_max.value < 0
    # tip deflection of overhang a=0.5 past span l=1.5: P a^2 (l + a) / 3EI
    assert close(r.fem.v_max.value if r.fem else np.interp(PL, r.x, r.v), 1000.0 * 0.25 * 2.0 / (3 * P_EI))


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(L=-1.0),
    lambda c: c.update(L="abc"),
    lambda c: c.update(L=float("nan")),
    lambda c: c.pop("section"),
    lambda c: c["supports"].__setitem__(1, {"type": "roller", "x": 5.0}),  # outside the beam
    lambda c: c["supports"].__setitem__(1, {"type": "hinge", "x": 1.0}),  # unknown type
    lambda c: c["loads"].append({"kind": "spring", "x": 1.0}),
    lambda c: c["loads"].append({"kind": "distributed", "x1": 1.5, "x2": 0.5, "w1": 1, "w2": 1,
                                 "direction": "down"}),
    lambda c: c["loads"].append({"kind": "point", "x": 3.0, "magnitude": 1.0, "angle_deg": -90}),
    lambda c: c.update(section={"shape": "hollow_circ", "D": 0.05, "d": 0.06}),
    lambda c: c["section"].update(b=0.0),
    lambda c: c["material"].update(E=-1.0),
])
def test_playground_invalid_configs(mutate):
    c = cfg("ss_central")
    mutate(c)
    with pytest.raises(ModelError):
        convert.config_to_model(c)


def test_playground_mechanism_rejected():
    c = cfg("ss_central")
    c["supports"] = [{"type": "roller", "x": 0.0}]
    m = convert.config_to_model(c)  # data valid ...
    with pytest.raises(UnstableStructureError):  # ... but analysis refuses the mechanism
        analyze(m)


@pytest.mark.parametrize("key", list(convert.PRESETS))
def test_playground_save_restore_round_trip(key, tmp_path):
    c = cfg(key)
    c["loads"].append({"kind": "moment", "x": 0.7, "M": -250.0, "id": "lx"})
    convert.save_config(c, f"cfg {key}", tmp_path)
    back = convert.load_config(f"cfg {key}", tmp_path)
    m0, m1 = convert.config_to_model(c), convert.config_to_model(back)
    assert m0.to_dict() == m1.to_dict()
    assert np.allclose(analyze(m0).v, analyze(m1).v)
    # BeamModel JSON round trip as well
    assert type(m0).from_json(m0.to_json()).to_dict() == m0.to_dict()


def test_playground_live_action_handling():
    state = {}
    c = cfg("ss_central")
    assert convert.accept_message(state, None) is None
    assert convert.accept_message(state, {"config": c}) is None  # no nonce
    assert convert.accept_message(state, {"config": c, "action": "live", "nonce": 1, "version": 3}) == "live"
    assert state["playground_config"] is c and state["playground_seen"] == 3
    # the component re-returns its last value on every rerun: same nonce is ignored
    assert convert.accept_message(state, {"config": c, "action": "live", "nonce": 1, "version": 3}) is None
    assert convert.accept_message(state, {"config": c, "action": "analyse", "nonce": 2}) == "analyse"
    assert convert.accept_message(state, {"config": c, "action": "drag", "nonce": 3, "version": 4}) == "edit"
    assert state["playground_seen"] == 4
    # live re-analysis of the accepted config reproduces a fresh analysis
    c2 = dict(c, loads=[dict(c["loads"][0], x=0.6)])
    assert convert.accept_message(state, {"config": c2, "action": "live", "nonce": 4}) == "live"
    r = run(state["playground_config"])
    assert close(fy_at(r.reactions, PL), 2000.0 * 0.6 / PL)
    payload = convert.result_payload(r)
    import json
    json.dumps(payload)  # JSON safe


# =========================================================================== 19. reports
def test_report_pdf_bytes(ss_udl):
    res = analyze(ss_udl)
    pdf = build_pdf(ss_udl, analysis=res, fem=fem.solve(ss_udl, 8), safety=safety.assess(ss_udl, res))
    assert isinstance(pdf, bytes) and pdf.startswith(b"%PDF") and pdf.rstrip().endswith(b"%%EOF")
    assert build_pdf(ss_udl).startswith(b"%PDF")


def test_lab_report_pdf_bytes():
    er = lab.run_experiment("height_vs_stiffness")
    assert lab_report_pdf(er).startswith(b"%PDF")


# =========================================================================== 20. general FEM vs analytical
def test_general_overhang_fem_vs_analytical():
    m = beam([Support("pin", 0.3), Support("roller", 1.6)],
             [DistributedLoad(0.0, 1.2, 0.0, 2000.0), PointLoad(L, 700.0), MomentLoad.cw(1.0, 400.0)])
    a, f = analyze(m), fem.solve(m, n_elements=32)
    assert a.method == "analytical"
    for ra, rf in zip(sorted(a.reactions, key=lambda r: r.x), sorted(f.reactions, key=lambda r: r.x)):
        assert close(ra.Fy, rf.Fy, 1e-9)
    assert close(a.v_max.value, f.v_max.value) and close(a.M_max.value, f.M_max.value)
    for k in ("sum_Fx", "sum_Fy", "sum_M0"):
        assert abs(a.equilibrium[k]) < 1e-8, k
    # finite-difference checks of dM/dx = V, dV/dx = q: O(h^2) checker error only (h = L/500)
    assert a.equilibrium["dM_dx_minus_V"] < 1e-4 and a.equilibrium["dV_dx_minus_q"] < 1e-6


# =========================================================================== 21. regression: no round-off axial force
@pytest.mark.parametrize("angle, fx, fy", [(-90.0, 0.0, -1.0), (90.0, 0.0, 1.0), (0.0, 1.0, 0.0), (180.0, -1.0, 0.0),
                                           (270.0, 0.0, -1.0), (-180.0, -1.0, 0.0)])
def test_point_load_components_exact_at_right_angles(angle, fx, fy):
    ld = PointLoad(1.0, 1000.0, angle)
    assert ld.fx == 1000.0 * fx and ld.fy == 1000.0 * fy  # exactly, not cos(-90 deg) = 6e-17


def test_unstressed_beam_reports_infinite_fos():
    """Load directly over the wall: no bending, no axial force -> sigma = 0, FOS = inf (contract, metrics.py)."""
    from smartbeam.metrics import evaluate
    m = beam([Support("fixed", 0.0)], [PointLoad(0.0, P)])
    r = analyze(m)
    assert np.all(r.N == 0) and r.sigma_max.value == 0
    assert evaluate(m, r).fos == math.inf and safety.yield_check(r).fos == math.inf


# =========================================================================== 22. seeded random cross-checks
def _random_determinate(rng, L_=2.0):
    loads = []
    for _ in range(rng.integers(1, 4)):
        k = rng.integers(0, 3)
        if k == 0:
            loads.append(PointLoad(round(float(rng.uniform(0, L_)), 3), float(rng.uniform(100, 2000)),
                                   float(rng.choice([-90.0, 90.0]))))
        elif k == 1:
            a, b = sorted(rng.uniform(0, L_, 2).round(3))
            b = min(max(b, a + 0.05), L_)
            loads.append(DistributedLoad(float(a), float(b), float(rng.uniform(0, 2000)), float(rng.uniform(0, 2000)),
                                         str(rng.choice(["down", "up"]))))
        else:
            loads.append(MomentLoad(round(float(rng.uniform(0, L_)), 3), float(rng.uniform(-800, 800))))
    if rng.random() < 0.5:
        sup = [Support("fixed", float(rng.choice([0.0, L_])))]
    else:
        a, b = sorted(rng.uniform(0, L_, 2).round(3))
        if b - a < 0.2:
            a, b = 0.0, L_
        sup = [Support("pin", float(a)), Support("roller", float(b))]
    return sup, loads


@pytest.mark.parametrize("seed", range(40))
def test_random_determinate_analytical_equals_fem(seed):
    """Two independent engines (statics + integration vs stiffness method) agree on random determinate beams."""
    rng = np.random.default_rng(seed)
    m = beam(*_random_determinate(rng))
    a, f = analyze(m), fem.solve(m, n_elements=40)
    scale = max(1.0, np.max(np.abs(a.M)))
    for ra, rf in zip(sorted(a.reactions, key=lambda r: r.x), sorted(f.reactions, key=lambda r: r.x)):
        assert abs(ra.Fy - rf.Fy) + abs(ra.M - rf.M) < 1e-5 * scale
    vs = np.max(np.abs(a.v))
    assert np.max(np.abs(np.interp(a.x, f.x, f.v) - a.v)) < 1e-3 * vs
    xn = np.r_[f.mesh.nodes[:-1] + 1e-7, f.mesh.nodes[1:] - 1e-7]  # interior side of every node
    assert np.max(np.abs(np.interp(xn, a.x, a.M) - np.interp(xn, f.x, f.M))) < 1e-3 * scale


@pytest.mark.parametrize("seed", range(20))
def test_random_timoshenko_equals_eb_plus_shear_deflection(seed):
    """Determinate beam: v_T = v_EB + v_s with gamma = v_s' = -V/(kappa G A) (this sign convention), plus a rigid
    correction fixed by the supports (fixed end: v_s = 0 and v_s' = gamma; pin/roller: v_s = 0)."""
    from scipy.integrate import cumulative_trapezoid
    rng = np.random.default_rng(100 + seed)
    m = beam(*_random_determinate(rng, 1.0), L=1.0, section=Section.rect(0.03, 0.15))  # deep: shear matters
    eb, t = fem.solve(m, 24), fem.solve(m, 24, theory="timoshenko")
    kGA = 5 / 6 * m.material.G * B * 0.15
    a = analyze(m, n_points=4001)
    xn = eb.mesh.nodes
    vs = -np.interp(xn, a.x, cumulative_trapezoid(a.V, a.x, initial=0.0)) / kGA
    sup = m.supports
    if sup[0].type == "fixed":
        c1, c0 = 0.0, -vs[np.argmin(abs(xn - sup[0].x))]
    else:
        j1, j2 = (np.argmin(abs(xn - s.x)) for s in sup)
        c1 = -(vs[j2] - vs[j1]) / (xn[j2] - xn[j1])
        c0 = -vs[j1] - c1 * xn[j1]
    exp = eb.v_nodes + vs + c1 * xn + c0
    assert np.max(np.abs(t.v_nodes - exp)) < 1e-5 * np.max(np.abs(t.v_nodes))
