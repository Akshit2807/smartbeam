"""Classical solid-mechanics engine + the analysis FACADE `analyze()`. OWNER: analytical engineer.

The facade dispatch (implemented below, binding) never applies transverse Euler-Bernoulli formulas to
configurations they do not cover:
    invalid data            -> ModelError
    mechanism               -> UnstableStructureError
    transverse & determinate -> analytical path (statics + numerical integration), method='analytical'
    transverse & indeterminate -> fem.solve(theory='euler', frame=False), method='fem-euler'
    any axial/inclined load  -> fem.solve(theory='euler', frame=True),  method='fem-frame'
Signs: see docs/ARCHITECTURE.md (V = dM/dx, dV/dx = q (+y), sagging M +, tension N +, sigma = N/A - M y / I).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from smartbeam import sections
from smartbeam.model import (POS_TOL, BeamModel, DistributedLoad, Extreme, ModelError, MomentLoad, PointLoad,
                             Reaction, Stability, UnstableStructureError, check_stability, classify,
                             is_transverse_only, validate)


@dataclass
class TextbookCheck:
    case: str  # 'cantilever-end-load' | 'cantilever-udl' | 'ss-central-load' | 'ss-udl'
    formulas: dict[str, str]  # quantity -> formula text, e.g. {'delta_max': 'PL^3/(3EI)'}
    textbook: dict[str, float]  # quantity -> closed-form value (SI)
    numerical: dict[str, float] = field(default_factory=dict)  # same keys, from the engine
    rel_error: dict[str, float] = field(default_factory=dict)  # |num - tb| / |tb|


@dataclass
class AnalysisResult:
    model: BeamModel
    method: str  # 'analytical' | 'fem-euler' | 'fem-frame'
    # Sampled fields. x is non-decreasing; at concentrated forces/moments x is REPEATED (left value, then right
    # value) so jumps in V / M plot vertically.
    x: np.ndarray  # m
    V: np.ndarray  # N
    M: np.ndarray  # N*m
    N: np.ndarray  # N
    theta: np.ndarray  # rad
    v: np.ndarray  # m
    sigma_top: np.ndarray  # Pa
    sigma_bottom: np.ndarray  # Pa
    eps_top: np.ndarray  # sigma_top / E
    eps_bottom: np.ndarray  # sigma_bottom / E
    reactions: list[Reaction]
    V_max: Extreme
    M_max: Extreme
    N_max: Extreme
    theta_max: Extreme
    v_max: Extreme
    sigma_max: Extreme  # largest |sigma| of either fibre, where='top'|'bottom'
    sigma_tension_max: Extreme  # most positive sigma (0 if none)
    sigma_compression_max: Extreme  # most negative sigma (0 if none)
    stability: Stability
    equilibrium: dict[str, float]  # residuals {'sum_Fx', 'sum_Fy', 'sum_M0'} of loads + reactions
    textbook: TextbookCheck | None = None  # only for the four guarded textbook cases
    warnings: list[str] = field(default_factory=list)
    fem: object | None = None  # fem.FEMResult when the FEM route was used


def reactions(model: BeamModel) -> list[Reaction]:
    """Reactions from sum Fy = 0 and sum M = 0 (transverse, statically determinate only; Fx = 0).
    Raises ValueError for any other configuration instead of guessing.

    Unknowns are the restrained transverse DOFs (v -> force Fy, theta -> moment M) of every support; a
    determinate transverse beam has exactly two of them (one fixed support, or two pin/roller supports)."""
    if not is_transverse_only(model):
        raise ValueError("Axial/inclined loads present: static reactions here cover transverse loading only.")
    stab = check_stability(model, mode="transverse")
    if not stab.determinate:
        raise ValueError(f"Statics alone cannot give the reactions. {stab.message}")
    unknowns = [(i, dof) for i, sp in enumerate(model.supports) for dof in sp.restrained if dof in ("v", "theta")]
    # Row 0: sum Fy = 0.  Row 1: sum of CCW moments about x = 0 = 0  (force Fy at x gives x * Fy).
    A = np.array([[1.0, model.supports[i].x] if dof == "v" else [0.0, 1.0] for i, dof in unknowns]).T
    _, Fy, M0 = _applied_resultants(model)
    sol = np.linalg.solve(A, [-Fy, -M0])
    out = [Reaction(i, sp.type, sp.x) for i, sp in enumerate(model.supports)]
    for (i, dof), val in zip(unknowns, sol):
        if dof == "v":
            out[i].Fy = float(val)
        else:
            out[i].M = float(val)
    return out


def _applied_resultants(model: BeamModel) -> tuple[float, float, float]:
    """(sum Fx, sum Fy, sum CCW moment about x = 0) of the applied loads only (exact for linear UDLs)."""
    Fx = Fy = M0 = 0.0
    for ld in model.loads:
        if isinstance(ld, PointLoad):
            Fx += ld.fx
            Fy += ld.fy
            M0 += ld.x * ld.fy
        elif isinstance(ld, DistributedLoad):
            q1, q2 = ld.sign * ld.w1, ld.sign * ld.w2
            F, Fm = _linear_load_integrals(ld.x1, ld.x2, q1, q2)
            Fy += F
            M0 += Fm
        elif isinstance(ld, MomentLoad):
            M0 += ld.M
    return Fx, Fy, M0


def _linear_load_integrals(a, b, qa, qb):
    """For q linear from qa at a to qb at b: (int q dx, int q x dx). Works elementwise on arrays."""
    c = b - a
    return 0.5 * c * (qa + qb), c / 6.0 * (qa * (2.0 * a + b) + qb * (a + 2.0 * b))


def internal_forces(model: BeamModel, reacts: list[Reaction], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(V, M, N) at x by summing everything strictly to the LEFT of each x (incl. reactions):
    V = sum Fy_left, M = sum Fy_left (x - a) - sum C_left (C CCW +), N = -sum Fx_left.

    Where x is repeated (x[i] == x[i-1], see sample_points) the second copy is the value just to the RIGHT,
    i.e. it also includes concentrated actions sitting exactly at that x. Distributed loads are integrated
    exactly (V quadratic, M cubic for a linearly varying load)."""
    x = np.asarray(x, dtype=float)
    right = np.zeros(x.shape, dtype=bool)
    right[1:] = x[1:] == x[:-1]
    tol = _tol(model)
    V, M, N = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

    def concentrated(a, fx, fy, c):
        on = (x > a + tol) | ((np.abs(x - a) <= tol) & right)
        V[on] += fy
        M[on] += fy * (x[on] - a) - c
        N[on] -= fx

    for r in reacts:
        concentrated(r.x, r.Fx, r.Fy, r.M)
    for ld in model.loads:
        if isinstance(ld, PointLoad):
            concentrated(ld.x, ld.fx, ld.fy, 0.0)
        elif isinstance(ld, MomentLoad):
            concentrated(ld.x, 0.0, 0.0, ld.M)
        elif isinstance(ld, DistributedLoad):
            q1, q2 = ld.sign * ld.w1, ld.sign * ld.w2
            xe = np.clip(x, ld.x1, ld.x2)  # loaded part left of the cut is [x1, xe]
            qe = q1 + (q2 - q1) * (xe - ld.x1) / (ld.x2 - ld.x1)
            F, Fm = _linear_load_integrals(ld.x1, xe, q1, qe)
            V += F
            M += x * F - Fm  # int q(s) (x - s) ds
    return V, M, N


def sample_points(model: BeamModel, n_points: int = 501) -> np.ndarray:
    """Uniform grid plus every support/load key point, with key points of concentrated actions duplicated."""
    L, tol = model.L, _tol(model)
    clip = lambda v: min(max(float(v), 0.0), L)  # noqa: E731  (positions within POS_TOL of the ends)
    conc = {clip(sp.x) for sp in model.supports}
    conc |= {clip(ld.x) for ld in model.loads if isinstance(ld, (PointLoad, MomentLoad))}
    keys = conc | {0.0, L} | {clip(v) for ld in model.distributed_loads for v in (ld.x1, ld.x2)}
    keys = np.array(sorted(keys))
    grid = np.linspace(0.0, L, max(int(n_points), 2))
    grid = grid[np.min(np.abs(grid[:, None] - keys[None, :]), axis=1) > tol]
    dup = np.array([k for k in keys if k in conc])
    return np.sort(np.concatenate([grid, keys, dup]))


def slope_deflection(model: BeamModel, x: np.ndarray, M: np.ndarray, V: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """theta, v from EI v'' = M by cumulative numerical integration, constants from the support BCs
    (fixed: v = theta = 0; pin/roller: v = 0). Determinate transverse cases only.

    Integration is the cumulative trapezoid rule. When V (= dM/dx) is given it adds the Euler-Maclaurin end
    correction -h^2/12 (f'(b) - f'(a)) on every interval, which makes the theta integral exact for the cubic
    M of linearly varying loads and the v integral 4th-order accurate. Zero-length intervals (repeated x at
    concentrated actions) contribute nothing, so theta and v stay continuous across jumps in V and M."""
    x = np.asarray(x, dtype=float)
    M = np.asarray(M, dtype=float)
    EI = model.material.E * sections.second_moment(model.section)
    kappa = M / EI
    h = np.diff(x)
    d_theta = 0.5 * h * (kappa[:-1] + kappa[1:])
    if V is not None:
        V = np.asarray(V, dtype=float)
        d_theta -= h**2 / 12.0 * (V[1:] - V[:-1]) / EI
    theta_p = np.concatenate([[0.0], np.cumsum(d_theta)])  # particular solution, theta_p(0) = 0
    d_v = 0.5 * h * (theta_p[:-1] + theta_p[1:])
    if V is not None:
        d_v -= h**2 / 12.0 * (kappa[1:] - kappa[:-1])
    v_p = np.concatenate([[0.0], np.cumsum(d_v)])

    # Homogeneous part: theta = theta_p + C1, v = v_p + C1 x + C2. One equation per restrained transverse DOF.
    rows, rhs = [], []
    for sp in model.supports:
        if "v" in sp.restrained:
            rows.append([sp.x, 1.0])
            rhs.append(-np.interp(sp.x, x, v_p))
        if "theta" in sp.restrained:
            rows.append([1.0, 0.0])
            rhs.append(-np.interp(sp.x, x, theta_p))
    if len(rows) != 2:
        raise ValueError("slope_deflection needs exactly two transverse restraints (statically determinate beam).")
    C1, C2 = np.linalg.solve(np.array(rows), np.array(rhs))
    return theta_p + C1, v_p + C1 * x + C2


def fibre_stresses(model: BeamModel, M, N=0.0) -> tuple[np.ndarray, np.ndarray]:
    """(sigma_top, sigma_bottom) = N/A - M y/I at y_top (>0) and y_bottom (<0). Sagging M -> bottom in tension."""
    s = model.section
    A, I = sections.area(s), sections.second_moment(s)
    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    return N / A - M * sections.y_top(s) / I, N / A - M * sections.y_bottom(s) / I


def stress_distribution(model: BeamModel, M: float, N: float = 0.0, n: int = 51) -> tuple[np.ndarray, np.ndarray]:
    """Linear normal-stress profile across the depth: returns (y, sigma(y)), y from bottom to top fibre."""
    s = model.section
    y = np.linspace(sections.y_bottom(s), sections.y_top(s), n)
    return y, N / sections.area(s) - M * y / sections.second_moment(s)


def equilibrium_residual(model: BeamModel, reacts: list[Reaction]) -> dict[str, float]:
    """{'sum_Fx', 'sum_Fy', 'sum_M0'} of all applied loads + reactions (moments about x = 0, CCW +)."""
    Fx, Fy, M0 = _applied_resultants(model)
    return {"sum_Fx": Fx + sum(r.Fx for r in reacts),
            "sum_Fy": Fy + sum(r.Fy for r in reacts),
            "sum_M0": M0 + sum(r.x * r.Fy + r.M for r in reacts)}


def differential_residuals(model: BeamModel, x: np.ndarray, V: np.ndarray, M: np.ndarray) -> dict[str, float]:
    """Check dM/dx = V and dV/dx = q (= -w) on every non-zero grid interval.

    Over an interval, (M_b - M_a)/h is the mean of V and (V_b - V_a)/h the mean of q. They are compared with
    the trapezoid mean of V and the mid-point q (exact for linear q). Residuals are relative to max|V| and
    max(|q|, max|V|/L); a few 1e-6 means the diagrams satisfy the differential relations."""
    x, V, M = (np.asarray(a, dtype=float) for a in (x, V, M))
    h = np.diff(x)
    ok = h > 0
    h, xm = h[ok], 0.5 * (x[:-1] + x[1:])[ok]
    q = sum((ld.q(xm) for ld in model.distributed_loads), np.zeros_like(xm))
    dM = (M[1:] - M[:-1])[ok] / h - 0.5 * (V[:-1] + V[1:])[ok]
    dV = (V[1:] - V[:-1])[ok] / h - q
    v_scale = np.max(np.abs(V), initial=0.0)
    q_scale = max(np.max(np.abs(q), initial=0.0), v_scale / model.L)
    rel = lambda r, s: float(np.max(np.abs(r), initial=0.0) / s) if s > 0 else 0.0  # noqa: E731
    return {"dM_dx_minus_V": rel(dM, v_scale), "dV_dx_minus_q": rel(dV, q_scale)}


def textbook_check(model: BeamModel, result: AnalysisResult | None = None) -> TextbookCheck | None:
    """Closed-form values ONLY for the exact configurations they are valid for (else None):
    cantilever + single vertical end load (theta = PL^2/2EI, delta = PL^3/3EI); cantilever + full-length UDL
    (theta = wL^3/6EI, delta = wL^4/8EI); simply supported + central vertical point load (delta = PL^3/48EI);
    simply supported + full-length UDL (delta = 5wL^4/384EI). With `result`, fills numerical + rel_error."""
    if validate(model) or len(model.loads) != 1:
        return None
    kind, L, ld, tol = classify(model), model.L, model.loads[0], _tol(model)
    EI = model.material.E * sections.second_moment(model.section)
    Z = sections.section_modulus(model.section)
    at = lambda a, b: abs(a - b) <= tol  # noqa: E731
    full_udl = (isinstance(ld, DistributedLoad) and at(ld.x1, 0.0) and at(ld.x2, L)
                and abs(ld.w1 - ld.w2) <= 1e-9 * max(abs(ld.w1), abs(ld.w2)) and ld.w1 != 0)
    point = isinstance(ld, PointLoad) and ld.is_transverse and ld.fy != 0

    if kind == "cantilever" and point and at(ld.x, L - model.supports[0].x):  # load at the free end
        P = abs(ld.fy)
        case, f = "cantilever-end-load", {
            "R": ("P", P), "V_max": ("P", P), "M_max": ("PL", P * L),
            "theta_max": ("PL^2/(2EI)", P * L**2 / (2 * EI)), "delta_max": ("PL^3/(3EI)", P * L**3 / (3 * EI))}
    elif kind == "cantilever" and full_udl:
        w = abs(ld.w1)
        case, f = "cantilever-udl", {
            "R": ("wL", w * L), "V_max": ("wL", w * L), "M_max": ("wL^2/2", w * L**2 / 2),
            "theta_max": ("wL^3/(6EI)", w * L**3 / (6 * EI)), "delta_max": ("wL^4/(8EI)", w * L**4 / (8 * EI))}
    elif kind == "simply-supported" and point and at(ld.x, L / 2):
        P = abs(ld.fy)
        case, f = "ss-central-load", {
            "R": ("P/2", P / 2), "V_max": ("P/2", P / 2), "M_max": ("PL/4", P * L / 4),
            "theta_max": ("PL^2/(16EI)", P * L**2 / (16 * EI)), "delta_max": ("PL^3/(48EI)", P * L**3 / (48 * EI))}
    elif kind == "simply-supported" and full_udl:
        w = abs(ld.w1)
        case, f = "ss-udl", {
            "R": ("wL/2", w * L / 2), "V_max": ("wL/2", w * L / 2), "M_max": ("wL^2/8", w * L**2 / 8),
            "theta_max": ("wL^3/(24EI)", w * L**3 / (24 * EI)),
            "delta_max": ("5wL^4/(384EI)", 5 * w * L**4 / (384 * EI))}
    else:
        return None
    f["sigma_max"] = ("M_max/Z", f["M_max"][1] / Z)

    tb = TextbookCheck(case, {k: s for k, (s, _) in f.items()}, {k: float(v) for k, (_, v) in f.items()})
    if result is not None:
        tb.numerical = {
            "R": max(abs(r.Fy) for r in result.reactions), "V_max": abs(result.V_max.value),
            "M_max": abs(result.M_max.value), "theta_max": abs(result.theta_max.value),
            "delta_max": abs(result.v_max.value), "sigma_max": abs(result.sigma_max.value)}
        tb.rel_error = {k: abs(tb.numerical[k] - v) / abs(v) for k, v in tb.textbook.items()}
    return tb


def _analyze_determinate(model: BeamModel, n_points: int, stab: Stability) -> AnalysisResult:
    """Analytical path: reactions -> V, M -> stresses/strains -> slope/deflection -> textbook check."""
    reacts = reactions(model)
    x = sample_points(model, n_points)
    V, M, N = internal_forces(model, reacts, x)
    theta, v = slope_deflection(model, x, M, V)
    s_top, s_bot = fibre_stresses(model, M, N)
    warnings = []
    ratio = model.L / model.section.depth
    if ratio < 10:
        warnings.append(f"L/depth = {ratio:.1f} < 10: Euler-Bernoulli theory neglects shear deformation, so "
                        f"deflections are underestimated; compare with Timoshenko on the Beam Theories page.")
    result = _build_result(model, "analytical", x, V, M, N, theta, v, s_top, s_bot, reacts, stab, warnings)
    result.equilibrium.update(differential_residuals(model, x, V, M))
    result.textbook = textbook_check(model, result)
    return result


def _build_result(model, method, x, V, M, N, theta, v, s_top, s_bot, reacts, stab, warnings, fem=None):
    """Assemble an AnalysisResult (strains, extremes, equilibrium residuals) from sampled fields."""
    E = model.material.E
    both = np.concatenate([s_top, s_bot])
    xx = np.concatenate([x, x])
    where = ["top"] * len(x) + ["bottom"] * len(x)
    i_abs, i_hi, i_lo = int(np.argmax(np.abs(both))), int(np.argmax(both)), int(np.argmin(both))
    tension = Extreme(float(both[i_hi]), float(xx[i_hi]), where[i_hi]) if both[i_hi] > 0 else Extreme(0.0, 0.0)
    compression = Extreme(float(both[i_lo]), float(xx[i_lo]), where[i_lo]) if both[i_lo] < 0 else Extreme(0.0, 0.0)
    return AnalysisResult(
        model=model, method=method, x=x, V=V, M=M, N=N, theta=theta, v=v,
        sigma_top=s_top, sigma_bottom=s_bot, eps_top=s_top / E, eps_bottom=s_bot / E, reactions=reacts,
        V_max=Extreme.of(x, V), M_max=Extreme.of(x, M), N_max=Extreme.of(x, N),
        theta_max=Extreme.of(x, theta), v_max=Extreme.of(x, v),
        sigma_max=Extreme(float(both[i_abs]), float(xx[i_abs]), where[i_abs]),
        sigma_tension_max=tension, sigma_compression_max=compression, stability=stab,
        equilibrium=equilibrium_residual(model, reacts), textbook=None, warnings=list(warnings), fem=fem)


def _tol(model: BeamModel) -> float:
    return max(POS_TOL, 1e-9 * model.L)


def _from_fem(model: BeamModel, fem_result, stab: Stability, warnings: list[str]) -> AnalysisResult:
    """Map a fem.FEMResult onto AnalysisResult (method 'fem-euler' / 'fem-frame', textbook=None)."""
    f = fem_result
    method = "fem-frame" if f.element == "frame" else "fem-euler"
    arr = lambda a: np.asarray(a, dtype=float)  # noqa: E731
    return _build_result(model, method, arr(f.x), arr(f.V), arr(f.M), arr(f.N), arr(f.theta), arr(f.v),
                         arr(f.sigma_top), arr(f.sigma_bottom), list(f.reactions), stab,
                         list(warnings) + list(f.warnings), fem=f)


def analyze(model: BeamModel, n_points: int = 501, n_elements: int = 40) -> AnalysisResult:
    """Single entry point for every page. See module docstring for the routing rules."""
    errors = validate(model)
    if errors:
        raise ModelError("; ".join(errors))
    stab = check_stability(model)
    if not stab.stable:
        raise UnstableStructureError(stab.message)
    transverse = is_transverse_only(model)
    if transverse and stab.determinate:
        return _analyze_determinate(model, n_points, stab)

    from smartbeam import fem  # local import: fem must not depend on this module

    warnings = []
    if not transverse:
        warnings.append("Axial/inclined load present: solved with planar frame elements (u, v, θ per node); "
                        "transverse beam formulas are not applicable.")
    if not stab.determinate:
        warnings.append(stab.message)
    fem_result = fem.solve(model, n_elements=n_elements, theory="euler", frame=not transverse)
    return _from_fem(model, fem_result, stab, warnings)
