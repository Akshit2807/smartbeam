"""Design optimization with SciPy SLSQP. OWNER: optimization engineer.

Every candidate is evaluated by the real engine; the final design is independently re-verified with
analytical.analyze() and is labelled OPTIMIZED only if every constraint passes.

Design variables are the section dimensions (Section.dims() keys), scaled to millimetres for conditioning.
Constraints (normalised, >= 0):  1 - delta_max/delta_allow,  1 - FOS_min sigma_max/sigma_y,  and, when the member
carries axial compression with identifiable end conditions, 1 - FOS_min P/Pcr (Euler).  hollow_circ also needs
(D - d)/2 >= t_min. SLSQP is started from several points (current design, upper corner, mid-box, seeded random)
and the best independently verified design is kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

from smartbeam import materials, safety, sections
from smartbeam.analytical import analyze
from smartbeam.metrics import ConstraintCheck, DesignMetrics, check, evaluate
from smartbeam.model import BeamModel, ModelError, Section, validate

MM = 1e-3
DEFAULT_BOUND_FACTORS = (0.2, 5.0)  # default bounds = (0.2 x, 5 x) the current dimension when not given
CONSTRAINT_MARGIN = 1e-6  # relative reserve inside SLSQP so re-verification never fails on round-off
N_RANDOM_STARTS = 3


@dataclass
class OptimizationProblem:
    objective: str = "volume"  # 'volume' | 'mass' | 'cost'
    delta_allow: float = 0.005  # m, |v|max <= delta_allow
    fos_min: float = 1.5  # yield FOS >= fos_min
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)  # keys = Section.dims() keys, m
    t_min: float = 1e-3  # hollow_circ minimum wall (D - d)/2, m
    cost_per_kg: float | None = None  # overrides material.cost_per_kg


@dataclass
class OptimizationResult:
    status: str  # 'OPTIMIZED' | 'INFEASIBLE' | 'FAILED'
    success: bool  # SciPy's own flag
    feasible: bool  # independent re-verification passed
    initial_dims: dict[str, float]
    optimal_dims: dict[str, float] | None
    initial: DesignMetrics
    optimized: DesignMetrics | None
    reduction_pct: float | None  # objective reduction vs initial
    checks: list[ConstraintCheck]  # re-verified constraints of the final design
    message: str
    infeasibility_reasons: list[str]
    n_iterations: int
    optimized_model: BeamModel | None


# --------------------------------------------------------------------------- helpers
def with_dims(model: BeamModel, dims: dict[str, float]) -> BeamModel:
    """Copy of `model` with the same section shape and new dimensions (m)."""
    m = model.copy()
    m.section = Section(model.section.shape, **dims)
    return m


def resolve_bounds(model: BeamModel, problem: OptimizationProblem) -> dict[str, tuple[float, float]]:
    """Bounds per dimension (m): user values, else DEFAULT_BOUND_FACTORS x current dimension."""
    lo_f, hi_f = DEFAULT_BOUND_FACTORS
    out = {}
    for k, v in model.section.dims().items():
        lo, hi = problem.bounds.get(k, (lo_f * v, hi_f * v))
        if not (lo > 0 and hi >= lo):
            raise ValueError(f"Bounds for {k} must satisfy 0 < min <= max (got {lo!r}, {hi!r}).")
        out[k] = (float(lo), float(hi))
    return out


def _rate(model: BeamModel, problem: OptimizationProblem) -> float | None:
    return problem.cost_per_kg if problem.cost_per_kg is not None else model.material.cost_per_kg


def objective_value(model: BeamModel, problem: OptimizationProblem, dims: dict[str, float]) -> float:
    """Volume (m^3), mass (kg) or cost (INR) of the beam with these dimensions."""
    vol = sections.area(Section(model.section.shape, **dims)) * model.L
    if problem.objective == "volume":
        return vol
    mass = vol * model.material.density
    return mass if problem.objective == "mass" else mass * _rate(model, problem)


@dataclass
class _Eval:
    delta: float
    sigma: float
    P: float  # compressive axial force, N
    Pcr: float | None  # Euler load when buckling applies


def _evaluate(model: BeamModel) -> _Eval:
    r = analyze(model)
    P = safety.compressive_force(r)
    Pcr = None
    if P > 0:
        bk = safety.buckling(model, P_applied=P, n_points=3)
        Pcr = bk.Pcr if bk.applicable else None
    return _Eval(abs(r.v_max.value), abs(r.sigma_max.value), P, Pcr)


def verify(model: BeamModel, problem: OptimizationProblem,
           bounds: dict[str, tuple[float, float]] | None = None) -> list[ConstraintCheck]:
    """Independent constraint table for a design: fresh analyze() + buckling + bounds."""
    r = analyze(model)
    met = evaluate(model, r, problem.cost_per_kg)
    out = [check("Max deflection |v|max (m)", met.delta_max, problem.delta_allow, "<="),
           check("Yield FOS", met.fos, problem.fos_min, ">=")]
    P = safety.compressive_force(r)
    if P > 0:
        bk = safety.buckling(model, P_applied=P, n_points=3)
        if bk.applicable:
            out.append(check("Buckling FOS (Euler)", bk.fos, problem.fos_min, ">="))
    for k, v in model.section.dims().items():
        if bounds and k in bounds:
            lo, hi = bounds[k]
            out.append(check(f"{k} >= {k}_min (m)", v, lo, ">="))
            out.append(check(f"{k} <= {k}_max (m)", v, hi, "<="))
    if model.section.shape == "hollow_circ":
        out.append(check("Wall thickness (D - d)/2 (m)", (model.section.D - model.section.d) / 2, problem.t_min, ">="))
    return out


def _strongest_dims(shape: str, bounds, t_min: float) -> dict[str, float] | None:
    """Dimensions of maximum I, Z and I_min inside the bounds (None if the bounds admit no valid section)."""
    if shape == "hollow_circ":
        D, d = bounds["D"][1], bounds["d"][0]
        return {"D": D, "d": d} if D - d >= 2 * t_min else None
    return {k: hi for k, (lo, hi) in bounds.items()}


# --------------------------------------------------------------------------- optimize
def optimize(model: BeamModel, problem: OptimizationProblem) -> OptimizationResult:
    if problem.objective not in ("volume", "mass", "cost"):
        raise ValueError("objective must be 'volume', 'mass' or 'cost'.")
    if not problem.delta_allow > 0:
        raise ValueError("Allowable deflection must be positive.")
    if not problem.fos_min >= 1.0:
        raise ValueError("Minimum FOS must be >= 1.")
    if problem.objective == "cost" and _rate(model, problem) is None:
        raise ValueError("Cost objective needs a cost per kg (material has none; set cost_per_kg).")
    errs = validate(model)
    if errs:
        raise ModelError("; ".join(errs))

    shape = model.section.shape
    keys = list(model.section.dims())
    bounds = resolve_bounds(model, problem)
    sy = model.material.yield_strength
    init_dims = model.section.dims()
    initial = evaluate(model, cost_per_kg=problem.cost_per_kg)
    obj0 = objective_value(model, problem, init_dims)

    def dims_of(z):  # z in mm
        return {k: float(v) * MM for k, v in zip(keys, z)}

    cache: dict[tuple, _Eval | None] = {}

    def ev(z) -> _Eval | None:
        key = tuple(np.round(z, 12))
        if key not in cache:
            dims = dims_of(z)
            try:
                cache[key] = _evaluate(with_dims(model, dims))
            except ModelError:  # SLSQP may probe an invalid section (e.g. d >= D) between iterations
                cache[key] = None
        return cache[key]

    def g_struct(z):
        e = ev(z)
        if e is None:
            return np.full(3, -10.0)
        g = [1 - e.delta / problem.delta_allow, 1 - problem.fos_min * e.sigma / sy,
             1 - problem.fos_min * e.P / e.Pcr if e.Pcr else 1.0]
        return np.array(g) - CONSTRAINT_MARGIN

    cons = [{"type": "ineq", "fun": g_struct}]
    if shape == "hollow_circ":
        iD, id_ = keys.index("D"), keys.index("d")
        cons.append({"type": "ineq", "fun": lambda z: np.array([(z[iD] - z[id_]) / 2 - problem.t_min / MM])})

    def f(z):
        return objective_value(model, problem, dims_of(z)) / obj0

    lo = np.array([bounds[k][0] for k in keys]) / MM
    hi = np.array([bounds[k][1] for k in keys]) / MM

    def project(z):
        z = np.clip(np.asarray(z, dtype=float), lo, hi)
        if shape == "hollow_circ":
            z[id_] = max(lo[id_], min(z[id_], z[iD] - 2 * problem.t_min / MM))
        return z

    rng = np.random.default_rng(0)
    starts = [project(np.array([init_dims[k] for k in keys]) / MM), project(hi), project((lo + hi) / 2)]
    starts += [project(lo + rng.random(len(keys)) * (hi - lo)) for _ in range(N_RANDOM_STARTS)]

    best = None  # (objective, model, checks, scipy result)
    scipy_ok = False
    for z0 in starts:
        res = minimize(f, z0, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-10, "eps": 1e-6})
        scipy_ok |= bool(res.success)
        cand = with_dims(model, dims_of(project(res.x)))
        if validate(cand):
            continue
        chk = verify(cand, problem, bounds)  # independent re-verification, not the optimizer's numbers
        if all(c.satisfied for c in chk):
            obj = objective_value(model, problem, cand.section.dims())
            if best is None or obj < best[0]:
                best = (obj, cand, chk, res)

    if best is not None:
        obj, cand, chk, res = best
        optimized = evaluate(cand, cost_per_kg=problem.cost_per_kg)
        red = (obj0 - obj) / obj0 * 100
        msg = (f"OPTIMIZED: {problem.objective} {'reduced' if red >= 0 else 'increased'} by {abs(red):.1f}% "
               f"(SLSQP, {len(starts)} starts; every constraint re-verified with an independent analysis).")
        if red < 0:
            msg += " The initial design violated a constraint, so the minimum feasible design is larger."
        return OptimizationResult("OPTIMIZED", bool(res.success), True, init_dims, cand.section.dims(), initial,
                                  optimized, red, chk, msg, [], int(res.nit), cand)

    # ---- no verified design: explain why
    reasons = _diagnose(model, problem, bounds)
    status = "INFEASIBLE" if reasons else "FAILED"
    if status == "INFEASIBLE":
        msg = "INFEASIBLE: no section within the bounds satisfies every constraint. " + " ".join(reasons)
        strongest = _strongest_dims(shape, bounds, problem.t_min)
        chk = verify(with_dims(model, strongest), problem, bounds) if strongest else []
    else:
        msg = ("FAILED: the strongest section within the bounds is feasible, but SLSQP did not converge to a "
               "verified design from any start point. Try wider bounds or a different initial design.")
        chk = []
    return OptimizationResult(status, scipy_ok, False, init_dims, None, initial, None, None, chk, msg, reasons,
                              0, None)


def _diagnose(model: BeamModel, problem: OptimizationProblem, bounds) -> list[str]:
    """Reasons why the bounds admit no feasible design (empty list if the strongest section is feasible)."""
    shape = model.section.shape
    dims = _strongest_dims(shape, bounds, problem.t_min)
    if dims is None:
        D_hi, d_lo = bounds["D"][1], bounds["d"][0]
        return [f"Bounds admit no valid tube: D_max - d_min = {(D_hi - d_lo) * 1e3:.4g} mm < 2 t_min = "
                f"{2 * problem.t_min * 1e3:.4g} mm."]
    m = with_dims(model, dims)
    e = _evaluate(m)
    s = m.section
    txt = ", ".join(f"{k} = {v * 1e3:.4g} mm" for k, v in dims.items())
    out = []
    if e.delta > problem.delta_allow:
        I = sections.second_moment(s)
        out.append(f"Deflection: even the stiffest section in the bounds ({txt}) deflects {e.delta * 1e3:.4g} mm "
                   f"> allowed {problem.delta_allow * 1e3:.4g} mm. Since δ is proportional to 1/I, the required "
                   f"I is about {I * e.delta / problem.delta_allow * 1e12:.4g} mm⁴, but the bounds allow at most "
                   f"{I * 1e12:.4g} mm^4.")
    sy = model.material.yield_strength
    if problem.fos_min * e.sigma > sy:
        Z = sections.section_modulus(s)
        out.append(f"Strength: at the largest section ({txt}) σmax = {e.sigma / 1e6:.4g} MPa gives FOS "
                   f"{sy / e.sigma:.3g} < {problem.fos_min:g}. Required section modulus about "
                   f"{Z * problem.fos_min * e.sigma / sy * 1e9:.4g} mm³ (bounds allow {Z * 1e9:.4g} mm³).")
    if e.Pcr and problem.fos_min * e.P > e.Pcr:
        out.append(f"Buckling: at the largest section Pcr = {e.Pcr / 1e3:.4g} kN, below FOS {problem.fos_min:g} x "
                   f"compressive {e.P / 1e3:.4g} kN.")
    return out


# --------------------------------------------------------------------------- Pareto
def _total_force(model: BeamModel) -> float:
    return (sum(abs(p.magnitude) for p in model.point_loads)
            + sum(abs(w.resultant) for w in model.distributed_loads))


def _pareto_mask(cost: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """Non-dominated rows: minimise cost columns, maximise benefit columns. NaN rows are never on the front."""
    pts = np.column_stack([cost, -benefit])  # all minimised
    ok = ~np.isnan(pts).any(axis=1)
    mask = np.zeros(len(pts), dtype=bool)
    for i in np.flatnonzero(ok):
        dominated = np.all(pts[ok] <= pts[i], axis=1) & np.any(pts[ok] < pts[i], axis=1)
        mask[i] = not dominated.any()
    return mask


def pareto_front(model: BeamModel, problem: OptimizationProblem, material_keys: list[str] | None = None,
                 n_points: int = 15) -> pd.DataFrame:
    """Trade-off samples. Columns: material, <dims>, mass, volume, cost, stiffness (N/m = total load / delta_max),
    fos, delta_max, sigma_max, feasible, pareto (bool, non-dominated in mass vs stiffness vs fos).

    Each material is sampled at n_points space-filling (Halton) section sizes inside the bounds, each one fully
    analysed. The first Pareto axis is the problem's objective (volume / mass / cost; mass by default)."""
    keys = list(model.section.dims())
    bounds = resolve_bounds(model, problem)
    lo = np.array([bounds[k][0] for k in keys])
    hi = np.array([bounds[k][1] for k in keys])
    if len(keys) == 1:
        u = np.linspace(0, 1, n_points)[:, None]
    else:
        u = qmc.Halton(d=len(keys), scramble=False).random(n_points + 1)[1:]  # skip the all-zero point
    F = _total_force(model)

    rows = []
    mats = [(k, materials.get_material(k)) for k in (material_keys or materials.material_keys())]
    for key, mat in mats:
        for ui in u:
            dims = dict(zip(keys, lo + ui * (hi - lo)))
            if model.section.shape == "hollow_circ":  # sample d inside the admissible wall range for this D
                d_hi = min(bounds["d"][1], dims["D"] - 2 * problem.t_min)
                if d_hi < bounds["d"][0]:
                    continue
                dims["d"] = bounds["d"][0] + ui[keys.index("d")] * (d_hi - bounds["d"][0])
            m = with_dims(model, dims)
            m.material = mat
            met = evaluate(m)  # each material at its own (approximate) cost per kg
            row = {"material": mat.name, "material_key": key, **dims, "mass": met.mass, "volume": met.volume,
                   "cost": np.nan if met.cost is None else met.cost,
                   "stiffness": F / met.delta_max if met.delta_max > 0 and F > 0 else np.nan,
                   "fos": met.fos, "delta_max": met.delta_max, "sigma_max": met.sigma_max,
                   "feasible": met.delta_max <= problem.delta_allow and met.fos >= problem.fos_min}
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    axis = problem.objective
    df["pareto"] = _pareto_mask(df[axis].to_numpy(float), df[["stiffness", "fos"]].to_numpy(float))
    return df
