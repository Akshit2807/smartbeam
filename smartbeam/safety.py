"""Failure prediction & safety. OWNER: safety engineer.

Predicts onset of YIELDING only; never claims ultimate fracture. Fatigue results only from user-supplied data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from smartbeam import sections
from smartbeam.model import POS_TOL, BeamModel, Extreme, Section, is_transverse_only

STATUS_SAFE, STATUS_WARNING, STATUS_YIELD = "SAFE", "WARNING", "YIELD PREDICTED"
STATUS_BUCKLING = "BUCKLING PREDICTED"  # Euler load exceeded: an instability, deliberately not called "yield"
STATUS_RANK = {STATUS_SAFE: 0, STATUS_WARNING: 1, STATUS_YIELD: 2, STATUS_BUCKLING: 2}
# Effective-length factors K (L_eff = K L), ideal end conditions.
BUCKLING_K = {"fixed-free": 2.0, "pinned-pinned": 1.0, "fixed-pinned": 0.699, "fixed-fixed": 0.5}
FIXED_PINNED_KL = 4.493409457909064  # first root of tan(kL) = kL; K = pi / kL = 0.699

DISCLAIMER = ("These checks predict the ONSET OF YIELDING (first fibre reaching the yield strength) and ideal "
              "elastic (Euler) buckling. Yield is not fracture: the ultimate/fracture load of a ductile member is "
              "higher and is not predicted here. Material values are approximate reference data; no stress "
              "concentrations, residual stresses, imperfections or dynamic effects are modelled.")


@dataclass
class YieldResult:
    sigma_max: Extreme  # governing |sigma| (Pa) and location
    sigma_y: float  # Pa
    fos: float  # sigma_y / |sigma_max| (inf if unstressed)
    first_yield_load_factor: float  # lambda: all loads x lambda -> first fibre yields (= fos, linear elastic)
    status: str  # STATUS_*
    min_fos: float
    message: str


@dataclass
class CombinedStressResult:
    x: float  # critical section, m
    y: float  # critical fibre (from centroid, + up), m
    sigma: float  # normal stress N/A - M y/I at (x, y), Pa
    tau: float  # V Q(y) / (I b(y)) at (x, y), Pa
    von_mises: float  # sqrt(sigma^2 + 3 tau^2), Pa
    fos: float  # sigma_y / von_mises
    y_profile: np.ndarray  # y samples at critical x
    vm_profile: np.ndarray  # von Mises along y_profile
    assumptions: list[str]


@dataclass
class BucklingResult:
    applicable: bool  # False when no compressive axial force or end condition not identifiable
    end_condition: str | None  # key of BUCKLING_K
    K: float | None
    L_eff: float | None  # m
    Pcr: float | None  # pi^2 E I_min / L_eff^2, N
    P_applied: float  # compressive axial force, N (>= 0)
    fos: float | None  # Pcr / P_applied
    slenderness: float | None  # L_eff / r_g
    slenderness_limit: float | None  # sqrt(2 pi^2 E / sigma_y) (Johnson transition); below -> Euler unconservative
    mode_x: np.ndarray  # m
    mode_shape: np.ndarray  # normalised to max |.| = 1
    warnings: list[str] = field(default_factory=list)


@dataclass
class FatigueResult:
    sigma_a: float  # (max - min)/2, Pa
    sigma_m: float  # (max + min)/2, Pa
    sigma_ar: float | None  # Goodman equivalent fully-reversed amplitude sigma_a / (1 - sigma_m/UTS), Pa
    cycles_to_failure: float | None  # Basquin N = 0.5 (sigma_ar / sigma_f')^(1/b); None without user data
    goodman_fos: float | None  # requires user endurance limit: 1 / (sigma_a/Se + sigma_m/UTS)
    infinite_life: bool | None
    sn_N: np.ndarray  # S-N curve cycles (empty without data)
    sn_S: np.ndarray  # S-N curve amplitude, Pa
    assumptions: list[str]
    warnings: list[str] = field(default_factory=list)
    sigma_ar_gerber: float | None = None  # Gerber equivalent amplitude sigma_a / (1 - (sigma_m/UTS)^2), Pa


@dataclass
class SafetyReport:
    yield_result: YieldResult
    combined: CombinedStressResult
    buckling: BucklingResult
    status: str  # worst of yield + buckling, STATUS_*
    messages: list[str]


def status_for(fos: float, min_fos: float) -> str:
    """SAFE if fos >= min_fos; WARNING if 1 <= fos < min_fos; YIELD PREDICTED if fos < 1."""
    if fos < 1.0:
        return STATUS_YIELD
    return STATUS_SAFE if fos >= min_fos else STATUS_WARNING


def _worst(*statuses: str) -> str:
    return max(statuses, key=lambda s: STATUS_RANK[s])


# --------------------------------------------------------------------------- yield
def yield_check(result, min_fos: float = 1.5) -> YieldResult:
    """result: analytical.AnalysisResult. status: fos < 1 -> YIELD PREDICTED; fos < min_fos -> WARNING; else SAFE."""
    if not min_fos >= 1.0:
        raise ValueError("Minimum FOS must be >= 1.")
    sy = result.model.material.yield_strength
    smax = result.sigma_max
    s = abs(smax.value)
    fos = sy / s if s > 0 else math.inf
    status = status_for(fos, min_fos)
    if math.isinf(fos):
        msg = "The beam is unstressed; no yielding."
    else:
        msg = (f"Largest normal stress |σ| = {s / 1e6:.4g} MPa at x = {smax.x:.4g} m ({smax.where or 'fibre'}); "
               f"yield strength {sy / 1e6:.4g} MPa -> FOS = {fos:.3g} (required >= {min_fos:g}). "
               f"Linear elastic: scaling ALL loads by {fos:.3g} makes the first fibre reach yield.")
        if status == STATUS_YIELD:
            msg += " The applied loads already exceed first yield (plastic behaviour beyond this elastic model)."
    return YieldResult(smax, sy, fos, fos, status, min_fos, msg)


# --------------------------------------------------------------------------- combined stress / von Mises
VM_ASSUMPTIONS = [
    "Normal stress σ = N/A − M·y/I (axial + bending, Euler-Bernoulli: plane sections remain plane).",
    "Shear stress τ = V·Q(y) / (I·b(y)) from elementary beam theory; for circular sections this is the "
    "average across the chord width b(y).",
    "Plane stress at each point (σx, τxy only), von Mises σvm = √(σ² + 3τ²), "
    "compared with the uniaxial yield strength.",
    "No stress concentrations (holes, notches, load/support bearing stresses) and no residual stresses.",
]


def von_mises_check(result, n_y: int = 41) -> CombinedStressResult:
    """Scan x and y for max sqrt(sigma^2 + 3 tau^2), sigma = N/A - M y/I, tau = V Q/(I b). States assumptions
    (plane stress at a point, no stress concentrations, shear from elementary beam theory)."""
    s = result.model.section
    A, I = sections.area(s), sections.second_moment(s)
    y = np.linspace(sections.y_bottom(s), sections.y_top(s), n_y)
    Q = np.asarray(sections.first_moment(s, y))
    b = np.asarray(sections.width_at(s, y))
    q_over_ib = np.divide(Q, I * b, out=np.zeros_like(b), where=b > 0)

    V, M, N = (np.asarray(a, dtype=float)[:, None] for a in (result.V, result.M, result.N))
    sigma = N / A - M * y[None, :] / I
    tau = V * q_over_ib[None, :]
    vm = np.sqrt(sigma**2 + 3.0 * tau**2)
    i, j = np.unravel_index(int(np.argmax(vm)), vm.shape)
    vmax = float(vm[i, j])
    sy = result.model.material.yield_strength
    return CombinedStressResult(
        x=float(result.x[i]), y=float(y[j]), sigma=float(sigma[i, j]), tau=float(tau[i, j]), von_mises=vmax,
        fos=sy / vmax if vmax > 0 else math.inf, y_profile=y, vm_profile=vm[i].copy(),
        assumptions=list(VM_ASSUMPTIONS))


# --------------------------------------------------------------------------- buckling
def min_second_moment(s: Section) -> float:
    """Minimum principal second moment (buckling axis): rect min(b h^3, h b^3)/12, circles = I. m^4"""
    if s.shape == "rect":
        return min(s.b * s.h**3, s.h * s.b**3) / 12.0
    return sections.second_moment(s)


def _at_end(x: float, L: float) -> str | None:
    if abs(x) <= POS_TOL:
        return "left"
    if abs(x - L) <= POS_TOL * max(1.0, L):
        return "right"
    return None


def infer_end_condition(model: BeamModel) -> str | None:
    """Map supports to a BUCKLING_K key (e.g. single fixed -> 'fixed-free', pin+roller -> 'pinned-pinned').
    Only members supported exactly at their ends are identifiable; anything else returns None."""
    sup = model.supports
    if any(_at_end(s.x, model.L) is None for s in sup):
        return None
    if len(sup) == 1:
        return "fixed-free" if sup[0].type == "fixed" else None
    if len(sup) == 2 and {_at_end(s.x, model.L) for s in sup} == {"left", "right"}:
        n_fixed = sum(s.type == "fixed" for s in sup)
        return ("pinned-pinned", "fixed-pinned", "fixed-fixed")[n_fixed]
    return None


def _mode_shape(end_condition: str, xi: np.ndarray, fixed_at_left: bool) -> np.ndarray:
    """First Euler mode, xi = x/L in [0, 1], normalised to max |w| = 1."""
    t = xi if fixed_at_left else 1.0 - xi  # measure from the fixed end where it matters
    if end_condition == "pinned-pinned":
        w = np.sin(np.pi * xi)
    elif end_condition == "fixed-free":
        w = 1.0 - np.cos(np.pi * t / 2.0)
    elif end_condition == "fixed-fixed":
        w = 1.0 - np.cos(2.0 * np.pi * xi)
    else:  # fixed-pinned, fixed at t = 0: w = sin(kx) - kx + kL (1 - cos(kx)), tan(kL) = kL
        k = FIXED_PINNED_KL
        w = np.sin(k * t) - k * t + k * (1.0 - np.cos(k * t))
    return w / np.max(np.abs(w))


def compressive_force(result) -> float:
    """Largest compressive internal axial force max(0, -min N) of an AnalysisResult, N.
    Round-off (e.g. cos(-90 deg) of a vertical load) below 1e-9 of the applied forces counts as zero."""
    model = result.model
    N = np.asarray(result.N, dtype=float)
    if is_transverse_only(model) or not N.size:
        return 0.0
    scale = (sum(abs(p.magnitude) for p in model.point_loads)
             + sum(abs(w.resultant) for w in model.distributed_loads))
    P = float(-N.min())
    return P if P > 1e-9 * scale else 0.0


def buckling(model: BeamModel, end_condition: str | None = None, P_applied: float | None = None,
             n_points: int = 201) -> BucklingResult:
    """Euler buckling. P_applied None -> compressive axial resultant from the loads. Uses the minimum I
    (rect: min(b h^3, h b^3)/12).

    The applied load is the largest compressive internal force N(x) from the engine (exact for a member
    loaded at its ends; conservative when N varies along the member)."""
    warnings: list[str] = []
    if P_applied is None:
        if is_transverse_only(model):
            P_applied = 0.0
        else:
            from smartbeam.analytical import analyze
            P_applied = compressive_force(analyze(model))
    P = float(max(0.0, P_applied))
    ec = end_condition or infer_end_condition(model)
    x = np.linspace(0.0, model.L, n_points)
    if ec is None:
        warnings.append("End condition not identifiable (supports must sit exactly at the member ends: one fixed "
                        "end, or two end supports). Choose an end condition explicitly to evaluate buckling.")
        return BucklingResult(False, None, None, None, None, P, None, None, None, x, np.zeros_like(x), warnings)
    if ec not in BUCKLING_K:
        raise ValueError(f"Unknown end condition {ec!r}; expected one of {list(BUCKLING_K)}.")

    s, m = model.section, model.material
    K = BUCKLING_K[ec]
    L_eff = K * model.L
    I_min, A = min_second_moment(s), sections.area(s)
    Pcr = math.pi**2 * m.E * I_min / L_eff**2
    lam = L_eff / math.sqrt(I_min / A)
    lam_t = math.sqrt(2.0 * math.pi**2 * m.E / m.yield_strength)
    fixed = [sp for sp in model.supports if sp.type == "fixed"]
    fixed_left = not fixed or _at_end(fixed[0].x, model.L) == "left"
    mode = _mode_shape(ec, x / model.L, fixed_left)

    fos = Pcr / P if P > 0 else None
    if P <= 0:
        warnings.append("No compressive axial force: Euler buckling does not govern (Pcr shown for reference).")
    if lam < lam_t:
        sig_j = m.yield_strength - (m.yield_strength * lam / (2 * math.pi)) ** 2 / m.E
        warnings.append(
            f"Stocky member: slenderness {lam:.1f} < transition {lam_t:.1f}. Euler's formula is UNCONSERVATIVE "
            f"here; inelastic (Johnson parabola) critical stress ~ {sig_j / 1e6:.4g} MPa, i.e. "
            f"Pcr ~ {sig_j * A / 1e3:.4g} kN instead of {Pcr / 1e3:.4g} kN.")
    if P > 0 and P / A >= m.yield_strength:
        warnings.append("Direct compressive stress P/A already reaches yield: the member squashes before buckling.")
    warnings.append(f"Ideal Euler column ({ec}, K = {K}): perfectly straight, concentric load, linear elastic; "
                    "real members with imperfections buckle below Pcr. Buckling about the weak axis (I_min) "
                    "assumes the same end restraint in both planes.")
    return BucklingResult(P > 0, ec, K, L_eff, Pcr, P, fos, lam, lam_t, x, mode, warnings)


# --------------------------------------------------------------------------- fatigue
def fatigue(sigma_max: float, sigma_min: float, uts: float | None, sigma_f_prime: float | None = None,
            b: float | None = None, endurance_limit: float | None = None) -> FatigueResult:
    """Stress-life introduction. Cycles/curve only when the user supplies sigma_f' and b (and/or Se)."""
    if sigma_min > sigma_max:
        raise ValueError("sigma_min must not exceed sigma_max.")
    if sigma_f_prime is not None and not sigma_f_prime > 0:
        raise ValueError("Fatigue strength coefficient sigma_f' must be positive.")
    if b is not None and not b < 0:
        raise ValueError("Basquin exponent b must be negative (typically -0.05 to -0.15).")
    if endurance_limit is not None and not endurance_limit > 0:
        raise ValueError("Endurance limit must be positive.")

    sa, sm = (sigma_max - sigma_min) / 2.0, (sigma_max + sigma_min) / 2.0
    assumptions = [
        "Constant-amplitude cycling between sigma_min and sigma_max; sigma_a = (max - min)/2, "
        "sigma_m = (max + min)/2.",
        "Stress-life (high-cycle) approach on smooth specimens: no notch, size, surface or reliability factors.",
    ]
    warnings: list[str] = []
    s_ar = s_ar_g = None
    if uts is None:
        warnings.append("Insufficient data: UTS unknown, so no mean-stress correction (Goodman/Gerber).")
    elif sm >= uts:
        warnings.append("Mean stress reaches the UTS: static failure, fatigue correction not meaningful.")
    elif sm <= 0:
        s_ar = s_ar_g = sa
        assumptions.append("Compressive/zero mean stress: no mean-stress benefit taken (sigma_ar = sigma_a, "
                           "conservative).")
    else:
        s_ar = sa / (1.0 - sm / uts)
        s_ar_g = sa / (1.0 - (sm / uts) ** 2)
        assumptions.append("Mean-stress correction: Goodman sigma_ar = sigma_a / (1 - sigma_m/UTS) (used for life, "
                           "conservative); Gerber sigma_a / (1 - (sigma_m/UTS)^2) shown for comparison.")

    has_basquin = sigma_f_prime is not None and b is not None
    s_eq = s_ar if s_ar is not None else (sa if uts is None else None)
    if uts is None and s_eq is not None:
        assumptions.append("Life computed from the uncorrected amplitude sigma_a (mean stress ignored).")

    infinite = None
    goodman_fos = None
    if endurance_limit is not None:
        if s_eq is not None:
            infinite = s_eq <= endurance_limit
        if uts is not None and sm < uts:
            denom = sa / endurance_limit + max(sm, 0.0) / uts
            goodman_fos = 1.0 / denom if denom > 0 else math.inf
    else:
        warnings.append("Insufficient data: no endurance limit Se supplied, so no Goodman FOS / infinite-life check.")

    cycles = None
    if has_basquin and s_eq is not None:
        assumptions.append("Basquin: sigma_ar = sigma_f' (2N)^b  ->  N = 0.5 (sigma_ar / sigma_f')^(1/b).")
        if s_eq <= 0 or infinite:
            cycles = math.inf
            if infinite:
                assumptions.append("Amplitude at or below the user endurance limit: infinite life assumed.")
        else:
            cycles = 0.5 * (s_eq / sigma_f_prime) ** (1.0 / b)
            if cycles < 1e3:
                warnings.append("Predicted life < 1e3 cycles: low-cycle regime, Basquin (high-cycle) not valid.")
    elif not has_basquin:
        warnings.append("Insufficient data: cycles to failure require user-supplied sigma_f' and b "
                        "(no fatigue properties are assumed for library materials).")

    if has_basquin:
        sn_N = np.logspace(3, 8, 121)
        sn_S = sigma_f_prime * (2.0 * sn_N) ** b
        if endurance_limit is not None:
            sn_S = np.maximum(sn_S, endurance_limit)
    else:
        sn_N = sn_S = np.array([])
    return FatigueResult(sa, sm, s_ar, cycles, goodman_fos, infinite, sn_N, sn_S, assumptions, warnings, s_ar_g)


# --------------------------------------------------------------------------- overall
def assess(model: BeamModel, result, min_fos: float = 1.5) -> SafetyReport:
    """Run yield, von Mises and buckling checks; overall status is the worst of them."""
    yr = yield_check(result, min_fos)
    vm = von_mises_check(result)
    bk = buckling(model, P_applied=compressive_force(result))
    vm_status = status_for(vm.fos, min_fos)
    statuses = [yr.status, vm_status]
    msgs = [yr.message,
            f"von Mises (axial + bending + shear): max {vm.von_mises / 1e6:.4g} MPa at x = {vm.x:.4g} m, "
            f"y = {vm.y * 1e3:.3g} mm -> FOS = {vm.fos:.3g} ({vm_status})."]
    if bk.applicable:
        bk_status = status_for(bk.fos, min_fos)
        if bk_status == STATUS_YIELD:
            bk_status = STATUS_BUCKLING
        statuses.append(bk_status)
        msgs.append(f"Euler buckling ({bk.end_condition}): Pcr = {bk.Pcr / 1e3:.4g} kN vs compressive "
                    f"{bk.P_applied / 1e3:.4g} kN -> FOS = {bk.fos:.3g} ({bk_status}).")
        msgs.extend(w for w in bk.warnings if w.startswith(("Stocky", "Direct")))
    elif bk.P_applied > 0:  # compression present but the end condition is not identifiable
        msgs.append(f"Buckling: compressive force {bk.P_applied / 1e3:.4g} kN present, but {bk.warnings[0]}")
    else:
        msgs.append("Buckling: not applicable (no compressive axial force).")
    msgs.append(DISCLAIMER)
    return SafetyReport(yr, vm, bk, _worst(*statuses), msgs)
