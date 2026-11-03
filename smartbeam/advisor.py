"""Rule-based design advisor (no chatbot). OWNER: advisor engineer.
Each recommendation carries a genuinely re-analysed modified design.

Method: utilisation u = demand / capacity per criterion (deflection delta/delta_allow, strength FOS_min/FOS,
buckling FOS_min P/Pcr). Each fix is first sized analytically from the scaling laws of linear beam theory
(delta ~ 1/I, sigma ~ 1/Z, Pcr ~ I_min), e.g. h_new = h (delta/delta_allow)^(1/3) for a rectangle, then the
modified design is re-analysed with the real engine; if a combined effect (axial + bending) leaves a criterion
unmet, the scaling step is repeated from the re-analysed values. Dimensions are rounded UP to 0.1 mm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from smartbeam import materials, safety, sections
from smartbeam.analytical import analyze
from smartbeam.metrics import ConstraintCheck, DesignMetrics, check, evaluate
from smartbeam.model import BeamModel, Section

DEFAULT_SPAN_RATIO = 250  # delta_allow = L / 250 when not given (common serviceability guideline; editable)
EXCESS_UTILISATION = 0.5  # every criterion below this -> 'material excess' (FOS > 2x target, delta < half allowed)
EXCESS_TARGET = 0.9  # downsizing aims at 90 % utilisation of the governing criterion (10 % reserve)
TUBE_RATIO = 0.8  # d/D of the hollow-tube alternative (wall t = 0.1 D)
RECT_ASPECT = 2.0  # h/b of the rectangular alternative
ROUND = 1e-4  # m, dimensions rounded up to 0.1 mm
MAX_RESIZE_ITER = 8


@dataclass
class Recommendation:
    issue: str  # what is wrong
    quantity: str  # 'delta_max' | 'sigma_max' | 'buckling' | 'mass'
    current: float
    limit: float
    action: str  # recommended modification, human readable
    modified_model: BeamModel
    before: DesignMetrics
    after: DesignMetrics
    checks_after: list[ConstraintCheck]
    satisfies_all: bool


@dataclass
class AdvisorReport:
    limiting: str  # 'deflection' | 'stress' | 'buckling' | 'material excess' | 'none'
    utilisation: dict[str, float]  # demand / capacity per criterion (>1 = violated)
    checks: list[ConstraintCheck]
    recommendations: list[Recommendation]


# --------------------------------------------------------------------------- evaluation
@dataclass
class _State:
    model: BeamModel
    result: object  # AnalysisResult
    metrics: DesignMetrics
    buckling: safety.BucklingResult
    util: dict[str, float]  # keys 'deflection', 'stress', ('buckling')
    checks: list[ConstraintCheck]


def _state(model: BeamModel, delta_allow: float, min_fos: float) -> _State:
    r = analyze(model)
    met = evaluate(model, r)
    bk = safety.buckling(model, P_applied=safety.compressive_force(r), n_points=51)
    util = {"deflection": met.delta_max / delta_allow,
            "stress": min_fos * met.sigma_max / model.material.yield_strength}
    checks = [check("Max deflection |v|max (m)", met.delta_max, delta_allow, "<="),
              check("Yield FOS", met.fos, min_fos, ">=")]
    if bk.applicable:
        util["buckling"] = min_fos * bk.P_applied / bk.Pcr
        checks.append(check("Buckling FOS (Euler)", bk.fos, min_fos, ">="))
    return _State(model, r, met, bk, util, checks)


# --------------------------------------------------------------------------- resizing
def _ceil(v: float) -> float:
    return math.ceil(v / ROUND - 1e-9) * ROUND


def _exponents(s: Section, mode: str) -> dict[str, float]:
    """p in u_new = u / s^p when dimension(s) are scaled by s (pure bending / Euler scaling laws)."""
    if mode == "uniform":  # every dimension x s: I ~ s^4, Z ~ s^3
        return {"deflection": 4, "stress": 3, "buckling": 4}
    if mode == "depth":  # rect h x s: I ~ s^3, Z ~ s^2, I_min ~ s^3 if h is the minor dimension else s
        return {"deflection": 3, "stress": 2, "buckling": 3 if s.h <= s.b else 1}
    # 'width', rect b x s: I ~ s, Z ~ s, I_min ~ s^3 if b is the minor dimension else s
    return {"deflection": 1, "stress": 1, "buckling": 3 if s.b <= s.h else 1}


def _scaled(model: BeamModel, mode: str, f: float) -> BeamModel:
    m = model.copy()
    s = m.section
    if mode == "depth":
        s.h = _ceil(s.h * f)
    elif mode == "width":
        s.b = _ceil(s.b * f)
    else:
        for k, v in s.dims().items():
            setattr(s, k, _ceil(v * f))
        if s.shape == "hollow_circ" and s.d >= s.D:  # keep a valid wall after rounding
            s.d = s.D - ROUND
    return m


def _resize(model: BeamModel, mode: str, delta_allow: float, min_fos: float, target: float = 1.0) -> _State:
    """Scale the section until every criterion has u <= target, the governing one close to target."""
    st = _state(model, delta_allow, min_fos)
    for _ in range(MAX_RESIZE_ITER):
        p = _exponents(st.model.section, mode)
        f = max((u / target) ** (1.0 / p[c]) for c, u in st.util.items() if u > 0)
        ok = all(u <= target * (1 + 1e-9) for u in st.util.values())
        if ok and f > 0.999:
            break
        st = _state(_scaled(st.model, mode, f), delta_allow, min_fos)
    return st


def _fmt_dims(s: Section) -> str:
    return ", ".join(f"{k} = {v * 1e3:.4g} mm" for k, v in s.dims().items())


def _rec(before: _State, after: _State, issue: str, quantity: str, current: float, limit: float,
         action: str) -> Recommendation:
    return Recommendation(issue, quantity, current, limit, action, after.model, before.metrics, after.metrics,
                          after.checks, all(c.satisfied for c in after.checks))


# --------------------------------------------------------------------------- recommendation builders
def _section_rec(st: _State, crit: str, issue: str, quantity: str, current: float, limit: float, da: float,
                 fos: float, target: float = 1.0, mode: str | None = None) -> Recommendation:
    """Resize the current section (rect: the most effective dimension for `crit`; circles: uniformly)."""
    s = st.model.section
    if mode is None:
        mode = "uniform" if s.shape != "rect" else ("width" if crit == "buckling" and s.b <= s.h else "depth")
    p = _exponents(s, mode)
    # analytical sizing step: every criterion must reach u <= target, u_new = u / f^p
    f, gov = max(((st.util[c] / target) ** (1.0 / p[c]), c) for c in st.util if st.util[c] > 0)
    new = _resize(st.model, mode, da, fos, target)
    name = {"depth": "depth h", "width": "width b", "uniform": "all dimensions (same proportions)"}[mode]
    law = {"deflection": "δ ∝ 1/I", "stress": "σ ∝ 1/Z", "buckling": "Pcr ∝ I_min"}[gov]
    action = (f"{'Increase' if f > 1 else 'Reduce'} {name}: governed by {gov}, {law} gives a scale factor "
              f"({st.util[gov]:.3g}/{target:g})^(1/{p[gov]}) = {f:.4g}; after re-analysis: "
              f"{_fmt_dims(new.model.section)} (was {_fmt_dims(s)}).")
    return _rec(st, new, issue, quantity, current, limit, action)


def _material_rec(st: _State, crit: str, issue: str, quantity: str, current: float, limit: float, da: float,
                  fos: float) -> Recommendation | None:
    """Lightest library material that makes the SAME section satisfy every criterion (re-analysed)."""
    m0 = st.model.material
    if crit == "stress":
        need, prop, label = m0.yield_strength * st.util["stress"], "yield_strength", "yield strength"
    else:  # deflection and Euler buckling both scale with 1/E
        need, prop, label = m0.E * st.util[crit], "E", "Young's modulus"
    best = None
    for key in materials.material_keys():
        mat = materials.get_material(key)
        if getattr(mat, prop) < need or mat.name == m0.name:
            continue
        m = st.model.copy()
        m.material = mat
        new = _state(m, da, fos)
        if all(c.satisfied for c in new.checks) and (best is None or mat.density < best.model.material.density):
            best = new
    if best is None:
        return None
    unit = 1e9 if prop == "E" else 1e6
    u = "GPa" if prop == "E" else "MPa"
    action = (f"Change material to {best.model.material.name}: required {label} >= {need / unit:.4g} {u} "
              f"(current {getattr(m0, prop) / unit:.4g} {u}, new {getattr(best.model.material, prop) / unit:.4g} {u}); "
              f"lightest library material that passes every check with the same section. Values are approximate.")
    return _rec(st, best, issue, quantity, current, limit, action)


def _alternative_rec(st: _State, issue: str, quantity: str, current: float, limit: float, da: float,
                     fos: float) -> Recommendation:
    """Different section family sized for every criterion: rect/solid -> tube (d/D = 0.8), tube -> rect (h = 2b)."""
    s = st.model.section
    I = sections.second_moment(s)
    m = st.model.copy()
    if s.shape == "hollow_circ":  # equal-I rectangle as the starting guess: b (2b)^3 / 12 = I
        b = (12 * I / RECT_ASPECT**3) ** 0.25
        m.section = Section.rect(_ceil(b), _ceil(RECT_ASPECT * b))
        desc = f"rectangular section h = {RECT_ASPECT:g} b"
    else:  # equal-I tube: pi D^4 (1 - r^4) / 64 = I
        D = (64 * I / (math.pi * (1 - TUBE_RATIO**4))) ** 0.25
        m.section = Section.hollow_circ(_ceil(D), _ceil(TUBE_RATIO * D))
        desc = f"hollow circular tube d/D = {TUBE_RATIO:g}"
    new = _resize(m, "uniform", da, fos)
    action = (f"Alternative section: {desc}, sized for all criteria -> {_fmt_dims(new.model.section)}; "
              f"mass {new.metrics.mass:.4g} kg vs {st.metrics.mass:.4g} kg now.")
    return _rec(st, new, issue, quantity, current, limit, action)


# --------------------------------------------------------------------------- advise
def advise(model: BeamModel, delta_allow: float | None = None, min_fos: float = 1.5) -> AdvisorReport:
    if delta_allow is None:
        delta_allow = model.L / DEFAULT_SPAN_RATIO
    if not delta_allow > 0:
        raise ValueError("Allowable deflection must be positive.")
    if not min_fos >= 1.0:
        raise ValueError("Minimum FOS must be >= 1.")
    st = _state(model, delta_allow, min_fos)
    met, util = st.metrics, st.util
    recs: list[Recommendation] = []

    violated = sorted((c for c, u in util.items() if u > 1.0), key=lambda c: -util[c])
    for crit in violated:
        if crit == "deflection":
            issue = (f"Too flexible: max deflection {met.delta_max * 1e3:.4g} mm exceeds the allowed "
                     f"{delta_allow * 1e3:.4g} mm ({util[crit]:.2f}x).")
        elif crit == "stress":
            issue = (f"Insufficient strength: yield FOS {met.fos:.3g} < required {min_fos:g} "
                     f"(sigma_max = {met.sigma_max / 1e6:.4g} MPa, "
                     f"allowed {model.material.yield_strength / min_fos / 1e6:.4g} MPa).")
        else:
            bk = st.buckling
            issue = (f"Buckling risk: Euler FOS {bk.fos:.3g} < required {min_fos:g} (Pcr = {bk.Pcr / 1e3:.4g} kN, "
                     f"compression {bk.P_applied / 1e3:.4g} kN, {bk.end_condition}).")
        args = _args_of(crit, st, delta_allow, min_fos)
        sec = _section_rec(st, crit, issue, *args, delta_allow, min_fos)
        if not any(r.modified_model.section == sec.modified_model.section for r in recs):  # same fix already listed
            recs.append(sec)
        mat = _material_rec(st, crit, issue, *args, delta_allow, min_fos)
        if mat:
            recs.append(mat)
    if violated:
        recs.append(_alternative_rec(st, "Section family alternative for the violated criteria.",
                                     *_args_of(violated[0], st, delta_allow, min_fos), delta_allow, min_fos))
        limiting = violated[0]
    elif max(util.values()) < EXCESS_UTILISATION:
        limiting = "material excess"
        gov = max(util, key=util.get)
        issue = (f"Over-designed: highest utilisation is {util[gov]:.2f} ({gov}); yield FOS {met.fos:.3g} vs "
                 f"{min_fos:g} required, deflection {met.delta_max * 1e3:.4g} mm vs {delta_allow * 1e3:.4g} mm "
                 f"allowed. Material can be removed.")
        recs.append(_section_rec(st, gov, issue, "mass", met.mass, math.nan, delta_allow, min_fos,
                                 target=EXCESS_TARGET, mode="uniform"))
        recs[-1].limit = recs[-1].after.mass  # mass achievable while every criterion still passes
    else:
        limiting = "none"
    return AdvisorReport(limiting, util, st.checks, recs)


def _args_of(crit: str, st: _State, da: float, fos: float) -> tuple[str, float, float]:
    if crit == "deflection":
        return "delta_max", st.metrics.delta_max, da
    if crit == "stress":
        return "sigma_max", st.metrics.sigma_max, st.model.material.yield_strength / fos
    return "buckling", st.buckling.P_applied, st.buckling.Pcr / fos
