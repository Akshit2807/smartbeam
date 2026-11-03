"""Shared design metrics + constraint checks used by comparison, optimization, advisor and lab.
Every number comes from analytical.analyze(); nothing is estimated here."""
from __future__ import annotations

import math
from dataclasses import dataclass

from smartbeam import sections
from smartbeam.model import BeamModel


@dataclass
class DesignMetrics:
    name: str
    material: str
    shape: str
    dims: dict[str, float]  # m, shape-relevant only
    L: float  # m
    A: float  # m^2
    I: float  # m^4
    volume: float  # m^3
    mass: float  # kg
    cost: float | None  # INR (None when no cost/kg known)
    delta_max: float  # |v|max, m
    sigma_max: float  # |sigma|max, Pa
    M_max: float  # |M|max, N*m
    fos: float  # yield FOS = sigma_y / |sigma|max (inf if unstressed)
    method: str  # analysis route used


@dataclass
class ConstraintCheck:
    name: str
    value: float
    limit: float
    sense: str  # '<=' or '>='
    satisfied: bool
    margin_pct: float  # +ve = spare capacity, -ve = violation, relative to limit


def check(name: str, value: float, limit: float, sense: str = "<=") -> ConstraintCheck:
    if sense not in ("<=", ">="):
        raise ValueError("sense must be '<=' or '>='")
    slack = (limit - value) if sense == "<=" else (value - limit)
    ok = slack >= -1e-9 * max(abs(limit), 1e-300)
    margin = slack / abs(limit) * 100 if limit else math.copysign(math.inf, slack) if slack else 0.0
    return ConstraintCheck(name, float(value), float(limit), sense, bool(ok), float(margin))


def evaluate(model: BeamModel, result=None, cost_per_kg: float | None = None) -> DesignMetrics:
    """Metrics of a design. Pass an existing AnalysisResult to avoid re-solving."""
    if result is None:
        from smartbeam.analytical import analyze
        result = analyze(model)
    s, m = model.section, model.material
    A = sections.area(s)
    volume = A * model.L
    mass = volume * m.density
    rate = cost_per_kg if cost_per_kg is not None else m.cost_per_kg
    sig = abs(result.sigma_max.value)
    return DesignMetrics(
        name=model.name, material=m.name, shape=s.shape, dims=s.dims(), L=model.L, A=A,
        I=sections.second_moment(s), volume=volume, mass=mass, cost=None if rate is None else mass * rate,
        delta_max=abs(result.v_max.value), sigma_max=sig, M_max=abs(result.M_max.value),
        fos=m.yield_strength / sig if sig > 0 else math.inf, method=result.method,
    )
