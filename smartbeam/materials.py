"""Reference material library (SI). ALL VALUES ARE APPROXIMATE, typical handbook figures for the
named grade/temper; real stock varies with grade, heat treatment and supplier. Costs are rough Indian
market estimates in INR/kg and are user-editable. Fatigue properties are intentionally left None:
they must come from user-supplied test/handbook data, never from this library.
"""
from __future__ import annotations

import copy

from smartbeam.model import Material

GPa, MPa = 1e9, 1e6
APPROX_NOTE = "Approximate typical values for this grade; verify against supplier/standard data."

MATERIALS: dict[str, Material] = {
    "structural_steel": Material(
        name="Structural steel (IS 2062 E250 / ASTM A36)", E=200 * GPa, yield_strength=250 * MPa,
        uts=410 * MPa, nu=0.30, density=7850, cost_per_kg=70, note=APPROX_NOTE),
    "mild_steel": Material(
        name="Mild steel (AISI 1020, hot rolled)", E=200 * GPa, yield_strength=210 * MPa,
        uts=380 * MPa, nu=0.29, density=7870, cost_per_kg=60, note=APPROX_NOTE),
    "aluminium_6061_t6": Material(
        name="Aluminium 6061-T6", E=68.9 * GPa, yield_strength=276 * MPa,
        uts=310 * MPa, nu=0.33, density=2700, cost_per_kg=300, note=APPROX_NOTE),
    "copper": Material(
        name="Copper (C11000, annealed)", E=115 * GPa, yield_strength=69 * MPa,
        uts=220 * MPa, nu=0.34, density=8940, cost_per_kg=850, note=APPROX_NOTE),
    "brass": Material(
        name="Brass (C36000, half hard)", E=97 * GPa, yield_strength=310 * MPa,
        uts=400 * MPa, nu=0.31, density=8500, cost_per_kg=550, note=APPROX_NOTE),
}


def get_material(key: str) -> Material:
    """Return an independent copy of a library material (safe to edit)."""
    return copy.deepcopy(MATERIALS[key])


def material_keys() -> list[str]:
    return list(MATERIALS)


def custom_material(name: str, E: float, yield_strength: float, nu: float, density: float,
                    uts: float | None = None, cost_per_kg: float | None = None) -> Material:
    """User-defined material (SI). Raises ValueError on non-physical input."""
    errs = []
    if not E > 0:
        errs.append("E must be positive")
    if not yield_strength > 0:
        errs.append("yield strength must be positive")
    if not 0 <= nu < 0.5:
        errs.append("Poisson's ratio must be in [0, 0.5)")
    if not density > 0:
        errs.append("density must be positive")
    if uts is not None and uts < yield_strength:
        errs.append("UTS must not be below yield strength")
    if errs:
        raise ValueError("; ".join(errs))
    return Material(name=name, E=E, yield_strength=yield_strength, nu=nu, density=density, uts=uts,
                    cost_per_kg=cost_per_kg, approximate=False, note="User-defined material.")
