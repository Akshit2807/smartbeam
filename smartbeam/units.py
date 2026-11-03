"""Unit conversion. The engine is SI; convert only at the UI / report boundary.

    to_si(5, "kN") -> 5000.0          from_si(2e-3, "mm") -> 2.0
    convert(1, "kN*m", "N*mm") -> 1e6  fmt(2.5e8, "MPa") -> "250 MPa"
"""
from __future__ import annotations

FACTORS: dict[str, float] = {
    # length
    "m": 1.0, "cm": 1e-2, "mm": 1e-3,
    # force
    "N": 1.0, "kN": 1e3,
    # distributed load
    "N/m": 1.0, "kN/m": 1e3, "N/mm": 1e3,
    # moment
    "N*m": 1.0, "N·m": 1.0, "kN*m": 1e3, "kN·m": 1e3, "N*mm": 1e-3, "N·mm": 1e-3,
    # stress / modulus
    "Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9, "N/mm^2": 1e6, "N/mm²": 1e6,
    # area, second moment, section modulus
    "m^2": 1.0, "m²": 1.0, "cm^2": 1e-4, "cm²": 1e-4, "mm^2": 1e-6, "mm²": 1e-6,
    "m^3": 1.0, "m³": 1.0, "cm^3": 1e-6, "cm³": 1e-6, "mm^3": 1e-9, "mm³": 1e-9,
    "m^4": 1.0, "m⁴": 1.0, "cm^4": 1e-8, "cm⁴": 1e-8, "mm^4": 1e-12, "mm⁴": 1e-12,
    # mass, density
    "kg": 1.0, "g": 1e-3, "t": 1e3,
    "kg/m^3": 1.0, "kg/m³": 1.0, "g/cm^3": 1e3, "g/cm³": 1e3,
    # angle (SI = rad)
    "rad": 1.0, "mrad": 1e-3, "deg": 3.141592653589793 / 180.0, "°": 3.141592653589793 / 180.0,
    # dimensionless / strain
    "-": 1.0, "%": 1e-2, "µε": 1e-6, "ue": 1e-6,
}


def to_si(value, unit: str):
    """Value expressed in `unit` -> SI. Works on scalars and NumPy arrays."""
    return value * FACTORS[unit]


def from_si(value, unit: str):
    """SI value -> value expressed in `unit`."""
    return value / FACTORS[unit]


def convert(value, src: str, dst: str):
    return value * FACTORS[src] / FACTORS[dst]


def fmt(value_si: float, unit: str, sig: int = 4) -> str:
    """Format an SI value in `unit` with `sig` significant figures, e.g. fmt(0.0123, 'mm') -> '12.3 mm'."""
    return f"{from_si(value_si, unit):.{sig}g} {unit}"
