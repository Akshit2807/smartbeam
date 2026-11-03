"""Cross-section properties (SI). All shapes are doubly symmetric, so the centroid is at mid-depth.

Section y-coordinate: measured from the centroid, positive up. y_top > 0, y_bottom < 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from smartbeam.model import Section


def area(s: Section) -> float:
    """A: rect b*h; solid pi d^2/4; hollow pi (D^2 - d^2)/4.  m^2"""
    if s.shape == "rect":
        return s.b * s.h
    if s.shape == "solid_circ":
        return math.pi * s.d**2 / 4.0
    return math.pi * (s.D**2 - s.d**2) / 4.0


def centroid(s: Section) -> float:
    """Height of the centroid above the bottom fibre, m (= depth/2 for these symmetric shapes)."""
    return s.depth / 2.0


def second_moment(s: Section) -> float:
    """I about the horizontal centroidal axis: rect b h^3/12; solid pi d^4/64; hollow pi (D^4 - d^4)/64.  m^4"""
    if s.shape == "rect":
        return s.b * s.h**3 / 12.0
    if s.shape == "solid_circ":
        return math.pi * s.d**4 / 64.0
    return math.pi * (s.D**4 - s.d**4) / 64.0


def y_top(s: Section) -> float:
    """Signed y of the top fibre (positive), m."""
    return s.depth / 2.0


def y_bottom(s: Section) -> float:
    """Signed y of the bottom fibre (negative), m."""
    return -s.depth / 2.0


def section_modulus(s: Section) -> float:
    """Elastic section modulus Z = I / c, c = depth/2 (same for top and bottom).  m^3"""
    return second_moment(s) / y_top(s)


def radius_of_gyration(s: Section) -> float:
    """r = sqrt(I/A), m."""
    return math.sqrt(second_moment(s) / area(s))


def _chord(R: float, y):
    """Full chord width of a circle of radius R at height y (0 outside)."""
    return 2.0 * np.sqrt(np.clip(R**2 - np.square(y), 0.0, None))


def _q_circle(R: float, y):
    """First moment about the centroid of the part of a solid circle (radius R) above y: (2/3)(R^2 - y^2)^(3/2)."""
    y = np.clip(y, -R, R)
    return (2.0 / 3.0) * np.clip(R**2 - np.square(y), 0.0, None) ** 1.5


def width_at(s: Section, y):
    """Material width b(y) cut by a horizontal line at height y (scalar or array), m. 0 outside the section."""
    y = np.asarray(y, dtype=float)
    if s.shape == "rect":
        out = np.where(np.abs(y) <= s.h / 2.0, s.b, 0.0)
    elif s.shape == "solid_circ":
        out = _chord(s.d / 2.0, y)
    else:
        out = _chord(s.D / 2.0, y) - _chord(s.d / 2.0, y)
    return out if out.ndim else float(out)


def first_moment(s: Section, y):
    """Q(y): first moment, about the neutral axis, of the area lying above height y. m^3.

    rect: b/2 (c^2 - y^2); solid circle: (2/3)(R^2 - y^2)^1.5; hollow: outer minus inner circle.
    Q is symmetric in y and maximal at y = 0.
    """
    y = np.asarray(y, dtype=float)
    if s.shape == "rect":
        c = s.h / 2.0
        out = 0.5 * s.b * np.clip(c**2 - np.square(y), 0.0, None)
    elif s.shape == "solid_circ":
        out = _q_circle(s.d / 2.0, y)
    else:
        out = _q_circle(s.D / 2.0, y) - _q_circle(s.d / 2.0, y)
    return out if out.ndim else float(out)


def shear_stress(s: Section, V: float, y):
    """Transverse shear stress tau(y) = V Q(y) / (I b(y)), Pa (0 where b = 0)."""
    y = np.asarray(y, dtype=float)
    b = np.asarray(width_at(s, y), dtype=float)
    Q = np.asarray(first_moment(s, y), dtype=float)
    out = np.divide(V * Q, second_moment(s) * b, out=np.zeros_like(b), where=b > 0)
    return out if out.ndim else float(out)


def shear_correction_factor(s: Section, nu: float = 0.3) -> float:
    """Timoshenko shear correction factor kappa.

    rect: 5/6 (classical energy value); solid circle: 9/10 (classical value);
    hollow circle: Cowper (1966) kappa = 6(1+nu)(1+m^2)^2 / [(7+6nu)(1+m^2)^2 + (20+12nu) m^2], m = d/D,
    which tends to ~0.53 for thin tubes (nu = 0.3).
    """
    if s.shape == "rect":
        return 5.0 / 6.0
    if s.shape == "solid_circ":
        return 0.9
    m2 = (s.d / s.D) ** 2
    return 6 * (1 + nu) * (1 + m2) ** 2 / ((7 + 6 * nu) * (1 + m2) ** 2 + (20 + 12 * nu) * m2)


@dataclass(frozen=True)
class SectionProperties:
    A: float  # m^2
    I: float  # m^4
    Z: float  # m^3
    r_g: float  # m
    y_top: float  # m (positive)
    y_bottom: float  # m (negative)
    centroid: float  # m above bottom fibre
    depth: float  # m
    kappa: float  # shear correction factor
    Q_max: float  # m^3, first moment at the neutral axis


def properties(s: Section, nu: float = 0.3) -> SectionProperties:
    return SectionProperties(
        A=area(s), I=second_moment(s), Z=section_modulus(s), r_g=radius_of_gyration(s),
        y_top=y_top(s), y_bottom=y_bottom(s), centroid=centroid(s), depth=s.depth,
        kappa=shear_correction_factor(s, nu), Q_max=first_moment(s, 0.0),
    )
