"""Beam model: pure data classes, validation, JSON round-trip and statics bookkeeping.

All quantities are SI (m, N, N/m, N*m, Pa, kg/m^3). Sign convention (binding, see docs/ARCHITECTURE.md):

* x runs along the beam from the left end (0) to the right end (L); y points up.
* Forces are positive in +x / +y. A PointLoad is a magnitude plus a direction angle measured
  counter-clockwise from +x:  -90 deg = straight down (default), +90 = up, 0 = axial to the right,
  180 = axial to the left.
* DistributedLoad intensities w1, w2 are given along ``direction`` ('down' default or 'up');
  ``q(x)`` returns the signed intensity in +y.
* Concentrated moments (and moment reactions) are positive counter-clockwise (CCW).
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Union

import numpy as np

SUPPORT_TYPES = ("fixed", "pin", "roller")
SHAPES = ("rect", "solid_circ", "hollow_circ")
LOAD_DIRECTIONS = ("down", "up")
# Restrained planar DOFs per support type, in (u, v, theta) order.
RESTRAINTS = {"fixed": ("u", "v", "theta"), "pin": ("u", "v"), "roller": ("v",)}
POS_TOL = 1e-9  # m, positional coincidence tolerance
SCHEMA_VERSION = 1


class ModelError(ValueError):
    """Raised when a BeamModel fails validation."""


class UnstableStructureError(ValueError):
    """Raised when supports cannot prevent rigid-body motion (mechanism)."""


# --------------------------------------------------------------------------- data classes
@dataclass
class Material:
    name: str
    E: float  # Young's modulus, Pa
    yield_strength: float  # Pa
    nu: float  # Poisson's ratio
    density: float  # kg/m^3
    uts: float | None = None  # ultimate tensile strength, Pa
    cost_per_kg: float | None = None  # INR/kg, approximate, user-editable
    # Fatigue data: never pre-filled; only from user input.
    sigma_f_prime: float | None = None  # Basquin fatigue strength coefficient, Pa
    b_fatigue: float | None = None  # Basquin exponent (negative)
    endurance_limit: float | None = None  # Pa
    approximate: bool = True  # library values are approximate reference data
    note: str = ""

    @property
    def G(self) -> float:
        """Shear modulus G = E / (2 (1 + nu)), Pa."""
        return self.E / (2.0 * (1.0 + self.nu))


@dataclass
class Section:
    """Cross-section. rect: b (width), h (depth); solid_circ: d; hollow_circ: D (outer), d (inner). Metres."""

    shape: str
    b: float | None = None
    h: float | None = None
    d: float | None = None
    D: float | None = None

    @classmethod
    def rect(cls, b: float, h: float) -> "Section":
        return cls("rect", b=b, h=h)

    @classmethod
    def solid_circ(cls, d: float) -> "Section":
        return cls("solid_circ", d=d)

    @classmethod
    def hollow_circ(cls, D: float, d: float) -> "Section":
        return cls("hollow_circ", D=D, d=d)

    @property
    def depth(self) -> float:
        """Overall depth (extent in y), m."""
        return {"rect": self.h, "solid_circ": self.d, "hollow_circ": self.D}[self.shape]

    def dims(self) -> dict[str, float]:
        """Only the dimensions relevant to this shape, e.g. {'b':..,'h':..}."""
        keys = {"rect": ("b", "h"), "solid_circ": ("d",), "hollow_circ": ("D", "d")}[self.shape]
        return {k: getattr(self, k) for k in keys}


@dataclass
class Support:
    type: str  # 'fixed' | 'pin' | 'roller'
    x: float

    @property
    def restrained(self) -> tuple[str, ...]:
        return RESTRAINTS[self.type]


_RIGHT_ANGLES = {0.0: (1.0, 0.0), 90.0: (0.0, 1.0), 180.0: (-1.0, 0.0), 270.0: (0.0, -1.0)}


def _cos_sin(deg: float) -> tuple[float, float]:
    """(cos, sin) of an angle in degrees, exact at multiples of 90 deg. Plain cos(-90 deg) = 6e-17 would give every
    vertical load a round-off axial force, i.e. N(x) and sigma != 0 on an unstressed beam (FOS 7e18, not inf)."""
    exact = _RIGHT_ANGLES.get(deg % 360.0)
    if exact:
        return exact
    r = math.radians(deg)
    return math.cos(r), math.sin(r)


@dataclass
class PointLoad:
    x: float
    magnitude: float  # N
    angle_deg: float = -90.0  # CCW from +x; -90 = downward
    kind: str = field(default="point", init=False, repr=False)

    @property
    def fx(self) -> float:
        return self.magnitude * _cos_sin(self.angle_deg)[0]

    @property
    def fy(self) -> float:
        return self.magnitude * _cos_sin(self.angle_deg)[1]

    @property
    def is_transverse(self) -> bool:
        return abs(self.fx) <= 1e-9 * max(1.0, abs(self.magnitude))


@dataclass
class DistributedLoad:
    """Linearly varying transverse load from x1 to x2 (UDL: w1 == w2; triangular: one end 0)."""

    x1: float
    x2: float
    w1: float  # N/m at x1, along `direction`
    w2: float  # N/m at x2, along `direction`
    direction: str = "down"  # 'down' | 'up'
    kind: str = field(default="distributed", init=False, repr=False)

    @property
    def sign(self) -> float:
        return -1.0 if self.direction == "down" else 1.0

    def q(self, x):
        """Signed intensity in +y (N/m) at x (scalar or array); zero outside [x1, x2]."""
        x = np.asarray(x, dtype=float)
        t = (x - self.x1) / (self.x2 - self.x1)
        val = self.sign * (self.w1 + (self.w2 - self.w1) * t)
        return np.where((x >= self.x1) & (x <= self.x2), val, 0.0)

    @property
    def resultant(self) -> float:
        """Signed resultant force in +y, N."""
        return self.sign * 0.5 * (self.w1 + self.w2) * (self.x2 - self.x1)

    @property
    def centroid(self) -> float:
        """x of the resultant's line of action (for w1 + w2 != 0)."""
        a, w1, w2 = self.x2 - self.x1, self.w1, self.w2
        if abs(w1 + w2) < 1e-300:
            return 0.5 * (self.x1 + self.x2)
        return self.x1 + a * (w1 + 2.0 * w2) / (3.0 * (w1 + w2))


@dataclass
class MomentLoad:
    x: float
    M: float  # N*m, positive CCW (use negative for CW)
    kind: str = field(default="moment", init=False, repr=False)

    @classmethod
    def cw(cls, x: float, magnitude: float) -> "MomentLoad":
        return cls(x, -abs(magnitude))

    @classmethod
    def ccw(cls, x: float, magnitude: float) -> "MomentLoad":
        return cls(x, abs(magnitude))


Load = Union[PointLoad, DistributedLoad, MomentLoad]
_LOAD_CLASSES = {"point": PointLoad, "distributed": DistributedLoad, "moment": MomentLoad}


@dataclass
class Reaction:
    """Support reaction acting ON the beam. Fx, Fy in N (+x, +y); M in N*m (CCW +)."""

    support_index: int
    type: str
    x: float
    Fx: float = 0.0
    Fy: float = 0.0
    M: float = 0.0


@dataclass
class Extreme:
    """Signed value at the location of the maximum absolute value (or of the max/min for stress)."""

    value: float
    x: float
    where: str = ""  # optional tag, e.g. 'top' / 'bottom' fibre

    @classmethod
    def of(cls, x, values, where: str = "") -> "Extreme":
        values = np.asarray(values, dtype=float)
        i = int(np.argmax(np.abs(values)))
        return cls(float(values[i]), float(np.asarray(x)[i]), where)


@dataclass
class BeamModel:
    L: float
    section: Section
    material: Material
    supports: list[Support] = field(default_factory=list)
    loads: list[Load] = field(default_factory=list)
    name: str = ""

    @property
    def point_loads(self) -> list[PointLoad]:
        return [ld for ld in self.loads if isinstance(ld, PointLoad)]

    @property
    def distributed_loads(self) -> list[DistributedLoad]:
        return [ld for ld in self.loads if isinstance(ld, DistributedLoad)]

    @property
    def moment_loads(self) -> list[MomentLoad]:
        return [ld for ld in self.loads if isinstance(ld, MomentLoad)]

    def copy(self) -> "BeamModel":
        return copy.deepcopy(self)

    # ---- serialisation
    def to_dict(self) -> dict:
        loads = []
        for ld in self.loads:
            d = asdict(ld)
            d["kind"] = ld.kind
            loads.append(d)
        return {
            "version": SCHEMA_VERSION,
            "name": self.name,
            "L": self.L,
            "section": asdict(self.section),
            "material": asdict(self.material),
            "supports": [asdict(s) for s in self.supports],
            "loads": loads,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BeamModel":
        loads = []
        for d in data.get("loads", []):
            d = dict(d)
            klass = _LOAD_CLASSES[d.pop("kind")]
            loads.append(klass(**d))
        return cls(
            L=float(data["L"]),
            section=Section(**data["section"]),
            material=Material(**data["material"]),
            supports=[Support(**s) for s in data.get("supports", [])],
            loads=loads,
            name=data.get("name", ""),
        )

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "BeamModel":
        return cls.from_dict(json.loads(text))


# --------------------------------------------------------------------------- validation
def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate(model: BeamModel) -> list[str]:
    """Return human-readable data errors (empty list = valid). Stability is checked separately."""
    errs: list[str] = []
    L = model.L
    if not (_finite(L) and L > 0):
        return [f"Beam length must be a positive number (got {L!r})."]

    s = model.section
    if s.shape not in SHAPES:
        errs.append(f"Unknown section shape {s.shape!r}; expected one of {SHAPES}.")
    else:
        for k, v in s.dims().items():
            if not (_finite(v) and v > 0):
                errs.append(f"Section dimension {k} must be positive (got {v!r}).")
        if s.shape == "hollow_circ" and _finite(s.D) and _finite(s.d) and s.d >= s.D:
            errs.append("Hollow section: inner diameter d must be smaller than outer diameter D.")

    m = model.material
    for k in ("E", "yield_strength", "density"):
        v = getattr(m, k)
        if not (_finite(v) and v > 0):
            errs.append(f"Material {k} must be positive (got {v!r}).")
    if not (_finite(m.nu) and 0 <= m.nu < 0.5):
        errs.append(f"Poisson's ratio must be in [0, 0.5) (got {m.nu!r}).")
    if m.uts is not None and _finite(m.uts) and _finite(m.yield_strength) and m.uts < m.yield_strength:
        errs.append("Ultimate tensile strength must not be below yield strength.")

    if not model.supports:
        errs.append("At least one support is required.")
    xs = []
    for i, sp in enumerate(model.supports):
        if sp.type not in SUPPORT_TYPES:
            errs.append(f"Support {i + 1}: unknown type {sp.type!r}; expected one of {SUPPORT_TYPES}.")
        if not (_finite(sp.x) and -POS_TOL <= sp.x <= L + POS_TOL):
            errs.append(f"Support {i + 1}: position x={sp.x!r} must lie on the beam [0, {L:g}] m.")
        elif any(abs(sp.x - x0) <= POS_TOL for x0 in xs):
            errs.append(f"Support {i + 1}: another support already sits at x={sp.x:g} m.")
        xs.append(sp.x)

    for i, ld in enumerate(model.loads):
        tag = f"Load {i + 1} ({getattr(ld, 'kind', type(ld).__name__)})"
        if isinstance(ld, PointLoad):
            if not all(_finite(v) for v in (ld.x, ld.magnitude, ld.angle_deg)):
                errs.append(f"{tag}: position, magnitude and angle must be finite numbers.")
            elif not -POS_TOL <= ld.x <= L + POS_TOL:
                errs.append(f"{tag}: x={ld.x:g} m is off the beam [0, {L:g}] m.")
        elif isinstance(ld, DistributedLoad):
            if not all(_finite(v) for v in (ld.x1, ld.x2, ld.w1, ld.w2)):
                errs.append(f"{tag}: x1, x2, w1, w2 must be finite numbers.")
            else:
                if not (-POS_TOL <= ld.x1 < ld.x2 <= L + POS_TOL):
                    errs.append(f"{tag}: need 0 <= x1 < x2 <= L (got x1={ld.x1:g}, x2={ld.x2:g}).")
            if ld.direction not in LOAD_DIRECTIONS:
                errs.append(f"{tag}: direction must be one of {LOAD_DIRECTIONS}.")
        elif isinstance(ld, MomentLoad):
            if not (_finite(ld.x) and _finite(ld.M)):
                errs.append(f"{tag}: position and moment must be finite numbers.")
            elif not -POS_TOL <= ld.x <= L + POS_TOL:
                errs.append(f"{tag}: x={ld.x:g} m is off the beam [0, {L:g}] m.")
        else:
            errs.append(f"{tag}: unsupported load type {type(ld).__name__}.")
    return errs


# --------------------------------------------------------------------------- statics helpers
def is_transverse_only(model: BeamModel) -> bool:
    """True when no load has an axial (x) component, so pure transverse bending applies."""
    return all(ld.is_transverse for ld in model.point_loads)


def classify(model: BeamModel) -> str:
    """'cantilever' | 'simply-supported' | 'general'.

    cantilever: exactly one support, fixed, at an end.
    simply-supported: exactly two supports, at x=0 and x=L, each pin or roller, at least one pin.
    """
    sup, L = model.supports, model.L
    at_end = lambda x: abs(x) <= POS_TOL or abs(x - L) <= POS_TOL * max(1.0, L)  # noqa: E731
    if len(sup) == 1 and sup[0].type == "fixed" and at_end(sup[0].x):
        return "cantilever"
    if len(sup) == 2:
        a, b = sorted(sup, key=lambda s: s.x)
        if (abs(a.x) <= POS_TOL and abs(b.x - L) <= POS_TOL * max(1.0, L)
                and {a.type, b.type} <= {"pin", "roller"} and "pin" in (a.type, b.type)):
            return "simply-supported"
    return "general"


@dataclass
class Stability:
    stable: bool
    determinate: bool
    degree: int  # degree of static indeterminacy (0 = determinate); meaningless if unstable
    n_restraints: int
    rank: int
    required: int  # 3 (planar) or 2 (transverse)
    mode: str  # 'planar' | 'transverse'
    message: str


def check_stability(model: BeamModel, mode: str | None = None) -> Stability:
    """Count restraints and test geometric stability via the rank of the restraint matrix.

    mode='transverse' (default when is_transverse_only): rigid-body DOFs are (v0, rotation);
    each support restraint contributes a row: v at x -> [1, x], theta -> [0, 1]; axial restraints ignored.
    mode='planar': rigid-body DOFs (u, v, rotation about x=0); rows u -> [1,0,0], v -> [0,1,x], theta -> [0,0,1].
    Stable iff rank == number of rigid-body DOFs; degree of indeterminacy = restraints - required.
    """
    mode = mode or ("transverse" if is_transverse_only(model) else "planar")
    rows = []
    for sp in model.supports:
        for dof in sp.restrained:
            if mode == "transverse":
                if dof == "v":
                    rows.append([1.0, sp.x])
                elif dof == "theta":
                    rows.append([0.0, 1.0])
            else:
                rows.append({"u": [1.0, 0.0, 0.0], "v": [0.0, 1.0, sp.x], "theta": [0.0, 0.0, 1.0]}[dof])
    required = 2 if mode == "transverse" else 3
    n = len(rows)
    rank = 0
    if rows:
        A = np.array(rows)
        A[:, -1] /= max(model.L, 1e-12)  # non-dimensionalise the rotation column (rank is unit-independent)
        rank = int(np.linalg.matrix_rank(A))
    stable = rank == required
    degree = n - required if stable else 0
    if not stable:
        what = "vertical translation or rotation" if mode == "transverse" else "translation or rotation"
        msg = (f"Unstable: the supports provide {n} restraint(s) of rank {rank} < {required}; the beam can undergo "
               f"rigid-body {what}. Add or change supports (e.g. a fixed support, or a pin plus a roller "
               f"at different positions).")
        if mode == "planar" and all(s.type == "roller" for s in model.supports):
            msg += " Rollers cannot resist the axial load component; add a pin or fixed support."
    elif degree == 0:
        msg = f"Statically determinate ({mode}): reactions follow from equilibrium alone."
    else:
        msg = (f"Statically indeterminate to degree {degree} ({mode}): equilibrium alone is insufficient; "
               f"a stiffness (FEM) solution is required.")
    return Stability(stable, stable and degree == 0, degree, n, rank, required, mode, msg)
