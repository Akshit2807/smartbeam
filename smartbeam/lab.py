"""Virtual laboratory: guided experiments, design challenges, sensitivity analysis. OWNER: lab engineer.

Pure calculation helpers (no Streamlit). Every number comes from analytical.analyze / fem.solve /
fem.convergence_study / fem.compare_theories / optimize.optimize; conclusions are generated from the
computed tables, never written in advance. Tables use SI columns whose suffix names the unit
(see COLUMNS for display labels and units).
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from smartbeam import sections
from smartbeam.materials import get_material
from smartbeam.metrics import ConstraintCheck, DesignMetrics, check, evaluate
from smartbeam.model import (SHAPES, BeamModel, DistributedLoad, Material, MomentLoad, PointLoad, Section,
                             Support)
from smartbeam.units import fmt

# Sweep parameter names understood by scale_parameter / sensitivity helpers.
PARAMS = ("L", "E", "load", "b", "h", "d", "D")  # 'load' scales every load magnitude

PARAM_LABELS = {"L": "Span L", "E": "Young's modulus E", "load": "Load magnitude", "b": "Width b",
                "h": "Depth h", "d": "Diameter d", "D": "Outer diameter D"}

# SI column -> (display label, display unit). Unit "" = dimensionless / text.
COLUMNS: dict[str, tuple[str, str]] = {
    "h_m": ("Depth h", "mm"), "b_m": ("Width b", "mm"), "L_m": ("Span L", "m"), "a_m": ("Load position a", "m"),
    "D_m": ("Outer D", "mm"), "d_m": ("Inner d", "mm"), "wall_m": ("Wall t", "mm"),
    "E_Pa": ("E", "GPa"), "W_N": ("Total load W", "kN"), "A_m2": ("Area A", "mm²"), "I_m4": ("I", "mm⁴"),
    "volume_m3": ("Volume", "cm³"), "mass_kg": ("Mass", "kg"), "cost_inr": ("Cost (₹)", ""),
    "delta_max_m": ("Max deflection δ", "mm"), "x_delta_m": ("x at δmax", "m"),
    "sigma_max_Pa": ("Max stress σ", "MPa"), "M_max_Nm": ("Max moment M", "N·m"), "x_M_m": ("x at Mmax", "m"),
    "R_left_N": ("R left", "N"), "R_right_N": ("R right", "N"),
    "stiffness_N_per_m": ("Stiffness k = W/δ", "N/mm"), "k_per_kg": ("k per kg", "N/mm"),
    "fos": ("Yield FOS", ""), "v_eb_m": ("δ Euler–Bernoulli", "mm"), "v_timo_m": ("δ Timoshenko", "mm"),
    "diff_pct": ("Shear share (%)", ""), "diff_theory_pct": ("Closed-form share (%)", ""),
    "v_max_m": ("FEM δmax", "mm"), "M_fe_max_Nm": ("FEM M from curvature", "N·m"),
    "err_v_pct": ("δ error (%)", ""), "err_M_pct": ("M error (%)", ""), "err_M_fe_pct": ("M-curvature error (%)", ""),
    "case": ("Case", ""), "material": ("Material", ""), "support": ("Supports", ""), "load": ("Load", ""),
    "series": ("Series", ""), "scale": ("Scale factor s", ""), "ratio": ("Bore ratio d/D", ""),
    "a_over_L": ("a/L", ""), "L_over_h": ("L/h", ""), "phi": ("φ = 12EI/κGAL²", ""),
    "utilisation": ("Utilisation (≤ 1 passes)", ""), "elasticity": ("Elasticity (%/%)", ""),
    "n_requested": ("Elements requested", ""), "n_elements": ("Elements n", ""), "method": ("Solver route", ""),
}


@dataclass
class Experiment:
    key: str
    title: str
    objective: str
    explanation: str
    equations: list[str]  # LaTeX
    parameter: str  # swept quantity name (human readable)
    hypothesis: str = ""
    x: str = ""  # table column for the plot's x axis
    y: tuple[str, ...] = ()  # table columns to plot
    log_axes: bool = False  # log-log plot
    group: str | None = None  # table column that splits the plot into series
    plot: str = "line"  # 'line' | 'bar'


@dataclass
class ExperimentResult:
    key: str
    title: str
    objective: str
    explanation: str
    equations: list[str]
    parameter: str
    table: pd.DataFrame  # one row per run, SI columns
    conclusion: str  # derived from the computed numbers
    hypothesis: str = ""
    x: str = ""
    y: tuple[str, ...] = ()
    log_axes: bool = False
    group: str | None = None
    plot: str = "line"
    initial_conditions: dict[str, str] = field(default_factory=dict)  # readable description of the base beam
    findings: dict[str, float] = field(default_factory=dict)  # key computed numbers (fitted slopes, ratios)
    parameter_changes: dict[str, str] = field(default_factory=dict)  # what was varied, readable


@dataclass
class Challenge:
    key: str
    title: str
    brief: str
    base_model: BeamModel
    delta_allow: float | None  # m
    min_fos: float | None
    max_mass: float | None  # kg
    editable: tuple[str, ...] = field(default=("section", "material"))
    shapes: tuple[str, ...] = SHAPES  # allowed section shapes
    bounds: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)  # shape -> dim -> (lo, hi) m
    min_wall: float | None = None  # hollow tube minimum wall thickness, m
    beat_material: str | None = None  # library key: design must be lighter than that material's optimum


@dataclass
class ChallengeGrade:
    passed: bool
    score: float  # 0-100
    checks: list[ConstraintCheck]
    metrics: DesignMetrics
    feedback: list[str]
    reference: object | None = None  # optimize.OptimizationResult of the reference optimum (same shape/material)


DEFAULT_BOUNDS = {"rect": {"b": (0.005, 0.5), "h": (0.005, 0.5)}, "solid_circ": {"d": (0.005, 0.5)},
                  "hollow_circ": {"D": (0.01, 0.6), "d": (0.002, 0.59)}}
SCORE_RULE = ("Score: 50 points for the deflection, FOS, mass and wall constraints (equal share each; a failed "
              "constraint keeps partial credit 1 + margin, never below 0; dimension limits only gate PASS) + 50 points "
              "for efficiency, awarded only when every constraint passes, = 50 × (reference optimum mass / your mass), capped at 50.")


# --------------------------------------------------------------------------- basic helpers
def default_base_model() -> BeamModel:
    """Lab reference beam: steel rectangle 30 x 60 mm, 1 m cantilever, 1 kN down at the free end."""
    return BeamModel(L=1.0, section=Section.rect(0.030, 0.060), material=get_material("structural_steel"),
                     supports=[Support("fixed", 0.0)], loads=[PointLoad(1.0, 1000.0, -90.0)],
                     name="Lab reference cantilever")


def total_load(model: BeamModel) -> float:
    """Sum of transverse load magnitudes (N): |Fy| of point loads + |resultant| of distributed loads."""
    return float(sum(abs(p.fy) for p in model.point_loads) + sum(abs(d.resultant) for d in model.distributed_loads))


def loglog_slope(x, y) -> float:
    """Least-squares slope of log|y| vs log|x| (e.g. ~3 for delta vs L under end load, ~-3 for delta vs h)."""
    x, y = np.abs(np.asarray(x, float)), np.abs(np.asarray(y, float))
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return math.nan
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def scale_parameter(model: BeamModel, param: str, factor: float) -> BeamModel:
    """Copy of model with one PARAMS entry multiplied by factor (loads/supports rescaled with L)."""
    if param not in PARAMS:
        raise ValueError(f"Unknown parameter {param!r}; choose from {PARAMS}.")
    m = model.copy()
    if param == "L":
        m.L *= factor
        for s in m.supports:
            s.x *= factor
        for ld in m.loads:
            if isinstance(ld, DistributedLoad):
                ld.x1 *= factor
                ld.x2 *= factor
            else:
                ld.x *= factor
    elif param == "E":
        m.material.E *= factor
    elif param == "load":
        for ld in m.loads:
            if isinstance(ld, PointLoad):
                ld.magnitude *= factor
            elif isinstance(ld, DistributedLoad):
                ld.w1 *= factor
                ld.w2 *= factor
            else:
                ld.M *= factor
    else:
        if param not in m.section.dims():
            raise ValueError(f"Section shape {m.section.shape!r} has no dimension {param!r}.")
        setattr(m.section, param, getattr(m.section, param) * factor)
    return m


def run_case(model: BeamModel) -> dict:
    """Analyse one design and return the SI metrics used throughout the lab."""
    from smartbeam.analytical import analyze
    r = analyze(model)
    met = evaluate(model, r)
    W = total_load(model)
    return {"delta_max_m": met.delta_max, "x_delta_m": r.v_max.x, "sigma_max_Pa": met.sigma_max,
            "M_max_Nm": met.M_max, "x_M_m": r.M_max.x, "fos": met.fos, "mass_kg": met.mass,
            "volume_m3": met.volume, "A_m2": met.A, "I_m4": met.I, "cost_inr": met.cost,
            "stiffness_N_per_m": W / met.delta_max if met.delta_max > 0 and W > 0 else math.nan,
            "method": r.method, "_result": r}


def _row(model: BeamModel, **extra) -> dict:
    out = run_case(model)
    out.pop("_result")
    return {**extra, **out}


def describe(model: BeamModel) -> dict[str, str]:
    """Readable initial conditions of a beam (for lab reports)."""
    s = model.section
    dims = ", ".join(f"{k} = {fmt(v, 'mm')}" for k, v in s.dims().items())
    loads = []
    for ld in model.loads:
        if isinstance(ld, PointLoad):
            loads.append(f"point {fmt(ld.magnitude, 'N')} at {ld.angle_deg:g}° (x = {ld.x:g} m)")
        elif isinstance(ld, DistributedLoad):
            loads.append(f"{ld.direction} {ld.w1:g}→{ld.w2:g} N/m on {ld.x1:g}–{ld.x2:g} m")
        else:
            loads.append(f"moment {ld.M:g} N·m (CCW +) at x = {ld.x:g} m")
    return {"Span": fmt(model.L, "m"), "Section": f"{s.shape} ({dims})", "Material": model.material.name,
            "E": fmt(model.material.E, "GPa"), "Yield strength": fmt(model.material.yield_strength, "MPa"),
            "Supports": ", ".join(f"{sp.type} @ {sp.x:g} m" for sp in model.supports),
            "Loads": "; ".join(loads) or "none"}


def display_column(col: str, values):
    """(axis/header label, values in display units) for an SI lab-table column (unknown columns pass through)."""
    from smartbeam.units import from_si
    if col not in COLUMNS:
        return col, values
    label, unit = COLUMNS[col]
    if unit == "N/mm":
        values = np.asarray(values, float) / 1e3
    elif unit:
        values = from_si(np.asarray(values, float), unit)
    return (f"{label} ({unit})" if unit else label), values


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of an SI lab table with readable headers and display units."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        if not c.startswith("_"):
            label, vals = display_column(c, df[c])
            out[label] = vals
    return out


# --------------------------------------------------------------------------- model builders
SUPPORT_CONFIGS = ("cantilever", "simply supported", "propped cantilever", "fixed-fixed", "overhanging")
LOAD_TYPES = ("point", "udl", "triangular", "moment")


def support_set(config: str, L: float) -> list[Support]:
    """Support layout by name. 'overhanging' = pin at 0, roller at 0.75 L."""
    return {"cantilever": [Support("fixed", 0.0)],
            "simply supported": [Support("pin", 0.0), Support("roller", L)],
            "propped cantilever": [Support("fixed", 0.0), Support("roller", L)],
            "fixed-fixed": [Support("fixed", 0.0), Support("fixed", L)],
            "overhanging": [Support("pin", 0.0), Support("roller", 0.75 * L)]}[config]


def build_beam(L: float, section: Section, material: Material, support: str, load_type: str,
               magnitude: float, position: float | None = None, name: str = "") -> BeamModel:
    """Single-load beam for the explorer. magnitude: N (point, down), N/m (udl / triangular peak at L, down),
    N·m (moment, CCW +). position (m) applies to point and moment loads; default x = L."""
    x = L if position is None else position
    load = {"point": lambda: PointLoad(x, magnitude, -90.0),
            "udl": lambda: DistributedLoad(0.0, L, magnitude, magnitude),
            "triangular": lambda: DistributedLoad(0.0, L, 0.0, magnitude),
            "moment": lambda: MomentLoad(x, magnitude)}[load_type]()
    return BeamModel(L=L, section=section, material=material, supports=support_set(support, L), loads=[load],
                     name=name)


def _with(model: BeamModel, **kw) -> BeamModel:
    m = model.copy()
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def _rect_of(model: BeamModel) -> Section:
    s = model.section
    return Section.rect(s.b, s.h) if s.shape == "rect" else Section.rect(0.030, s.depth)


def _W(model: BeamModel) -> float:
    return total_load(model) or 1000.0


# --------------------------------------------------------------------------- experiments
EXPERIMENTS: dict[str, Experiment] = {e.key: e for e in [
    Experiment(
        "height_vs_stiffness", "Section depth vs stiffness",
        "Measure how the beam's bending stiffness k = W/δ changes with section depth h at constant width.",
        "For a rectangle I = bh³/12. Deflection is inversely proportional to EI, so stiffness grows with the cube of "
        "depth while mass grows only linearly. This is why beams are deep and narrow, and why floor joists stand on edge.",
        [r"I = \frac{b h^3}{12}", r"\delta \propto \frac{1}{EI} \propto h^{-3}", r"k = \frac{W}{\delta} \propto h^{3}"],
        "Depth h", "Doubling h makes the beam about 8× stiffer (δ ∝ h⁻³).",
        x="h_m", y=("delta_max_m",), log_axes=True),
    Experiment(
        "length_vs_deflection", "Span vs deflection",
        "Measure how maximum deflection grows with span when load positions scale with the span.",
        "Bending moment grows with span, and curvature is integrated twice over the span, so deflection grows with a high "
        "power of L: L³ for concentrated loads, L⁴ for a distributed load of fixed intensity (the total load also grows).",
        [r"\delta_{cant} = \frac{PL^3}{3EI}", r"\delta_{cant,UDL} = \frac{wL^4}{8EI}", r"\delta \propto L^{n}"],
        "Span L", "Deflection grows as L³ for point loads (L⁴ for constant-intensity distributed loads).",
        x="L_m", y=("delta_max_m",), log_axes=True),
    Experiment(
        "steel_vs_aluminium", "Steel vs aluminium",
        "Compare deflection, stress and mass of identical geometry in structural steel and aluminium 6061-T6, "
        "then size an aluminium section to match the steel stiffness.",
        "Deflection scales with 1/E, so aluminium (E ≈ 69 GPa) deflects about 2.9× more than steel (E ≈ 200 GPa). In a "
        "statically determinate beam the stress does not depend on E at all. Aluminium is about 3× lighter, so a deeper "
        "aluminium section can match the steel stiffness and still weigh less.",
        [r"\delta \propto \frac{1}{E}", r"\frac{\delta_{Al}}{\delta_{St}} = \frac{E_{St}}{E_{Al}}",
         r"h_{Al} = h_{St}\left(\frac{E_{St}}{E_{Al}}\right)^{1/3}"],
        "Material", "δ_Al/δ_St equals E_St/E_Al; the stiffness-matched aluminium beam is lighter.",
        x="case", y=("delta_max_m", "mass_kg"), plot="bar"),
    Experiment(
        "rect_vs_circular", "Rectangular vs circular section",
        "Compare sections with the same area (same mass): a deep rectangle, a square and a solid circle.",
        "With equal area, the section that puts material farthest from the neutral axis has the largest I and Z. A deep "
        "rectangle beats a square, which beats a circle (I_circle = A²/4π vs I_square = A²/12).",
        [r"I_{rect} = \frac{bh^3}{12}", r"I_{circ} = \frac{\pi d^4}{64} = \frac{A^2}{4\pi}",
         r"I_{square} = \frac{A^2}{12}"],
        "Section shape", "At equal area the deep rectangle is stiffest; the circle is least stiff.",
        x="case", y=("delta_max_m", "sigma_max_Pa"), plot="bar"),
    Experiment(
        "solid_vs_hollow", "Solid vs hollow circular section",
        "Keep the cross-section area (mass) constant and move material outwards by increasing the bore ratio d/D.",
        "At constant area, a hollow tube with a larger outer diameter has a much larger I: I/I_solid = (1 + r²)/(1 − r²) "
        "with r = d/D. Very thin walls eventually fail by local buckling, which this 1D beam model does not check.",
        [r"A = \frac{\pi}{4}(D^2 - d^2)", r"I = \frac{\pi}{64}(D^4 - d^4)", r"\frac{I}{I_{solid}} = \frac{1+r^2}{1-r^2}"],
        "Bore ratio r = d/D", "Hollow tubes of equal mass deflect less; the gain grows rapidly as r → 1.",
        x="ratio", y=("delta_max_m",)),
    Experiment(
        "point_vs_distributed", "Point load vs distributed load",
        "Apply the same total load W as one concentrated force and as a uniform load, on a cantilever and on a "
        "simply supported beam.",
        "Spreading the same total load reduces the peak bending moment by half in both cases and reduces deflection to "
        "3/8 (cantilever, tip load vs UDL) or 5/8 (simply supported, central load vs UDL).",
        [r"M_{cant} = WL \;\text{vs}\; \frac{WL}{2}", r"M_{SS} = \frac{WL}{4} \;\text{vs}\; \frac{WL}{8}",
         r"\frac{\delta_{UDL}}{\delta_P}\Big|_{cant} = \frac{3}{8},\quad \frac{\delta_{UDL}}{\delta_P}\Big|_{SS} = \frac{5}{8}"],
        "Load distribution", "A distributed load gives half the moment and less deflection than the same point load.",
        x="case", y=("delta_max_m", "M_max_Nm"), plot="bar"),
    Experiment(
        "load_position", "Load position on a simply supported beam",
        "Move a single point load across a simply supported span and record deflection, moment and reactions.",
        "The bending moment under the load is M = P a (L − a)/L, largest when the load is at mid-span. Reactions share "
        "the load in inverse proportion to the distance from each support.",
        [r"R_A = P\frac{L-a}{L},\; R_B = P\frac{a}{L}", r"M_{max} = \frac{P a (L-a)}{L}", r"\delta_{mid} = \frac{PL^3}{48EI}\;(a = L/2)"],
        "Load position a/L", "Moment and deflection peak with the load at mid-span.",
        x="a_over_L", y=("delta_max_m", "M_max_Nm")),
    Experiment(
        "cantilever_vs_ss", "Support conditions",
        "Compare the same beam and load under four support layouts: cantilever, simply supported, propped cantilever "
        "and fixed-fixed.",
        "Restraints reduce deflection dramatically. A cantilever deflects 16× more than a simply supported beam under "
        "the equivalent point load and 9.6× more under a UDL. The propped cantilever and fixed-fixed beams are statically "
        "indeterminate, so the engine solves them with the stiffness (FEM) method.",
        [r"\delta_{cant} = \frac{PL^3}{3EI},\; \delta_{SS} = \frac{PL^3}{48EI}", r"\delta_{FF} = \frac{PL^3}{192EI}",
         r"\delta_{cant,UDL} = \frac{wL^4}{8EI},\; \delta_{SS,UDL} = \frac{5wL^4}{384EI}"],
        "Support layout", "More restraint means less deflection; the cantilever is the most flexible.",
        x="support", y=("delta_max_m",), group="load", plot="bar"),
    Experiment(
        "eb_vs_timoshenko", "Euler–Bernoulli vs Timoshenko",
        "Find how the shear-deformation share of deflection depends on span-to-depth ratio L/h (cantilever, end load).",
        "Euler–Bernoulli theory ignores shear deformation. Timoshenko theory adds it: for a cantilever with an end load "
        "δ = PL³/3EI + PL/(κGA). The extra share is 3EI/(κGAL²), which falls as (h/L)², so it matters only for short, "
        "deep beams.",
        [r"\delta_T = \frac{PL^3}{3EI} + \frac{PL}{\kappa G A}", r"\frac{\delta_T - \delta_{EB}}{\delta_{EB}} = \frac{3EI}{\kappa G A L^2}",
         r"G = \frac{E}{2(1+\nu)}"],
        "Span-to-depth ratio L/h", "The difference falls as (L/h)⁻² and is small for slender beams (L/h > 10).",
        x="L_over_h", y=("diff_pct",), log_axes=True),
    Experiment(
        "mesh_convergence", "Mesh refinement and convergence",
        "Refine the FEM mesh of a simply supported beam under a full UDL and compare with the closed-form solution.",
        "Hermite beam elements with consistent loads reproduce the exact nodal deflections on every mesh, so nodal errors "
        "sit at round-off. The bending moment taken directly from the curvature of the cubic field (EI·v″) is only "
        "linear inside each element and genuinely converges as the mesh is refined.",
        [r"K_e = \frac{EI}{L_e^3}\begin{bmatrix}12 & 6L_e & -12 & 6L_e\\ 6L_e & 4L_e^2 & -6L_e & 2L_e^2\\ -12 & -6L_e & 12 & -6L_e\\ 6L_e & 2L_e^2 & -6L_e & 4L_e^2\end{bmatrix}",
         r"\delta_{SS,UDL} = \frac{5wL^4}{384EI}", r"M_{fe} = EI\,v''(x)"],
        "Number of elements n", "Nodal deflection is exact; curvature-based moment error falls as n increases.",
        x="n_elements", y=("err_M_fe_pct", "err_v_pct"), log_axes=True),
    Experiment(
        "volume_vs_performance", "Material volume vs performance",
        "Add material in two ways (uniform scaling of the section vs increasing depth only) and compare the gain in "
        "stiffness and strength per unit volume.",
        "Scaling every dimension by s multiplies A by s² and I by s⁴, so δ ∝ V⁻². Increasing only the depth multiplies A by "
        "s and I by s³, so δ ∝ V⁻³: adding depth is the most efficient use of material in bending.",
        [r"\text{uniform: } A \propto s^2,\; I \propto s^4 \Rightarrow \delta \propto V^{-2}",
         r"\text{depth only: } A \propto s,\; I \propto s^3 \Rightarrow \delta \propto V^{-3}", r"\sigma \propto \frac{1}{Z}"],
        "Section volume", "Depth-only scaling gives a steeper δ–V curve (exponent ≈ −3 vs −2).",
        x="volume_m3", y=("delta_max_m",), log_axes=True, group="series"),
]}

DEFAULT_VALUES = {
    "height_vs_stiffness": lambda m: _rect_of(m).h * np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]),
    "length_vs_deflection": lambda m: m.L * np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0]),
    "steel_vs_aluminium": lambda m: ["structural_steel", "aluminium_6061_t6"],
    "rect_vs_circular": lambda m: None,
    "solid_vs_hollow": lambda m: [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9],
    "point_vs_distributed": lambda m: None,
    "load_position": lambda m: np.round(np.linspace(0.05, 0.95, 19), 4),
    "cantilever_vs_ss": lambda m: ["cantilever", "simply supported", "propped cantilever", "fixed-fixed"],
    "eb_vs_timoshenko": lambda m: [2, 3, 4, 5, 6, 8, 10, 15, 20, 30],
    "mesh_convergence": lambda m: [1, 2, 4, 8, 16, 32, 64],
    "volume_vs_performance": lambda m: [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
}


def _exp_height(base, values):
    s = _rect_of(base)
    rows = [_row(_with(base, section=Section.rect(s.b, h)), h_m=float(h), b_m=s.b) for h in values]
    df = pd.DataFrame(rows)
    n = loglog_slope(df.h_m, df.delta_max_m)
    nk = loglog_slope(df.h_m, df.stiffness_N_per_m)
    nm = loglog_slope(df.h_m, df.mass_kg)
    c = (f"Fitted log–log slope of δ vs h = {n:.3f} (theory −3); stiffness k ∝ h^{nk:.3f} while mass ∝ h^{nm:.3f}. "
         f"Doubling the depth multiplies stiffness by 2^{nk:.2f} = {2 ** nk:.2f} for only {2 ** nm:.2f}× the mass. "
         f"Over the tested range δ fell from {fmt(df.delta_max_m.iloc[0], 'mm')} to {fmt(df.delta_max_m.iloc[-1], 'mm')}.")
    return df, c, {"slope_delta_h": n, "slope_k_h": nk}


def _exp_length(base, values):
    rows = [_row(scale_parameter(base, "L", L / base.L), L_m=float(L)) for L in values]
    df = pd.DataFrame(rows)
    n = loglog_slope(df.L_m, df.delta_max_m)
    if base.loads and all(isinstance(ld, PointLoad) for ld in base.loads):
        exp, why = 3, "only concentrated forces act (δ ∝ PL³/EI)"
    elif base.loads and all(isinstance(ld, DistributedLoad) for ld in base.loads):
        exp, why = 4, "only distributed loads of fixed intensity act (δ ∝ wL⁴/EI)"
    else:
        exp, why = None, "the load case is mixed, so no single power law is expected"
    c = f"Fitted log–log slope of δ vs L = {n:.3f}. "
    c += (f"Theory predicts {exp} because {why}; difference {abs(n - exp):.2g}. " if exp else f"Here {why}. ")
    c += (f"A {values[-1] / values[0]:.2g}× longer span gave {df.delta_max_m.iloc[-1] / df.delta_max_m.iloc[0]:.3g}× "
          f"the deflection; stress grew {df.sigma_max_Pa.iloc[-1] / df.sigma_max_Pa.iloc[0]:.3g}×.")
    return df, c, {"slope_delta_L": n, **({"expected": exp} if exp else {})}


def _exp_materials(base, values):
    rows = []
    for key in values:
        mat = get_material(key)
        rows.append(_row(_with(base, material=mat), case=mat.name.split(" (")[0], material=mat.name, E_Pa=mat.E))
    st, al = get_material(values[0]), get_material(values[-1])
    a, b = st.name.split(" (")[0], al.name.split(" (")[0]
    ratio = st.E / al.E
    s = base.section
    matched = Section.rect(s.b, s.h * ratio ** (1 / 3)) if s.shape == "rect" else Section(
        s.shape, **{k: v * ratio ** 0.25 for k, v in s.dims().items()})
    rows.append(_row(_with(base, material=al, section=matched), case=f"{al.name.split(' (')[0]} (stiffness-matched)",
                     material=al.name, E_Pa=al.E))
    df = pd.DataFrame(rows)
    d_ratio = df.delta_max_m[1] / df.delta_max_m[0]
    s_ratio = df.sigma_max_Pa[1] / df.sigma_max_Pa[0]
    m_ratio = df.mass_kg[2] / df.mass_kg[0]
    same_sigma = abs(s_ratio - 1) < 1e-6
    c = (f"Same geometry: δ({b})/δ({a}) = {d_ratio:.4f} and E({a})/E({b}) = {ratio:.4f} (difference "
         f"{abs(d_ratio / ratio - 1) * 100:.2g}%), confirming δ ∝ 1/E. Stress ratio = {s_ratio:.4f}: "
         + ("the stress is unchanged, as expected for a statically determinate beam. " if same_sigma else
            "the stress changed because the beam is statically indeterminate. ")
         + f"Mass ratio {a}/{b} at equal size = {df.mass_kg[0] / df.mass_kg[1]:.3f}. Scaling the {b} section for "
         f"equal stiffness gives δ = {fmt(df.delta_max_m[2], 'mm')} ({a}: {fmt(df.delta_max_m[0], 'mm')}) at "
         f"{m_ratio * 100:.1f}% of the {a} mass.")
    return df, c, {"delta_ratio": d_ratio, "E_ratio": ratio, "matched_mass_ratio": m_ratio}


def _exp_rect_circ(base, values):
    A = sections.area(base.section)
    r = base.section if base.section.shape == "rect" and base.section.h > base.section.b else Section.rect(
        math.sqrt(A / 2), math.sqrt(2 * A))
    cases = [(f"Rectangle {r.b * 1e3:.1f}×{r.h * 1e3:.1f} mm", r),
             (f"Square {math.sqrt(A) * 1e3:.1f} mm", Section.rect(math.sqrt(A), math.sqrt(A))),
             (f"Solid circle d = {math.sqrt(4 * A / math.pi) * 1e3:.1f} mm", Section.solid_circ(math.sqrt(4 * A / math.pi)))]
    df = pd.DataFrame([_row(_with(base, section=s), case=name) for name, s in cases])
    best, worst = df.loc[df.delta_max_m.idxmin()], df.loc[df.delta_max_m.idxmax()]
    c = (f"All three sections have A = {A * 1e6:.1f} mm² (mass {df.mass_kg[0]:.3g} kg). The {best.case} is stiffest "
         f"(δ = {fmt(best.delta_max_m, 'mm')}); the {worst.case} deflects {worst.delta_max_m / best.delta_max_m:.2f}× more. "
         f"I ratio square/circle = {df.I_m4[1] / df.I_m4[2]:.4f} (theory π/3 = {math.pi / 3:.4f}). "
         f"Peak stress ranges from {fmt(df.sigma_max_Pa.min(), 'MPa')} to {fmt(df.sigma_max_Pa.max(), 'MPa')}.")
    return df, c, {"I_square_over_circle": df.I_m4[1] / df.I_m4[2]}


def _exp_hollow(base, values):
    A = sections.area(base.section)
    rows = []
    for r in values:
        D = math.sqrt(4 * A / (math.pi * (1 - r * r)))
        s = Section.solid_circ(D) if r == 0 else Section.hollow_circ(D, r * D)
        rows.append(_row(_with(base, section=s), ratio=float(r), D_m=D, d_m=r * D, wall_m=D * (1 - r) / 2))
    df = pd.DataFrame(rows)
    i_theory = (1 + df.ratio ** 2) / (1 - df.ratio ** 2)
    i_err = float(np.max(np.abs(df.I_m4 / df.I_m4[0] / i_theory - 1)))
    c = (f"Equal area A = {A * 1e6:.1f} mm² throughout. Going from a solid bar to r = {df.ratio.iloc[-1]:g} cut δ from "
         f"{fmt(df.delta_max_m[0], 'mm')} to {fmt(df.delta_max_m.iloc[-1], 'mm')} "
         f"({df.delta_max_m[0] / df.delta_max_m.iloc[-1]:.2f}× stiffer) and σ by "
         f"{(1 - df.sigma_max_Pa.iloc[-1] / df.sigma_max_Pa[0]) * 100:.1f}%, with the same mass. I/I_solid matched "
         f"(1 + r²)/(1 − r²) to within {i_err:.1e}. The thinnest wall is {fmt(df.wall_m.iloc[-1], 'mm')}; local wall "
         "buckling is not modelled, so very thin tubes need a separate check.")
    return df, c, {"stiffness_gain": df.delta_max_m[0] / df.delta_max_m.iloc[-1]}


def _exp_point_udl(base, values):
    L, W = base.L, _W(base)
    rows = []
    for sup, px in (("cantilever", L), ("simply supported", L / 2)):
        for load, ld in (("point", PointLoad(px, W, -90.0)), ("UDL", DistributedLoad(0.0, L, W / L, W / L))):
            m = _with(base, supports=support_set(sup, L), loads=[ld])
            rows.append(_row(m, case=f"{sup} · {load}", support=sup, load=load, W_N=W))
    df = pd.DataFrame(rows)
    rc_d, rc_m = df.delta_max_m[1] / df.delta_max_m[0], df.M_max_Nm[1] / df.M_max_Nm[0]
    rs_d, rs_m = df.delta_max_m[3] / df.delta_max_m[2], df.M_max_Nm[3] / df.M_max_Nm[2]
    c = (f"Total load W = {fmt(W, 'kN')} in every case. Cantilever: δ_UDL/δ_P = {rc_d:.4f} (theory 0.375), "
         f"M_UDL/M_P = {rc_m:.4f} (theory 0.5). Simply supported: δ_UDL/δ_P = {rs_d:.4f} (theory 0.625), "
         f"M_UDL/M_P = {rs_m:.4f} (theory 0.5). Spreading the load always lowered both the peak moment and the deflection.")
    return df, c, {"cant_delta_ratio": rc_d, "ss_delta_ratio": rs_d, "cant_M_ratio": rc_m, "ss_M_ratio": rs_m}


def _exp_position(base, values):
    L, P = base.L, _W(base)
    rows = []
    for t in values:
        m = _with(base, supports=support_set("simply supported", L), loads=[PointLoad(t * L, P, -90.0)])
        r = run_case(m)
        res = r.pop("_result")
        rows.append({"a_over_L": float(t), "a_m": t * L, **r,
                     "R_left_N": res.reactions[0].Fy, "R_right_N": res.reactions[1].Fy})
    df = pd.DataFrame(rows)
    a = df.a_m
    m_theory = P * a * (L - a) / L
    err = float(np.max(np.abs(df.M_max_Nm / m_theory - 1)))
    imax = int(df.delta_max_m.idxmax())
    c = (f"P = {fmt(P, 'kN')} on a {L:g} m simply supported span. The largest deflection "
         f"({fmt(df.delta_max_m[imax], 'mm')}) occurs with the load at a/L = {df.a_over_L[imax]:g}; the largest moment "
         f"({df.M_max_Nm.max():.4g} N·m) at a/L = {df.a_over_L[int(df.M_max_Nm.idxmax())]:g}. The computed M_max matches "
         f"Pa(L − a)/L to within {err:.1e} (relative) at every position. Reactions follow R_A = P(L − a)/L: with the load "
         f"at a/L = {df.a_over_L[0]:g}, the near support carries {df.R_left_N[0] / P * 100:.1f}% of it. Note the maximum "
         "deflection does not sit under an off-centre load; it stays near mid-span.")
    return df, c, {"M_rel_error": err}


def _exp_supports(base, values):
    L, W = base.L, _W(base)
    rows = []
    for sup in values:
        for load in ("UDL", "point"):
            px = L if sup == "cantilever" else L / 2
            ld = DistributedLoad(0.0, L, W / L, W / L) if load == "UDL" else PointLoad(px, W, -90.0)
            rows.append(_row(_with(base, supports=support_set(sup, L), loads=[ld]), support=sup, load=load, W_N=W))
    df = pd.DataFrame(rows)
    get = lambda s, ld: df[(df.support == s) & (df.load == ld)].delta_max_m.iloc[0]  # noqa: E731
    c = f"Total load W = {fmt(W, 'kN')} (point loads at the tip for the cantilever, at mid-span otherwise). "
    found = {}
    if {"cantilever", "simply supported"} <= set(values):
        found["cant_ss_point"] = get("cantilever", "point") / get("simply supported", "point")
        found["cant_ss_udl"] = get("cantilever", "UDL") / get("simply supported", "UDL")
        c += (f"δ_cantilever/δ_SS = {found['cant_ss_point']:.3f} for the point load (theory 16) and "
              f"{found['cant_ss_udl']:.3f} for the UDL (theory 9.6). ")
    if {"fixed-fixed", "simply supported"} <= set(values):
        found["ff_ss_point"] = get("fixed-fixed", "point") / get("simply supported", "point")
        c += f"Fixing both ends cut the central-load deflection to {found['ff_ss_point']:.4f} of the SS value (theory 0.25). "
    stiff = df.loc[df.delta_max_m.idxmin()]
    c += (f"Stiffest layout: {stiff.support} ({stiff.load}). Indeterminate layouts were solved by "
          f"{', '.join(sorted(set(df.method) - {'analytical'})) or 'statics only'}.")
    return df, c, found


def _exp_timoshenko(base, values):
    from smartbeam.fem import compare_theories
    s, mat = base.section, base.material
    h = s.depth
    E, I, A = mat.E, sections.second_moment(s), sections.area(s)
    kappa = sections.shear_correction_factor(s, mat.nu)
    W = _W(base)
    rows = []
    for ratio in values:
        L = ratio * h
        m = _with(base, L=L, supports=support_set("cantilever", L), loads=[PointLoad(L, W, -90.0)])
        tc = compare_theories(m, n_elements=10)
        rows.append({"L_over_h": float(ratio), "L_m": L, "v_eb_m": tc.v_eb, "v_timo_m": tc.v_timoshenko,
                     "diff_pct": tc.diff_pct, "diff_theory_pct": 100 * 3 * E * I / (kappa * mat.G * A * L * L),
                     "phi": tc.phi_beam})
    df = pd.DataFrame(rows)
    n = loglog_slope(df.L_over_h, df.diff_pct)
    err = float(np.max(np.abs(df.diff_pct / df.diff_theory_pct - 1)))
    # L/h where the shear share drops to 5 % and 1 % (share ∝ (L/h)^-2 exactly for this case)
    c5 = math.sqrt(df.diff_theory_pct[0] / 5) * df.L_over_h[0]
    c1 = math.sqrt(df.diff_theory_pct[0] / 1) * df.L_over_h[0]
    c = (f"Cantilever, end load {fmt(W, 'kN')}, depth {fmt(h, 'mm')}, κ = {kappa:.4g}, G = {fmt(mat.G, 'GPa')}. "
         f"The Timoshenko FEM adds {df.diff_pct.iloc[0]:.2f}% at L/h = {df.L_over_h.iloc[0]:g} but only "
         f"{df.diff_pct.iloc[-1]:.3f}% at L/h = {df.L_over_h.iloc[-1]:g}. Fitted log–log slope = {n:.3f} (theory −2). "
         f"FEM matched the closed form 3EI/(κGAL²) to within {err:.1e} (relative). Shear deformation drops below 5% "
         f"at L/h ≈ {c5:.1f} and below 1% at L/h ≈ {c1:.1f}; beyond that Euler–Bernoulli is adequate.")
    return df, c, {"slope": n, "L_over_h_5pct": c5, "L_over_h_1pct": c1}


def _exp_mesh(base, values):
    from smartbeam.fem import convergence_study
    L, W = base.L, _W(base)
    m = _with(base, supports=support_set("simply supported", L), loads=[DistributedLoad(0.0, L, W / L, W / L)])
    cs = convergence_study(m, n_list=tuple(int(v) for v in values), theory="euler")
    df = pd.DataFrame({"n_requested": cs.n_requested or list(values), "n_elements": cs.n_elements,
                       "v_max_m": cs.v_max, "M_max_Nm": cs.M_max, "M_fe_max_Nm": cs.M_fe_max})
    found = {}
    if cs.error_pct:
        df["err_v_pct"] = cs.error_pct["v_max"]
        df["err_M_pct"] = cs.error_pct["M_max"]
        df["err_M_fe_pct"] = cs.error_pct["M_fe"]
        e, n = df.err_M_fe_pct.to_numpy(), df.n_elements.to_numpy()
        order = math.log(e[-2] / e[-1]) / math.log(n[-1] / n[-2]) if len(e) >= 2 and e[-1] > 0 else math.nan
        found = {"order_M_fe": order, "err_v_first_pct": float(df.err_v_pct.iloc[0]),
                 "err_v_last_pct": float(df.err_v_pct.iloc[-1])}
        c = (f"Simply supported, UDL w = {W / L:.4g} N/m. Reference δ = 5wL⁴/384EI = {fmt(cs.reference['v_max'], 'mm')}. "
             f"Peak-deflection error: {df.err_v_pct.iloc[0]:.3g}% with {n[0]} element(s), {df.err_v_pct.iloc[-1]:.1e}% "
             f"with {n[-1]}. Nodal deflections are exact on every mesh; a large error appears only while mid-span is "
             f"not a node and the peak comes from the cubic interpolation. The curvature-based moment EI·v″ erred by "
             f"{e[0]:.3g}% with {n[0]} element(s) and {e[-1]:.2g}% with {n[-1]}; observed convergence order from the two "
             f"finest meshes ≈ {order:.2f} (error ∝ n^−{order:.2f}). The equilibrium-recovered moment is exact on every "
             f"mesh (max error {df.err_M_pct.max():.1e}%).")
    else:
        c = "No closed-form reference applies; " + cs.note
    return df, c, found


def _exp_volume(base, values):
    s = base.section
    rows = []
    for f in values:
        rows.append(_row(_with(base, section=Section(s.shape, **{k: v * f for k, v in s.dims().items()})),
                         series="uniform scaling", scale=float(f)))
        if s.shape == "rect":
            rows.append(_row(_with(base, section=Section.rect(s.b, s.h * f)), series="depth only", scale=float(f)))
    df = pd.DataFrame(rows)
    df["k_per_kg"] = df.stiffness_N_per_m / df.mass_kg
    found = {}
    parts = []
    for name, g in df.groupby("series", sort=False):
        nd, ns = loglog_slope(g.volume_m3, g.delta_max_m), loglog_slope(g.volume_m3, g.sigma_max_Pa)
        found[f"{name}: delta"] = nd
        found[f"{name}: sigma"] = ns
        parts.append(f"{name}: δ ∝ V^{nd:.3f}, σ ∝ V^{ns:.3f}")
    c = "Fitted exponents — " + "; ".join(parts) + ". "
    if s.shape == "rect":
        c += ("Theory: uniform scaling δ ∝ V⁻², σ ∝ V⁻¹·⁵; depth-only δ ∝ V⁻³, σ ∝ V⁻². For the same added material, "
              "increasing the depth gives the larger gain in stiffness and strength.")
    else:
        c += "Theory for uniform scaling: δ ∝ V⁻², σ ∝ V⁻¹·⁵ (circular sections have no separate depth)."
    return df, c, found


_RUNNERS = {"height_vs_stiffness": _exp_height, "length_vs_deflection": _exp_length,
            "steel_vs_aluminium": _exp_materials, "rect_vs_circular": _exp_rect_circ, "solid_vs_hollow": _exp_hollow,
            "point_vs_distributed": _exp_point_udl, "load_position": _exp_position, "cantilever_vs_ss": _exp_supports,
            "eb_vs_timoshenko": _exp_timoshenko, "mesh_convergence": _exp_mesh,
            "volume_vs_performance": _exp_volume}


def run_experiment(key: str, base_model: BeamModel | None = None, values=None) -> ExperimentResult:
    """Run one guided experiment. base_model supplies span, section, material, supports and loads (each experiment
    changes only what it studies); values = the swept values (SI / keys), default DEFAULT_VALUES[key]."""
    if key not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {key!r}.")
    base = (base_model or default_base_model()).copy()
    if values is None:
        values = DEFAULT_VALUES[key](base)
    df, conclusion, found = _RUNNERS[key](base, values)
    e = EXPERIMENTS[key]
    label, shown = display_column(e.x, df[e.x].drop_duplicates())
    changes = {"Swept quantity": e.parameter, "Runs": str(len(df)),
               label: ", ".join(f"{v:.4g}" if isinstance(v, (int, float, np.number)) else str(v) for v in shown)}
    if e.group:
        changes["Series"] = ", ".join(map(str, df[e.group].unique()))
    return ExperimentResult(e.key, e.title, e.objective, e.explanation, list(e.equations), e.parameter,
                            df.reset_index(drop=True), conclusion, e.hypothesis, e.x, e.y, e.log_axes, e.group,
                            e.plot, describe(base), {k: float(v) for k, v in found.items()}, changes)


# --------------------------------------------------------------------------- design challenges
def _ss(L, w):
    return support_set("simply supported", L), [DistributedLoad(0.0, L, w, w)]


CHALLENGES: list[Challenge] = [
    Challenge(
        "pedestrian_plank", "Pedestrian bridge plank",
        "A 3 m plank spans a drainage channel, simply supported at both ends, and carries 2 kN/m of foot traffic. "
        "Keep deflection within L/360 and a yield FOS of at least 2, using a plank at least 200 mm wide. Minimise mass.",
        BeamModel(3.0, Section.rect(0.25, 0.03), get_material("structural_steel"), *_ss(3.0, 2000.0),
                  name="Pedestrian plank"),
        delta_allow=3.0 / 360, min_fos=2.0, max_mass=None, shapes=("rect",),
        bounds={"rect": {"b": (0.20, 0.60), "h": (0.01, 0.30)}}),
    Challenge(
        "cantilever_balcony", "Cantilever balcony joist",
        "A 1.5 m balcony joist is built into a wall (fixed) and carries 3 kN/m plus a 1 kN person at the edge. "
        "Deflection ≤ L/180, yield FOS ≥ 2 and mass ≤ 60 kg. Joist width between 50 and 300 mm.",
        BeamModel(1.5, Section.rect(0.10, 0.15), get_material("structural_steel"), [Support("fixed", 0.0)],
                  [DistributedLoad(0.0, 1.5, 3000.0, 3000.0), PointLoad(1.5, 1000.0, -90.0)], name="Balcony joist"),
        delta_allow=1.5 / 180, min_fos=2.0, max_mass=60.0,
        bounds={"rect": {"b": (0.05, 0.30), "h": (0.02, 0.40)}, "solid_circ": {"d": (0.02, 0.40)},
                "hollow_circ": {"D": (0.02, 0.40), "d": (0.005, 0.39)}}, min_wall=0.003),
    Challenge(
        "shelf_bracket", "Shelf bracket",
        "A 250 mm wall bracket (cantilever) supports 400 N at its tip. Tip deflection ≤ 0.5 mm, yield FOS ≥ 3, "
        "mass ≤ 0.5 kg.",
        BeamModel(0.25, Section.rect(0.02, 0.03), get_material("structural_steel"), [Support("fixed", 0.0)],
                  [PointLoad(0.25, 400.0, -90.0)], name="Shelf bracket"),
        delta_allow=0.0005, min_fos=3.0, max_mass=0.5,
        bounds={"rect": {"b": (0.005, 0.08), "h": (0.005, 0.10)}, "solid_circ": {"d": (0.005, 0.10)},
                "hollow_circ": {"D": (0.006, 0.10), "d": (0.002, 0.098)}}, min_wall=0.001),
    Challenge(
        "aluminium_vs_steel", "Aluminium vs steel weight challenge",
        "A 2 m simply supported beam carries 5 kN at mid-span. The material is fixed to aluminium 6061-T6. Meet "
        "δ ≤ L/250 and FOS ≥ 1.5 AND weigh less than the lightest possible steel rectangle (the optimum computed by "
        "the engine). Width between 20 and 200 mm.",
        BeamModel(2.0, Section.rect(0.04, 0.12), get_material("aluminium_6061_t6"), support_set("simply supported", 2.0),
                  [PointLoad(1.0, 5000.0, -90.0)], name="Aluminium beam"),
        delta_allow=2.0 / 250, min_fos=1.5, max_mass=None, editable=("section",), shapes=("rect",),
        bounds={"rect": {"b": (0.02, 0.20), "h": (0.02, 0.40)}}, beat_material="structural_steel"),
    Challenge(
        "hollow_tube", "Hollow tube sign post",
        "A 2.5 m sign post is a cantilever from its base, loaded by 600 N of wind at the top (treated as a transverse "
        "load). Use a circular tube with a wall of at least 2 mm. Top deflection ≤ L/100, FOS ≥ 2, mass ≤ 15 kg.",
        BeamModel(2.5, Section.hollow_circ(0.08, 0.07), get_material("structural_steel"), [Support("fixed", 0.0)],
                  [PointLoad(2.5, 600.0, -90.0)], name="Sign post"),
        delta_allow=0.025, min_fos=2.0, max_mass=15.0, shapes=("hollow_circ",),
        bounds={"hollow_circ": {"D": (0.02, 0.30), "d": (0.005, 0.296)}}, min_wall=0.002),
]
CHALLENGE_BY_KEY = {c.key: c for c in CHALLENGES}


def challenge_bounds(challenge: Challenge, shape: str) -> dict[str, tuple[float, float]]:
    return dict(challenge.bounds.get(shape) or DEFAULT_BOUNDS[shape])


def challenge_design(challenge: Challenge, model: BeamModel) -> BeamModel:
    """The graded design: the challenge's span, supports and loads with the submitted section (and material
    when editable). Anything else in `model` is ignored so a submission cannot change the problem."""
    d = challenge.base_model.copy()
    d.section = Section(**vars(model.section))
    if "material" in challenge.editable:
        d.material = model.material
    d.name = model.name or challenge.base_model.name
    return d


@functools.lru_cache(maxsize=64)
def _reference_cached(key: str, shape: str, material_json: str):
    import json

    from smartbeam.optimize import OptimizationProblem, optimize
    ch = CHALLENGE_BY_KEY[key]
    bounds = challenge_bounds(ch, shape)
    m = ch.base_model.copy()
    m.material = Material(**json.loads(material_json))
    start = {k: math.sqrt(lo * hi) for k, (lo, hi) in bounds.items()}
    if shape == "hollow_circ":
        start["d"] = max(bounds["d"][0], min(start["d"], start["D"] - 2 * (ch.min_wall or 1e-3) - 1e-4))
    m.section = Section(shape, **start)
    prob = OptimizationProblem(objective="mass", delta_allow=ch.delta_allow or 1e9, fos_min=max(ch.min_fos or 1.0, 1.0),
                               bounds=bounds, t_min=ch.min_wall or 1e-3)
    return optimize(m, prob)


def reference_optimum(challenge: Challenge, shape: str, material: Material):
    """optimize.optimize result for the lightest design of this shape/material meeting the challenge's deflection
    and FOS limits within its bounds (cached). Mass limits are checked against it, not imposed."""
    import json
    from dataclasses import asdict
    return _reference_cached(challenge.key, shape, json.dumps(asdict(material), sort_keys=True))


def grade_challenge(challenge: Challenge, model: BeamModel) -> ChallengeGrade:
    """Analyse the submitted design and grade each constraint pass/fail, plus a 0-100 score (see SCORE_RULE)."""
    design = challenge_design(challenge, model)
    shape = design.section.shape
    if shape not in challenge.shapes:
        raise ValueError(f"This challenge only allows {', '.join(challenge.shapes)} sections.")
    met = evaluate(design)
    checks: list[ConstraintCheck] = []
    if challenge.delta_allow is not None:
        checks.append(check("Max deflection (m)", met.delta_max, challenge.delta_allow, "<="))
    if challenge.min_fos is not None:
        checks.append(check("Yield FOS", met.fos, challenge.min_fos, ">="))
    if challenge.max_mass is not None:
        checks.append(check("Mass (kg)", met.mass, challenge.max_mass, "<="))
    beat = None
    if challenge.beat_material:
        beat = reference_optimum(challenge, shape, get_material(challenge.beat_material))
        if beat.optimized is not None:
            checks.append(check(f"Mass vs optimum {get_material(challenge.beat_material).name.split(' (')[0]} (kg)",
                                met.mass, beat.optimized.mass, "<="))
    if shape == "hollow_circ" and challenge.min_wall:
        checks.append(check("Wall thickness (m)", (design.section.D - design.section.d) / 2, challenge.min_wall, ">="))
    scored = list(checks)  # dimension limits below only gate PASS; they carry no score share
    if challenge.bounds:
        for k, v in design.section.dims().items():
            lo, hi = challenge_bounds(challenge, shape)[k]
            checks.append(check(f"{k} min (m)", v, lo, ">="))
            checks.append(check(f"{k} max (m)", v, hi, "<="))

    passed = all(c.satisfied for c in checks)
    share = 50.0 / len(scored) if scored else 50.0
    score = sum(share if c.satisfied else share * max(0.0, 1 + c.margin_pct / 100) for c in scored)
    ref = reference_optimum(challenge, shape, design.material)
    feedback = []
    for c in checks:
        if c.satisfied:
            continue
        if c.name.startswith("Max deflection"):
            f = met.delta_max / challenge.delta_allow
            feedback.append(f"Deflection {fmt(met.delta_max, 'mm')} exceeds {fmt(challenge.delta_allow, 'mm')} "
                            f"({f:.2f}×). δ ∝ 1/I: increase I by at least {f:.2f}× — for a rectangle, depth × "
                            f"{f ** (1 / 3):.3f}; or choose a stiffer material (higher E).")
        elif c.name == "Yield FOS":
            f = challenge.min_fos / met.fos
            feedback.append(f"FOS {met.fos:.2f} is below {challenge.min_fos:g}: stress must fall by {f:.2f}×. σ = M/Z — "
                            f"increase Z (rectangle depth × {f ** 0.5:.3f}) or use a stronger material.")
        elif c.name.startswith("Mass"):
            feedback.append(f"Mass {met.mass:.3g} kg exceeds the {c.limit:.3g} kg limit: remove material where it "
                            "contributes least (width rather than depth; inner bore of a tube).")
        else:
            feedback.append(f"{c.name.removesuffix(' (m)')}: {c.value * 1e3:.4g} mm violates the limit "
                            f"{c.limit * 1e3:.4g} mm.")
    if passed and ref.optimized is not None:
        eff = min(1.0, ref.optimized.mass / met.mass)
        score += 50 * eff
        over = (met.mass / ref.optimized.mass - 1) * 100
        feedback.append("All constraints satisfied. " + (
            f"Your design is {over:.1f}% heavier than the reference optimum for this shape and material." if over > 0.05
            else "Your design matches the reference optimum mass."))
    elif passed:
        score += 50
        feedback.append("All constraints satisfied. No reference optimum could be computed for this shape/material.")
    if not passed and ref.status != "OPTIMIZED":
        feedback.append(f"Note: the optimizer found no feasible design for this shape and material within the "
                        f"challenge bounds ({ref.message}) — try a different material or shape.")
    elif ref.optimized is not None and challenge.max_mass is not None and ref.optimized.mass > challenge.max_mass:
        feedback.append(f"Even the lightest feasible {shape} in this material weighs {ref.optimized.mass:.3g} kg, above "
                        f"the {challenge.max_mass:g} kg limit: change material or shape.")
    return ChallengeGrade(passed, round(min(score, 100.0), 1), checks, met, feedback, ref)


def utilisation(c: ConstraintCheck) -> float:
    """Demand/capacity ratio of a constraint: <= 1 passes."""
    if c.sense == "<=":
        return c.value / c.limit if c.limit else math.inf
    return c.limit / c.value if c.value else math.inf


def challenge_report(challenge: Challenge, grade: ChallengeGrade) -> ExperimentResult:
    """Package a graded challenge as an ExperimentResult (for report.lab_report_pdf and CSV export)."""
    rows = []
    for c in grade.checks:
        mm_ = c.name.endswith("(m)")
        f = 1e3 if mm_ else 1.0
        rows.append({"constraint": c.name.removesuffix(" (m)").removesuffix(" (kg)"), "value": c.value * f,
                     "limit": c.limit * f, "unit": "mm" if mm_ else ("kg" if c.name.endswith("(kg)") else "-"),
                     "sense": c.sense, "result": "PASS" if c.satisfied else "FAIL", "margin_pct": c.margin_pct,
                     "utilisation": utilisation(c)})
    m = grade.metrics
    ref = grade.reference
    concl = (f"{'PASSED' if grade.passed else 'FAILED'} with score {grade.score:.1f}/100. Design: {m.shape} "
             f"{', '.join(f'{k} = {fmt(v, 'mm')}' for k, v in m.dims.items())}, {m.material}; δ = "
             f"{fmt(m.delta_max, 'mm')}, σ = {fmt(m.sigma_max, 'MPa')}, FOS = {m.fos:.2f}, mass = {m.mass:.3g} kg. ")
    if ref is not None and ref.optimized is not None:
        o = ref.optimized
        concl += (f"Reference optimum (SLSQP, re-verified): {', '.join(f'{k} = {fmt(v, 'mm')}' for k, v in o.dims.items())}, "
                  f"mass {o.mass:.3g} kg. ")
    concl += " ".join(grade.feedback)
    design = challenge_design(challenge, BeamModel(1.0, Section(m.shape, **m.dims), challenge.base_model.material))
    ic = describe(challenge.base_model)
    ic["Submitted section"] = describe(design)["Section"]
    ic["Submitted material"] = m.material
    return ExperimentResult(
        f"challenge_{challenge.key}", f"Design challenge: {challenge.title}", challenge.brief,
        SCORE_RULE, [r"\delta_{max} \le \delta_{allow}", r"FOS = \frac{\sigma_y}{|\sigma|_{max}} \ge FOS_{min}",
                     r"m = \rho A L"],
        "Submitted design", pd.DataFrame(rows), concl, x="constraint", y=("utilisation",), plot="bar",
        initial_conditions=ic,
        parameter_changes={"Section": ic["Submitted section"], "Material": m.material},
        findings={"score": grade.score, "mass_kg": m.mass, "delta_max_m": m.delta_max, "fos": m.fos})


# --------------------------------------------------------------------------- sensitivity
OUTPUTS = {"delta_max": "Max deflection", "sigma_max": "Max stress", "stiffness": "Stiffness k",
           "mass": "Mass", "fos": "Yield FOS"}


def _outputs(model: BeamModel) -> dict[str, float]:
    r = run_case(model)
    return {"delta_max": r["delta_max_m"], "sigma_max": r["sigma_max_Pa"], "stiffness": r["stiffness_N_per_m"],
            "mass": r["mass_kg"], "fos": r["fos"]}


def sensitivity_sweep(model: BeamModel, param: str, factors=np.linspace(0.5, 2.0, 16)) -> pd.DataFrame:
    """Columns: factor, value, delta_max, sigma_max, stiffness, mass, fos.  value = the parameter's SI value.
    Factors giving an invalid model (e.g. hollow d >= D) are skipped."""
    from smartbeam.model import ModelError
    base_val = _param_value(model, param)
    rows = []
    for f in np.asarray(factors, float):
        try:
            out = _outputs(scale_parameter(model, param, float(f)))
        except ModelError:
            continue
        rows.append({"factor": float(f), "value": base_val * f, **out})
    return pd.DataFrame(rows, columns=["factor", "value", *OUTPUTS])


def _param_value(model: BeamModel, param: str) -> float:
    if param == "L":
        return model.L
    if param == "E":
        return model.material.E
    if param == "load":
        return total_load(model)
    return getattr(model.section, param)


def sensitivity_params(model: BeamModel) -> tuple[str, ...]:
    return ("L", "E", "load", *model.section.dims())


def tornado(model: BeamModel, pct: float = 10.0, params=("L", "E", "load", "b", "h")) -> pd.DataFrame:
    """Columns: param, output, low_pct, high_pct (% change of output for -pct / +pct parameter change),
    elasticity = normalized sensitivity (dY/Y)/(dX/X), taken as ln(Y+/Y-)/ln(X+/X-): exactly n for Y ∝ Xⁿ.
    A parameter whose ±pct change gives an invalid model (e.g. hollow d >= D) is left out. Params the section lacks are skipped."""
    from smartbeam.model import ModelError
    base = _outputs(model)
    f = pct / 100
    rows = []
    for p in params:
        if p in ("b", "h", "d", "D") and p not in model.section.dims():
            continue
        try:
            lo, hi = _outputs(scale_parameter(model, p, 1 - f)), _outputs(scale_parameter(model, p, 1 + f))
        except ModelError:  # e.g. hollow D - pct% <= d: this parameter cannot be varied that far
            continue
        for k in OUTPUTS:
            b = base[k]
            if not (np.isfinite(b) and b != 0):
                continue
            el = (math.log(hi[k] / lo[k]) / math.log((1 + f) / (1 - f)) if hi[k] / lo[k] > 0
                  else (hi[k] - lo[k]) / b / (2 * f))
            rows.append({"param": p, "output": k, "low_pct": (lo[k] / b - 1) * 100, "high_pct": (hi[k] / b - 1) * 100,
                         "elasticity": el})
    return pd.DataFrame(rows, columns=["param", "output", "low_pct", "high_pct", "elasticity"])


def sensitivity_report(model: BeamModel, tor: pd.DataFrame, pct: float) -> ExperimentResult:
    """Package a tornado study as an ExperimentResult (lab report / CSV)."""
    d = tor[tor.output == "delta_max"].assign(a=lambda t: t.elasticity.abs()).sort_values("a", ascending=False)
    s = tor[tor.output == "sigma_max"].assign(a=lambda t: t.elasticity.abs()).sort_values("a", ascending=False)
    concl = (f"±{pct:g}% one-at-a-time study. Deflection is most sensitive to "
             + ", ".join(f"{PARAM_LABELS[r.param]} (elasticity {r.elasticity:+.2f})" for r in d.head(3).itertuples())
             + "; stress to "
             + ", ".join(f"{PARAM_LABELS[r.param]} ({r.elasticity:+.2f})" for r in s.head(3).itertuples())
             + ". An elasticity of n means a 1% parameter change shifts the output by about n%.")
    return ExperimentResult(
        "sensitivity", "Structural sensitivity analysis",
        "Rank how strongly each geometric, material and load parameter drives deflection, stress, stiffness, mass and FOS.",
        "Each parameter is changed by ±p% while all others stay fixed, and the engine re-analyses the beam. The "
        "normalized sensitivity (elasticity) is the % output change per % parameter change; for a power law "
        "Y ∝ Xⁿ it equals the exponent n.",
        [r"S = \frac{\partial Y / Y}{\partial X / X} \approx \frac{\ln(Y^+/Y^-)}{\ln(X^+/X^-)}",
         r"\delta \propto \frac{W L^3}{E I}", r"\sigma = \frac{M y}{I}"],
        "All parameters ±%", tor.reset_index(drop=True), concl, x="param", y=("elasticity",), group="output",
        plot="bar", initial_conditions=describe(model),
        findings={f"{r.param}->{r.output}": float(r.elasticity) for r in tor.itertuples()},
        parameter_changes={"Change": f"±{pct:g}% one at a time",
                           "Parameters": ", ".join(PARAM_LABELS[p] for p in tor.param.unique())})


# --------------------------------------------------------------------------- analytical vs numerical lab
def numerical_comparison(model: BeamModel, n_list=(1, 2, 4, 8, 16, 32, 64), theory: str = "euler") -> pd.DataFrame:
    """FEM peaks per mesh vs the engine's analytical result (only when analyze() takes the analytical route).
    Columns: n_requested, n_elements, v_fem, M_fem, sigma_fem, M_fe, v_ref, M_ref, sigma_ref, err_v_pct, err_M_pct,
    err_sigma_pct, err_M_fe_pct (NaN when no independent analytical reference exists)."""
    from smartbeam.analytical import analyze
    from smartbeam.fem import convergence_study
    cs = convergence_study(model, n_list=tuple(n_list), theory=theory)
    df = pd.DataFrame({"n_requested": cs.n_requested or list(n_list), "n_elements": cs.n_elements,
                       "v_fem": cs.v_max, "M_fem": cs.M_max, "sigma_fem": cs.sigma_max, "M_fe": cs.M_fe_max})
    r = analyze(model)
    if r.method == "analytical":
        ref = {"v": abs(r.v_max.value), "M": abs(r.M_max.value), "sigma": abs(r.sigma_max.value)}
        for k, col in (("v", "v_fem"), ("M", "M_fem"), ("sigma", "sigma_fem")):
            df[f"{k}_ref"] = ref[k]
            df[f"err_{k}_pct"] = 100 * (df[col] - ref[k]).abs() / ref[k] if ref[k] else math.nan
        df["err_M_fe_pct"] = 100 * (df.M_fe - ref["M"]).abs() / ref["M"] if ref["M"] else math.nan
    else:
        for c in ("v_ref", "M_ref", "sigma_ref", "err_v_pct", "err_M_pct", "err_sigma_pct", "err_M_fe_pct"):
            df[c] = math.nan
    return df
