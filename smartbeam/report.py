"""PDF (ReportLab, Matplotlib PNG figures) and CSV export. OWNER: reports engineer.

Every number in a report comes from the result objects passed in (analytical / FEM / safety / optimization /
advisor). Sections whose inputs are None are omitted, never filled with placeholder data. Figures are drawn with
matplotlib.figure.Figure (no pyplot state, no global backend change) and embedded as PNGs.
"""
from __future__ import annotations

import datetime as _dt
import io
import math
import os
from dataclasses import dataclass
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from smartbeam import sections
from smartbeam.model import BeamModel, RESTRAINTS, check_stability, classify

# --------------------------------------------------------------------------- fonts & styles
# DejaVu ships with matplotlib and covers Greek, superscripts, the rupee sign and math symbols; ReportLab's
# built-in Helvetica is Latin-1 only (σ, δ, ₹ would print as boxes).
_TTF = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for _name, _file in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                     ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"), ("DejaVuMono", "DejaVuSansMono.ttf")):
    if _name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(_name, os.path.join(_TTF, _file)))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
                              boldItalic="DejaVu-Bold")

INK = colors.HexColor("#0A1B33")
MUTED = colors.HexColor("#1F4A80")
ACCENT = colors.HexColor("#2D6CDF")  # print-safe dark cyan
RULE = colors.HexColor("#DCC9A1")
ZEBRA = colors.HexColor("#F7F3E8")
PASS_C, FAIL_C, WARN_C = colors.HexColor("#2E7D4F"), colors.HexColor("#B4443A"), colors.HexColor("#9A7B3F")
STATUS_COLOR = {"SAFE": PASS_C, "WARNING": WARN_C, "YIELD PREDICTED": FAIL_C, "BUCKLING PREDICTED": FAIL_C,
                "OPTIMIZED": PASS_C, "INFEASIBLE": FAIL_C, "FAILED": FAIL_C, "PASS": PASS_C, "FAIL": FAIL_C}

_S = {
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=26, leading=32, textColor=INK),
    "tagline": ParagraphStyle("tagline", fontName="DejaVu", fontSize=12, leading=16, textColor=ACCENT),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=15, leading=20, textColor=INK, spaceBefore=10,
                         spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11.5, leading=15, textColor=ACCENT, spaceBefore=6,
                         spaceAfter=3),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=4),
    "small": ParagraphStyle("small", fontName="DejaVu", fontSize=8, leading=11, textColor=MUTED, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=8.5, leading=11, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="DejaVu-Bold", fontSize=8.5, leading=11, textColor=colors.white),
    "eq": ParagraphStyle("eq", fontName="DejaVuMono", fontSize=9.5, leading=13, textColor=INK, leftIndent=12),
    "caption": ParagraphStyle("caption", fontName="DejaVu-Oblique", fontSize=8, leading=10, textColor=MUTED,
                              alignment=TA_CENTER, spaceAfter=8),
}
CONTENT_W = A4[0] - 36 * mm

# matplotlib print style (light background, same accent family as the app)
_C = {"cyan": "#2D6CDF", "amber": "#B8975A", "violet": "#12355B", "green": "#2E7D4F", "red": "#E06A5A",
      "ink": "#0A1B33", "grid": "#EFE8D6", "muted": "#1F4A80"}
_RC = {"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#6B7F99", "axes.labelcolor": _C["ink"],
       "axes.grid": True, "grid.color": _C["grid"], "grid.linewidth": 0.6, "xtick.color": _C["muted"],
       "ytick.color": _C["muted"], "axes.spines.top": False, "axes.spines.right": False,
       "axes.titleweight": "bold", "axes.titlesize": 10, "legend.frameon": False, "legend.fontsize": 8}


# --------------------------------------------------------------------------- small helpers
def _g(v, sig: int = 4) -> str:
    """Number -> text with `sig` significant figures; handles None / inf / nan honestly."""
    if v is None:
        return "—"
    v = float(v)
    if math.isnan(v):
        return "—"
    if math.isinf(v):
        return "∞" if v > 0 else "−∞"
    if 10 ** sig <= abs(v) < 1e7:  # 13190 rather than 1.319e+04
        return f"{round(v, sig - 1 - int(math.floor(math.log10(abs(v))))):.0f}".replace("-", "−")
    return f"{v:.{sig}g}".replace("-", "−")


def _clean(v: float, ref: float) -> float:
    """Zero out floating-point noise: |v| below 1e-9 of the problem's force/moment scale."""
    return 0.0 if abs(v) <= 1e-9 * abs(ref) else v


def _p(text, style: str = "body") -> Paragraph:
    return Paragraph(escape(str(text)), _S[style])


def _table(rows: list[list], header: bool = True, widths: list[float] | None = None,
           status_col: int | None = None) -> Table:
    """Styled table. Cells are escaped text; `status_col` colours PASS/FAIL/SAFE... cells."""
    data = []
    for i, row in enumerate(rows):
        style = "cellh" if header and i == 0 else "cell"
        cells = []
        for j, c in enumerate(row):
            text = escape(str(c))
            col = STATUS_COLOR.get(str(c).upper()) if j == status_col and style == "cell" else None
            if col is not None:
                text = f'<font color="#{col.hexval()[2:]}"><b>{text}</b></font>'
            cells.append(Paragraph(text, _S[style]))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    ts = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        ts.append(("BACKGROUND", (0, 0), (-1, 0), INK))
    for i in range(1 if header else 0, len(rows)):
        if i % 2 == 0:
            ts.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(ts))
    return t


def _kv(rows: list[tuple[str, str]]) -> Table:
    return _table([list(r) for r in rows], header=False, widths=[0.34 * CONTENT_W, 0.66 * CONTENT_W])


def _png(fig, width: float = CONTENT_W) -> Image:
    """Matplotlib figure -> embedded PNG flowable scaled to `width` (aspect kept)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image(buf)
    scale = width / img.drawWidth
    img.drawWidth, img.drawHeight = width, img.drawHeight * scale
    return img


def _figure(nrows: int = 1, ncols: int = 1, size=(7.0, 2.6)):
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=size, layout="constrained")
        axes = fig.subplots(nrows, ncols, squeeze=False)
    return fig, axes.ravel()


def _style_axes(fig) -> None:
    for ax in fig.axes:  # rc_context only applies at creation; enforce the look for later artists
        ax.grid(True, color=_C["grid"], linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def _diagram(ax, x, y, color, ylabel, title, ext=None, ext_scale=1.0, ext_unit=""):
    ax.plot(x, y, color=color, linewidth=1.6)
    ax.fill_between(x, y, 0, color=color, alpha=0.15, linewidth=0)
    ax.axhline(0, color=_C["ink"], linewidth=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if ext is not None:
        val = ext.value * ext_scale
        ax.plot([ext.x], [val], "o", color=_C["amber"], markersize=5, zorder=5)
        right = ext.x > 0.5 * (x[0] + x[-1])
        ax.annotate(f"{_g(val)} {ext_unit} @ x = {_g(ext.x)} m", (ext.x, val), textcoords="offset points",
                    xytext=(-8 if right else 8, -12 if val >= 0 else 6), ha="right" if right else "left",
                    fontsize=8, color=_C["ink"])
        ax.margins(y=0.12)


def format_check(c) -> tuple[str, float, float, str]:
    """ConstraintCheck -> (name, value, limit, unit) for display: lengths in mm instead of m, others unchanged.

    Used by the PDF and by the Streamlit pages so constraint tables read the same everywhere."""
    if c.name.endswith("(m)"):
        return c.name[:-3].rstrip(), c.value * 1e3, c.limit * 1e3, "mm"
    return c.name, c.value, c.limit, ""


CHECK_WIDTHS = [0.34 * CONTENT_W, 0.17 * CONTENT_W, 0.19 * CONTENT_W, 0.15 * CONTENT_W, 0.15 * CONTENT_W]


def _check_rows(checks) -> list[list]:
    rows = [["Constraint", "Value", "Limit", "Margin", "Result"]]
    for c in checks:
        name, v, lim, unit = format_check(c)
        u = f" {unit}" if unit else ""
        rows.append([name, _g(v) + u, f"{c.sense} {_g(lim)}{u}", f"{_g(c.margin_pct, 3)} %",
                     "PASS" if c.satisfied else "FAIL"])
    return rows


def _dims_text(s) -> str:
    names = {"b": "b", "h": "h", "d": "d", "D": "D"}
    return ", ".join(f"{names[k]} = {_g(v * 1e3)} mm" for k, v in s.dims().items())


QUANTITY_NAMES = {"delta_max": "Max deflection |v|", "sigma_max": "Max normal stress |σ|",
                  "buckling": "Axial compression vs allowable (Pcr / FOS)", "mass": "Mass"}


def _quantity_text(quantity: str, value: float) -> str:
    """Advisor Recommendation.current / .limit in display units."""
    scale, unit = {"delta_max": (1e3, "mm"), "sigma_max": (1e-6, "MPa"), "buckling": (1e-3, "kN"),
                   "mass": (1.0, "kg")}.get(quantity, (1.0, ""))
    return f"{_g(value * scale)} {unit}".strip()


SHAPE_NAMES = {"rect": "Rectangular", "solid_circ": "Solid circular", "hollow_circ": "Hollow circular"}


# --------------------------------------------------------------------------- model description tables
def _load_rows(model: BeamModel) -> list[list]:
    rows = [["#", "Type", "Position", "Magnitude", "Direction / components"]]
    for i, ld in enumerate(model.loads, 1):
        if ld.kind == "point":
            rows.append([i, "Point force", f"x = {_g(ld.x)} m", f"{_g(ld.magnitude / 1e3)} kN",
                         f"θ = {_g(ld.angle_deg)}° from +x; Fx = {_g(_clean(ld.fx, ld.magnitude))} N, "
                         f"Fy = {_g(_clean(ld.fy, ld.magnitude))} N"])
        elif ld.kind == "distributed":
            kind = "UDL" if ld.w1 == ld.w2 else "Linearly varying"
            rows.append([i, f"Distributed ({kind})", f"x = {_g(ld.x1)} → {_g(ld.x2)} m",
                         f"{_g(ld.w1 / 1e3)} → {_g(ld.w2 / 1e3)} kN/m",
                         f"{ld.direction}; resultant {_g(ld.resultant)} N at x = {_g(ld.centroid)} m"])
        else:
            rows.append([i, "Applied moment", f"x = {_g(ld.x)} m", f"{_g(abs(ld.M))} N·m",
                         "counter-clockwise" if ld.M >= 0 else "clockwise"])
    return rows


def _support_rows(model: BeamModel) -> list[list]:
    rows = [["#", "Type", "Position", "Restrained DOFs"]]
    for i, s in enumerate(model.supports, 1):
        rows.append([i, s.type, f"x = {_g(s.x)} m", ", ".join(RESTRAINTS[s.type])])
    return rows


def _reaction_rows(reactions) -> list[list]:
    rows = [["Support", "x (m)", "Rx (N)", "Ry (N)", "M (N·m, CCW +)"]]
    ref = max([abs(r.Fx) + abs(r.Fy) for r in reactions] + [0.0])
    for r in reactions:
        rows.append([f"{r.support_index + 1} ({r.type})", _g(r.x), _g(_clean(r.Fx, ref)), _g(_clean(r.Fy, ref)),
                     _g(_clean(r.M, ref))])
    return rows


def _section_props_rows(model: BeamModel) -> list[tuple[str, str]]:
    p = sections.properties(model.section, model.material.nu)
    return [("Shape", SHAPE_NAMES[model.section.shape]), ("Dimensions", _dims_text(model.section)),
            ("Area A", f"{_g(p.A * 1e6)} mm²"), ("Second moment I", f"{_g(p.I * 1e12)} mm⁴"),
            ("Section modulus Z = I / (depth/2)", f"{_g(p.Z * 1e9)} mm³"),
            ("Radius of gyration r", f"{_g(p.r_g * 1e3)} mm"),
            ("Centroid height (from bottom fibre)", f"{_g(p.centroid * 1e3)} mm"),
            ("Extreme fibres y_top / y_bottom", f"{_g(p.y_top * 1e3)} / {_g(p.y_bottom * 1e3)} mm"),
            ("Shear correction factor κ", _g(p.kappa)), ("First moment Q at neutral axis", f"{_g(p.Q_max * 1e9)} mm³")]


def _material_rows(model: BeamModel) -> list[tuple[str, str]]:
    m = model.material
    rows = [("Name", m.name), ("Young's modulus E", f"{_g(m.E / 1e9)} GPa"),
            ("Shear modulus G = E / 2(1 + ν)", f"{_g(m.G / 1e9)} GPa"),
            ("Yield strength σy", f"{_g(m.yield_strength / 1e6)} MPa"),
            ("Ultimate tensile strength", f"{_g(m.uts / 1e6)} MPa" if m.uts else "not given"),
            ("Poisson's ratio ν", _g(m.nu)), ("Density ρ", f"{_g(m.density)} kg/m³"),
            ("Cost rate", f"₹ {_g(m.cost_per_kg)} /kg (rough estimate)" if m.cost_per_kg else "not given")]
    if m.approximate:
        rows.append(("Data quality", "Approximate reference values — verify against supplier / standard data."))
    return rows


# --------------------------------------------------------------------------- figures
def fig_sfd_bmd(result):
    fig, (a1, a2) = _figure(2, 1, (7.0, 4.6))
    _diagram(a1, result.x, result.V / 1e3, _C["cyan"], "V (kN)", "Shear force diagram", result.V_max, 1e-3, "kN")
    _diagram(a2, result.x, result.M / 1e3, _C["amber"], "M (kN·m)", "Bending moment diagram (sagging +)",
             result.M_max, 1e-3, "kN·m")
    _style_axes(fig)
    return fig


def fig_deflection(result):
    fig, (a1, a2) = _figure(2, 1, (7.0, 4.6))
    _diagram(a1, result.x, result.v * 1e3, _C["cyan"], "v (mm)", "Deflection v(x) (+ up)", result.v_max, 1e3, "mm")
    _diagram(a2, result.x, result.theta * 1e3, _C["violet"], "θ (mrad)", "Slope θ(x) (CCW +)", result.theta_max,
             1e3, "mrad")
    _style_axes(fig)
    return fig


def fig_stress(result):
    from smartbeam.analytical import stress_distribution  # lazy: analytical imports the engine stack
    fig, (a1, a2) = _figure(1, 2, (7.0, 2.9))
    a1.plot(result.x, result.sigma_top / 1e6, color=_C["red"], linewidth=1.5, label="top fibre")
    a1.plot(result.x, result.sigma_bottom / 1e6, color=_C["cyan"], linewidth=1.5, label="bottom fibre")
    a1.axhline(0, color=_C["ink"], linewidth=0.8)
    a1.set(xlabel="x (m)", ylabel="σ (MPa, tension +)")
    a1.set_title("Fibre stress along the beam", loc="left")
    a1.legend()
    i = int(np.argmax(np.abs(result.M)))
    y, sig = stress_distribution(result.model, float(result.M[i]), float(result.N[i]))
    a2.plot(sig / 1e6, y * 1e3, color=_C["amber"], linewidth=1.6)
    a2.fill_betweenx(y * 1e3, sig / 1e6, 0, color=_C["amber"], alpha=0.15, linewidth=0)
    a2.axvline(0, color=_C["ink"], linewidth=0.8)
    a2.set(xlabel="σ (MPa)", ylabel="y (mm from centroid)")
    a2.set_title(f"Across depth at x = {_g(result.x[i])} m", loc="left")
    _style_axes(fig)
    return fig


def fig_strain(result):
    fig, (ax,) = _figure(1, 1, (7.0, 2.5))
    ax.plot(result.x, result.eps_top * 1e6, color=_C["red"], linewidth=1.5, label="top fibre")
    ax.plot(result.x, result.eps_bottom * 1e6, color=_C["cyan"], linewidth=1.5, label="bottom fibre")
    ax.axhline(0, color=_C["ink"], linewidth=0.8)
    ax.set(xlabel="x (m)", ylabel="ε (µε)")
    ax.set_title("Normal strain ε = σ / E", loc="left")
    ax.legend()
    _style_axes(fig)
    return fig


def fig_fem_vs_analytical(analysis, fem):
    fig, (ax,) = _figure(1, 1, (7.0, 2.6))
    ax.plot(analysis.x, analysis.v * 1e3, color=_C["cyan"], linewidth=1.6, label="Analytical (statics + integration)")
    ax.plot(fem.mesh.nodes, fem.v_nodes * 1e3, "o", color=_C["amber"], markersize=3.5,
            label=f"FEM nodes ({fem.mesh.n_elements} elements)")
    ax.set(xlabel="x (m)", ylabel="v (mm)")
    ax.set_title("Deflection: analytical vs FEM", loc="left")
    ax.legend()
    _style_axes(fig)
    return fig


def fig_convergence(conv):
    fig, (ax,) = _figure(1, 1, (7.0, 2.8))
    n = np.asarray(conv.n_elements, float)
    if conv.error_pct:
        names = {"v_max": "|v|max", "theta_max": "|θ|max", "M_max": "|M|max", "sigma_max": "|σ|max",
                 "M_fe": "|EI v″|max (raw FE field)"}
        for (k, label), c in zip(names.items(), (_C["cyan"], _C["violet"], _C["amber"], _C["red"], _C["green"])):
            if k in conv.error_pct:
                err = np.maximum(np.asarray(conv.error_pct[k], float), 1e-14)  # log axis: floor at round-off
                ax.loglog(n, err, "o-", color=c, linewidth=1.3, markersize=3.5, label=label)
        ax.set_ylabel("error vs closed form (%)")
        ax.set_title("Mesh convergence (error floored at 1e-14 % for the log axis)", loc="left")
    else:
        ref = conv.v_max[-1]
        ax.semilogx(n, conv.v_max / ref, "o-", color=_C["cyan"], label="|v|max / finest mesh")
        ax.semilogx(n, conv.M_max / conv.M_max[-1], "o-", color=_C["amber"], label="|M|max / finest mesh")
        ax.set_ylabel("normalised value")
        ax.set_title("Mesh convergence (no closed form: normalised to the finest mesh)", loc="left")
    ax.set_xlabel("number of elements")
    ax.legend()
    _style_axes(fig)
    return fig


# --------------------------------------------------------------------------- PDF scaffolding
class _Doc:
    """Numbered sections + flowable list."""

    def __init__(self):
        self.story: list = []
        self.n = 0
        self.toc: list[str] = []

    def section(self, title: str, new_page: bool = False):
        if new_page:
            self.story.append(PageBreak())
        self.n += 1
        self.toc.append(f"{self.n}. {title}")
        self.story.append(Paragraph(f"{self.n}. {escape(title)}", _S["h1"]))

    def add(self, *flowables):
        self.story.extend(flowables)

    def fig(self, fig, caption: str):
        self.story.append(KeepTogether([_png(fig), Paragraph(escape(caption), _S["caption"])]))


def _render(story, title: str) -> bytes:
    buf = io.BytesIO()

    def later(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, A4[1] - 12 * mm, title)
        canvas.drawRightString(A4[0] - 18 * mm, A4[1] - 12 * mm, "SmartBeam 2.0")
        canvas.setStrokeColor(RULE)
        canvas.line(18 * mm, A4[1] - 13.5 * mm, A4[0] - 18 * mm, A4[1] - 13.5 * mm)
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm,
                            bottomMargin=18 * mm, title=title, author="SmartBeam 2.0")
    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=later)
    return buf.getvalue()


def _title_page(meta: "ReportMeta", subtitle: str, summary: list[tuple[str, str]], toc: list[str]) -> list:
    out = [Spacer(1, 14 * mm), Paragraph("SMARTBEAM 2.0", _S["tagline"]), Spacer(1, 3 * mm),
           Paragraph(escape(meta.title), _S["title"]), Spacer(1, 2 * mm),
           Paragraph("Understand. Simulate. Predict. Optimize.", _S["tagline"]), Spacer(1, 6 * mm),
           _p(subtitle), Spacer(1, 8 * mm),
           _kv([("Project", meta.project or "—"), ("Team", meta.team), ("Institute", meta.institute),
                ("Guide", meta.guide), ("Date", meta.date or _dt.date.today().isoformat())])]
    if summary:
        out += [Spacer(1, 6 * mm), Paragraph("Key results", _S["h2"]), _kv(summary)]
    if toc:
        half = (len(toc) + 1) // 2
        col = [[_p(t, "small") for t in toc[:half]], [_p(t, "small") for t in toc[half:]]]
        grid = Table([[col[0], col[1]]], colWidths=[CONTENT_W / 2] * 2, hAlign="LEFT")
        grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        out += [Spacer(1, 6 * mm), Paragraph("Contents", _S["h2"]), grid]
    out.append(PageBreak())
    return out


# --------------------------------------------------------------------------- public API
@dataclass
class ReportMeta:
    title: str = "SmartBeam 2.0 Engineering Report"
    project: str = ""
    team: str = "[Team members]"  # editable placeholders, never invented
    institute: str = "[Institute]"
    guide: str = "[Guide]"
    date: str = ""


GENERAL_ASSUMPTIONS = [
    "Linear elastic, homogeneous, isotropic material; small deflections and small rotations (first-order theory).",
    "Euler–Bernoulli beam theory for the analytical route (plane sections remain plane and normal to the axis; "
    "shear deformation neglected). Accuracy falls for short/deep beams (span/depth below about 10).",
    "Prismatic member; bending about the horizontal principal axis; loads act through the shear centre "
    "(no torsion, no lateral–torsional buckling, no local buckling of thin walls).",
    "Supports are ideal (fixed: u, v, θ; pin: u, v; roller: v). Self-weight is not included unless entered as a "
    "distributed load.",
    "FEM uses 1D two-node line elements only (Euler–Bernoulli / Timoshenko beam, planar frame). It is not a 2D/3D "
    "continuum stress analysis; stress concentrations at supports and load points are not captured.",
    "Failure checks predict the onset of yielding and ideal elastic (Euler) buckling. Yield is not fracture: "
    "ultimate/fracture loads are not predicted.",
    "Material properties and ₹/kg cost rates are approximate reference values and must be verified for design use.",
]


def build_pdf(model: BeamModel, analysis=None, fem=None, convergence=None, safety=None, optimization=None,
              advisor=None, meta: ReportMeta | None = None) -> bytes:
    """Full engineering report; sections whose inputs are None are omitted (never filled with fake data).

    analysis: analytical.AnalysisResult · fem: fem.FEMResult · convergence: fem.ConvergenceResult ·
    safety: safety.SafetyReport · optimization: optimize.OptimizationResult · advisor: advisor.AdvisorReport."""
    meta = meta or ReportMeta(project=model.name)
    d = _Doc()
    stab = analysis.stability if analysis is not None else check_stability(model)

    # ---- model description
    d.section("Geometry and configuration")
    d.add(_kv([("Model name", model.name or "Untitled"), ("Span L", f"{_g(model.L)} m"),
               ("Configuration", classify(model)), ("Cross-section", f"{SHAPE_NAMES[model.section.shape]}, "
                                                                     f"{_dims_text(model.section)}"),
               ("Statics", stab.message),
               ("Analysis route", analysis.method if analysis is not None else "not analysed")]))
    d.add(_p("Sign convention: x from the left end, y up. Shear V = ΣFy to the left of the cut (dV/dx = q); "
             "bending moment sagging-positive (dM/dx = V); axial force tension-positive; deflection v up; slope "
             "θ counter-clockwise; σ = N/A − M·y/I, so sagging puts the bottom fibre in tension.", "small"))
    d.section("Material")
    d.add(_kv(_material_rows(model)))
    d.section("Loading")
    d.add(_table(_load_rows(model), widths=[8 * mm, 32 * mm, 32 * mm, 32 * mm, CONTENT_W - 104 * mm])
          if model.loads else _p("No loads applied."))
    d.section("Supports")
    d.add(_table(_support_rows(model), widths=[8 * mm, 30 * mm, 35 * mm, CONTENT_W - 73 * mm]),
          _p(f"Stability check ({stab.mode}): {stab.n_restraints} restraints, rank {stab.rank} of {stab.required} "
             f"required → {'stable' if stab.stable else 'UNSTABLE'}, "
             f"{'statically determinate' if stab.determinate else f'indeterminate to degree {stab.degree}'}.",
             "small"))
    d.section("Section properties")
    d.add(_kv(_section_props_rows(model)))

    summary: list[tuple[str, str]] = []
    if analysis is not None:
        r = analysis
        summary = [("Analysis route", r.method), ("Max |deflection|", f"{_g(abs(r.v_max.value) * 1e3)} mm"),
                   ("Max |bending moment|", f"{_g(abs(r.M_max.value))} N·m"),
                   ("Max |normal stress|", f"{_g(abs(r.sigma_max.value) / 1e6)} MPa")]
        d.section("Support reactions and equilibrium")
        d.add(_table(_reaction_rows(r.reactions)))
        eq = ", ".join(f"{k} = {_g(v, 3)}" for k, v in r.equilibrium.items())
        d.add(_p(f"Equilibrium residuals (loads + reactions; ≈ 0 means equilibrium is satisfied): {eq}.", "small"))

        d.section("Shear force and bending moment diagrams", new_page=True)
        d.fig(fig_sfd_bmd(r), "SFD and BMD from the engine (x repeated at concentrated loads so jumps are vertical).")
        d.add(_kv([("Max |V|", f"{_g(r.V_max.value)} N at x = {_g(r.V_max.x)} m"),
                   ("Max |M|", f"{_g(r.M_max.value)} N·m at x = {_g(r.M_max.x)} m"),
                   ("Max |N|", f"{_g(r.N_max.value)} N at x = {_g(r.N_max.x)} m"
                    if abs(r.N_max.value) > 1e-9 * abs(r.V_max.value) else "0 (no axial force)")]))

        d.section("Bending stress", new_page=True)
        d.fig(fig_stress(r), "Normal stress σ = N/A − M·y/I along both extreme fibres, and across the depth at "
                             "the section of maximum |M|.")
        d.add(_kv([("Max |σ|", f"{_g(r.sigma_max.value / 1e6)} MPa at x = {_g(r.sigma_max.x)} m "
                               f"({r.sigma_max.where} fibre)"),
                   ("Max tension", f"{_g(r.sigma_tension_max.value / 1e6)} MPa at x = {_g(r.sigma_tension_max.x)} m"),
                   ("Max compression",
                    f"{_g(r.sigma_compression_max.value / 1e6)} MPa at x = {_g(r.sigma_compression_max.x)} m")]))

        d.section("Strain")
        d.fig(fig_strain(r), "Extreme-fibre normal strain.")
        eps = max(np.max(np.abs(r.eps_top)), np.max(np.abs(r.eps_bottom)))
        d.add(_p(f"Max |ε| = {_g(eps * 1e6)} µε (= |σ|max / E)."))

        d.section("Slope and deflection", new_page=True)
        d.fig(fig_deflection(r), "Deflection and slope from EI v″ = M with the support boundary conditions.")
        d.add(_kv([("Max |v|", f"{_g(r.v_max.value * 1e3)} mm at x = {_g(r.v_max.x)} m"),
                   ("Max |θ|", f"{_g(r.theta_max.value * 1e3)} mrad at x = {_g(r.theta_max.x)} m")]))

        if r.textbook is not None:
            tb = r.textbook
            d.section("Textbook verification")
            d.add(_p(f"The model is exactly the textbook case '{tb.case}', so closed-form results apply."))
            rows = [["Quantity", "Formula", "Closed form", "Engine", "Rel. error"]]
            for k, f in tb.formulas.items():
                rows.append([k, f, _g(tb.textbook.get(k)), _g(tb.numerical.get(k)), _g(tb.rel_error.get(k), 2)])
            d.add(_table(rows), _p("Values in SI units (N, N·m, rad, m, Pa).", "small"))

    if fem is not None:
        d.section("Finite element results", new_page=True)
        d.add(_kv([("Element", f"{fem.element} ({fem.theory}), {fem.dofs_per_node} DOF per node"),
                   ("Mesh", f"{fem.mesh.n_elements} elements, {fem.mesh.n_nodes} nodes"),
                   ("Degrees of freedom", f"{fem.n_dof} total, {fem.n_free} free"),
                   ("Condition number (scaled K_ff)", _g(fem.condition_number, 3)),
                   ("Max |v|", f"{_g(fem.v_max.value * 1e3)} mm at x = {_g(fem.v_max.x)} m"),
                   ("Max |M|", f"{_g(fem.M_max.value)} N·m at x = {_g(fem.M_max.x)} m"),
                   ("Max |σ|", f"{_g(fem.sigma_max.value / 1e6)} MPa at x = {_g(fem.sigma_max.x)} m")]))
        d.add(Paragraph("FEM reactions", _S["h2"]), _table(_reaction_rows(fem.reactions)))
        step = max(1, math.ceil(fem.mesh.n_nodes / 25))
        rows = [["Node", "x (m)", "u (mm)", "v (mm)", "θ (mrad)"]]
        for i in range(0, fem.mesh.n_nodes, step):
            rows.append([i, _g(fem.mesh.nodes[i]), _g(fem.u_nodes[i] * 1e3), _g(fem.v_nodes[i] * 1e3),
                         _g(fem.theta_nodes[i] * 1e3)])
        d.add(Paragraph("Nodal displacements" + (f" (every {step}th node; full table in the CSV)" if step > 1 else ""),
                        _S["h2"]), _table(rows))

        if analysis is not None and analysis.method == "analytical":
            d.section("Analytical vs numerical")
            rows = [["Quantity", "Analytical", "FEM", "Difference"]]
            for label, a, f, scale, unit in (
                    ("Max |v|", analysis.v_max.value, fem.v_max.value, 1e3, "mm"),
                    ("Max |θ|", analysis.theta_max.value, fem.theta_max.value, 1e3, "mrad"),
                    ("Max |M|", analysis.M_max.value, fem.M_max.value, 1.0, "N·m"),
                    ("Max |V|", analysis.V_max.value, fem.V_max.value, 1.0, "N"),
                    ("Max |σ|", analysis.sigma_max.value, fem.sigma_max.value, 1e-6, "MPa")):
                diff = abs(abs(f) - abs(a)) / abs(a) * 100 if a else math.nan
                rows.append([label, f"{_g(abs(a) * scale)} {unit}", f"{_g(abs(f) * scale)} {unit}", f"{_g(diff, 3)} %"])
            d.add(_table(rows))
            d.fig(fig_fem_vs_analytical(analysis, fem), "Analytical deflection curve with FEM nodal values.")
            d.add(_p("Differences are reported as computed; for Euler–Bernoulli elements with consistent loads, "
                     "nodal values of a prismatic beam are exact, so differences sit at round-off level.", "small"))
        elif analysis is not None:
            d.add(_p(f"No independent analytical comparison: the structure was solved by the '{analysis.method}' "
                     "route because statics alone cannot solve it (indeterminate or axially loaded).", "small"))

    if convergence is not None:
        c = convergence
        d.section("Mesh convergence study")
        rows = [["Elements", "|v|max (mm)", "|θ|max (mrad)", "|M|max (N·m)", "|σ|max (MPa)"]
                + (["v error (%)", "M error (%)"] if c.error_pct else [])]
        for i, n in enumerate(c.n_elements):
            row = [n, _g(c.v_max[i] * 1e3, 6), _g(c.theta_max[i] * 1e3, 6), _g(c.M_max[i], 6),
                   _g(c.sigma_max[i] / 1e6, 6)]
            if c.error_pct:
                row += [_g(c.error_pct["v_max"][i], 2), _g(c.error_pct["M_max"][i], 2)]
            rows.append(row)
        d.add(_table(rows))
        d.fig(fig_convergence(c), "Convergence with mesh refinement.")
        d.add(_p(c.note, "small"))

    if safety is not None:
        s = safety
        y = s.yield_result
        d.section("Failure prediction and factor of safety", new_page=True)
        d.add(_table([["Check", "Result", "FOS", "Status"],
                      ["Yield (max |σ|)", f"{_g(abs(y.sigma_max.value) / 1e6)} MPa vs σy {_g(y.sigma_y / 1e6)} MPa",
                       _g(y.fos, 3), y.status],
                      ["von Mises (σ + τ)", f"{_g(s.combined.von_mises / 1e6)} MPa at x = {_g(s.combined.x)} m",
                       _g(s.combined.fos, 3), "—"],
                      ["Euler buckling",
                       (f"Pcr = {_g(s.buckling.Pcr / 1e3)} kN vs {_g(s.buckling.P_applied / 1e3)} kN "
                        f"({s.buckling.end_condition})") if s.buckling.applicable else "not applicable",
                       _g(s.buckling.fos, 3) if s.buckling.applicable else "—", "—"],
                      ["Overall", f"minimum required FOS {_g(y.min_fos)}", "", s.status]], status_col=3))
        d.add(_p(f"First-yield load factor λ = {_g(y.first_yield_load_factor, 3)} (linear elastic: all loads × λ "
                 "brings the most stressed fibre to yield)."))
        for msg in s.messages:
            d.add(_p(msg, "small"))
        d.add(Paragraph("von Mises assumptions", _S["h2"]))
        for a in s.combined.assumptions:
            d.add(_p(f"• {a}", "small"))

    if optimization is not None:
        o = optimization
        d.section("Design optimization", new_page=True)
        # the optimizer's message ends with the reasons; they are listed individually below
        d.add(_p(o.message.replace(" ".join(o.infeasibility_reasons), "").strip()))
        rows = [["Quantity", "Initial", "Optimized"]]
        opt = o.optimized
        rows.append(["Dimensions", ", ".join(f"{k} = {_g(v * 1e3)} mm" for k, v in o.initial_dims.items()),
                     ", ".join(f"{k} = {_g(v * 1e3)} mm" for k, v in o.optimal_dims.items()) if o.optimal_dims else "—"])
        for label, attr, scale, unit in (("Volume", "volume", 1e6, "cm³"), ("Mass", "mass", 1, "kg"),
                                         ("Cost", "cost", 1, "₹"), ("Max |v|", "delta_max", 1e3, "mm"),
                                         ("Max |σ|", "sigma_max", 1e-6, "MPa"), ("Yield FOS", "fos", 1, "")):
            iv = getattr(o.initial, attr)
            ov = getattr(opt, attr) if opt else None
            cell = (lambda v: f"₹ {_g(v)}" if v is not None else "—") if unit == "₹" else \
                (lambda v: f"{_g(v * scale)} {unit}".strip() if v is not None else "—")
            rows.append([label, cell(iv), cell(ov)])
        if o.reduction_pct is not None:
            rows.append(["Objective reduction", "", f"{_g(o.reduction_pct, 3)} %"])
        d.add(_table(rows))
        if o.checks:
            d.add(Paragraph("Independent constraint verification" if o.feasible
                            else "Constraints at the strongest section within the bounds", _S["h2"]),
                  _table(_check_rows(o.checks), widths=CHECK_WIDTHS, status_col=4))
        for reason in o.infeasibility_reasons:
            d.add(_p(f"• {reason}"))

    if advisor is not None:
        a = advisor
        d.section("Design recommendations", new_page=True)
        d.add(_p(f"Limiting condition: {a.limiting}. Utilisation (demand / capacity, > 1 means violated): "
                 + ", ".join(f"{k} {_g(v, 3)}" for k, v in a.utilisation.items()) + "."))
        if a.checks:
            d.add(_table(_check_rows(a.checks), widths=CHECK_WIDTHS, status_col=4))
        if not a.recommendations:
            d.add(_p("No changes recommended: every criterion passes without large excess capacity."))
        for i, rec in enumerate(a.recommendations, 1):
            d.add(Paragraph(f"Recommendation {i}", _S["h2"]))
            d.add(_kv([("What is wrong?", rec.issue),
                       ("Which quantity governs?", f"{QUANTITY_NAMES.get(rec.quantity, rec.quantity)}: current "
                                                   f"{_quantity_text(rec.quantity, rec.current)}, limit "
                                                   f"{_quantity_text(rec.quantity, rec.limit)}"),
                       ("Recommended modification", rec.action),
                       ("What changes after it?",
                        f"|v|max {_g(rec.before.delta_max * 1e3)} → {_g(rec.after.delta_max * 1e3)} mm; "
                        f"|σ|max {_g(rec.before.sigma_max / 1e6)} → {_g(rec.after.sigma_max / 1e6)} MPa; "
                        f"FOS {_g(rec.before.fos, 3)} → {_g(rec.after.fos, 3)}; "
                        f"mass {_g(rec.before.mass)} → {_g(rec.after.mass)} kg (re-analysed)"),
                       ("Satisfies all constraints?", "YES" if rec.satisfies_all else "NO")]))

    d.section("Assumptions and limitations", new_page=d.n > 0)
    for text in GENERAL_ASSUMPTIONS:
        d.add(_p(f"• {text}"))
    warns = list(analysis.warnings if analysis is not None else []) + list(fem.warnings if fem is not None else [])
    if warns:
        d.add(Paragraph("Solver warnings for this model", _S["h2"]))
        for w in dict.fromkeys(warns):
            d.add(_p(f"• {w}", "small"))

    subtitle = (f"Analysis of '{model.name or 'Untitled'}': {classify(model)} beam, L = {_g(model.L)} m, "
                f"{SHAPE_NAMES[model.section.shape].lower()} section ({_dims_text(model.section)}), "
                f"{model.material.name}.")
    if safety is not None:
        summary.append(("Safety status", safety.status))
    if optimization is not None:
        summary.append(("Optimization", optimization.status))
    return _render(_title_page(meta, subtitle, summary, d.toc) + d.story, meta.title)


# --------------------------------------------------------------------------- CSV
def analysis_csv(result) -> str:
    """Columns: x_m, V_N, M_Nm, N_N, theta_rad, v_m, sigma_top_Pa, sigma_bottom_Pa, eps_top, eps_bottom."""
    return pd.DataFrame({
        "x_m": result.x, "V_N": result.V, "M_Nm": result.M, "N_N": result.N, "theta_rad": result.theta,
        "v_m": result.v, "sigma_top_Pa": result.sigma_top, "sigma_bottom_Pa": result.sigma_bottom,
        "eps_top": result.eps_top, "eps_bottom": result.eps_bottom,
    }).to_csv(index=False)


def fem_csv(result) -> str:
    """Nodal table: node, x_m, u_m, v_m, theta_rad (+ reactions where restrained)."""
    nodes = result.mesh.nodes
    df = pd.DataFrame({"node": np.arange(len(nodes)), "x_m": nodes, "u_m": result.u_nodes, "v_m": result.v_nodes,
                       "theta_rad": result.theta_nodes,
                       "Rx_N": np.nan, "Ry_N": np.nan, "M_Nm": np.nan, "support": ""})
    for r in result.reactions:
        i = int(np.argmin(np.abs(nodes - r.x)))
        df.loc[i, ["Rx_N", "Ry_N", "M_Nm"]] = [r.Fx, r.Fy, r.M]
        df.loc[i, "support"] = r.type
    return df.to_csv(index=False)


def summary_csv(model: BeamModel, result, safety=None) -> str:
    """Key results as quantity,value,unit rows (SI), for spreadsheets."""
    p = sections.properties(model.section, model.material.nu)
    rows = [("name", model.name, ""), ("method", result.method, ""), ("L", model.L, "m"),
            ("shape", model.section.shape, "")]
    rows += [(k, v, "m") for k, v in model.section.dims().items()]
    rows += [("material", model.material.name, ""), ("E", model.material.E, "Pa"),
             ("yield_strength", model.material.yield_strength, "Pa"), ("A", p.A, "m^2"), ("I", p.I, "m^4"),
             ("Z", p.Z, "m^3")]
    for r in result.reactions:
        tag = f"support{r.support_index + 1}_{r.type}_x{r.x:g}"
        rows += [(f"{tag}_Rx", r.Fx, "N"), (f"{tag}_Ry", r.Fy, "N"), (f"{tag}_M", r.M, "N*m")]
    for label, ext, unit in (("V_max", result.V_max, "N"), ("M_max", result.M_max, "N*m"),
                             ("N_max", result.N_max, "N"), ("theta_max", result.theta_max, "rad"),
                             ("v_max", result.v_max, "m"), ("sigma_max", result.sigma_max, "Pa"),
                             ("sigma_tension_max", result.sigma_tension_max, "Pa"),
                             ("sigma_compression_max", result.sigma_compression_max, "Pa")):
        rows += [(label, ext.value, unit), (f"{label}_x", ext.x, "m")]
    rows += [(f"equilibrium_{k}", v, "") for k, v in result.equilibrium.items()]
    if safety is not None:
        rows += [("yield_fos", safety.yield_result.fos, ""), ("von_mises_fos", safety.combined.fos, ""),
                 ("buckling_fos", safety.buckling.fos if safety.buckling.applicable else None, ""),
                 ("safety_status", safety.status, "")]
    return pd.DataFrame(rows, columns=["quantity", "value", "unit"]).to_csv(index=False)


# --------------------------------------------------------------------------- lab reports
# mathtext lacks these LaTeX commands; map them to the closest supported ones
_LATEX_FIXES = (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\text{", "\\mathrm{"))


def _equation(eq: str):
    """LaTeX-style equation -> rendered PNG (matplotlib mathtext); plain text -> monospace paragraph."""
    if "\\" not in eq and "$" not in eq:
        return Paragraph(escape(eq), _S["eq"])
    tex = eq if eq.strip().startswith("$") else f"${eq.strip()}$"
    for a, b in _LATEX_FIXES:
        tex = tex.replace(a, b)
    try:
        fig = Figure(figsize=(0.1, 0.1))
        fig.text(0, 0, tex, fontsize=13, color=_C["ink"])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    except Exception:  # mathtext covers a LaTeX subset; anything else is shown verbatim rather than dropped
        return Paragraph(escape(eq), _S["eq"])
    buf.seek(0)
    img = Image(buf)
    w = min(img.drawWidth * 0.45, CONTENT_W - 12 * mm)
    img.drawHeight *= w / img.drawWidth
    img.drawWidth = w
    img.hAlign = "LEFT"
    return img


def _df_tables(df: pd.DataFrame, max_rows: int = 60, per_table: int = 7) -> list:
    """DataFrame -> tables of at most `per_table` columns (the first column repeats in each), so wide lab tables
    stay legible on A4 instead of shrinking to unreadable text."""
    df = df.head(max_rows)
    cols = list(df.columns)
    groups = [cols] if len(cols) <= per_table else \
        [[cols[0]] + cols[i:i + per_table - 1] for i in range(1, len(cols), per_table - 1)]
    out = []
    for grp in groups:
        rows = [[str(c).replace("_", "_\u200b") for c in grp]]  # zero-width space lets long names wrap
        for _, r in df[grp].iterrows():
            rows.append([_g(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else str(v)
                         for v in r.tolist()])
        out += [_table(rows), Spacer(1, 3 * mm)]
    return out


def build_lab_report(title: str, objective: str, initial_conditions: dict, equations: list[str],
                     parameter_changes, results_df: pd.DataFrame, figures: list, conclusions: list[str],
                     meta: ReportMeta | None = None) -> bytes:
    """Virtual-lab report PDF: objective, initial conditions, governing equations, parameter changes, results,
    graphs, conclusions.

    parameter_changes: str | list[str] | dict | DataFrame. figures: matplotlib Figures (pyplot or Figure objects;
    they are only saved, never closed). equations: LaTeX (rendered with matplotlib mathtext) or plain text."""
    meta = meta or ReportMeta(title=title)
    d = _Doc()
    d.section("Objective")
    d.add(_p(objective))
    d.section("Initial conditions")
    if initial_conditions:
        d.add(_kv([(str(k), str(v)) for k, v in initial_conditions.items()]))
    else:
        d.add(_p("None stated."))
    d.section("Governing equations")
    for eq in equations or []:
        d.add(_equation(eq), Spacer(1, 2 * mm))
    d.section("Parameter changes")
    if isinstance(parameter_changes, pd.DataFrame):
        d.add(*_df_tables(parameter_changes))
    elif isinstance(parameter_changes, dict):
        d.add(_kv([(str(k), str(v)) for k, v in parameter_changes.items()]))
    elif isinstance(parameter_changes, (list, tuple)):
        for item in parameter_changes:
            d.add(_p(f"• {item}"))
    else:
        d.add(_p(parameter_changes or "None."))
    d.section("Results")
    if results_df is not None and not results_df.empty:
        d.add(*_df_tables(results_df))
        if len(results_df) > 60:
            d.add(_p(f"First 60 of {len(results_df)} rows shown; the CSV has all of them.", "small"))
    for i, fig in enumerate(figures or [], 1):
        d.fig(fig, f"Figure {i}")
    d.section("Conclusions")
    for c in conclusions or []:
        d.add(_p(f"• {c}"))
    subtitle = f"Virtual laboratory report — {title}."
    return _render(_title_page(meta, subtitle, [], d.toc) + d.story, meta.title)


def lab_report_pdf(experiment_result, meta: ReportMeta | None = None) -> bytes:
    """Lab report: objective, initial conditions, governing equations, parameter changes, results, graphs, conclusion."""
    er = experiment_result
    df = er.table
    num = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    figs = []
    if len(df) > 1 and num and pd.api.types.is_numeric_dtype(df[df.columns[0]]):
        cols = num[:4]
        fig, axes = _figure(len(cols), 1, (7.0, 1.9 * len(cols) + 0.4))
        for ax, c, color in zip(axes, cols, (_C["cyan"], _C["amber"], _C["violet"], _C["green"])):
            ax.plot(df[df.columns[0]], df[c], "o-", color=color, linewidth=1.4, markersize=3.5)
            ax.set_ylabel(c)
        axes[-1].set_xlabel(str(df.columns[0]))
        _style_axes(fig)
        figs.append(fig)
    return build_lab_report(
        title=er.title, objective=er.objective,
        initial_conditions={"Experiment": er.title, "Swept parameter": er.parameter, "Explanation": er.explanation},
        equations=er.equations,
        parameter_changes=f"{er.parameter}: {len(df)} runs ({df.columns[0]} from {_g(df.iloc[0, 0])} to "
                          f"{_g(df.iloc[-1, 0])})." if len(df) and pd.api.types.is_numeric_dtype(df[df.columns[0]])
        else f"{er.parameter}: {len(df)} runs.",
        results_df=df, figures=figs, conclusions=[er.conclusion],
        meta=meta or ReportMeta(title=f"Lab report: {er.title}"))
