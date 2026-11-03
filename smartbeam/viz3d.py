"""Interactive 3D beam view (Plotly). OWNER: lead UI engineer.

Everything drawn comes from an AnalysisResult: the deformed shape is the computed v(x) (plus the
Euler-Bernoulli fibre shift -y*theta(x) and, on the frame route, the axial u(x)); surface colour is the
beam-theory normal stress sigma(x, y) = N/A - M*y/I across the depth. This is NOT a 3D continuum solution:
stress is uniform across the width and varies linearly through the depth by assumption.

Plot axes: X = beam axis x, Y = section width z, Z = vertical y (up).
No Streamlit imports here (engine-side module).
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from smartbeam import sections
from smartbeam.model import DistributedLoad, MomentLoad, PointLoad

DISCLAIMER = ("Beam-theory visualisation, not a 3D continuum (solid FEA) solution: the colour is "
              "σ = N/A − M·y/I, linear through the depth and uniform across the width; the deformed shape is "
              "the centreline deflection v(x) with plane sections rotated by θ(x), magnified by the "
              "amplification factor.")

CYAN, AMBER, MUTED, TEXT, GREEN, RED = "#22D3EE", "#F5A524", "#8FA0BA", "#E6EDF7", "#34D399", "#F87171"
STRESS_SCALE = [[0.0, "#2563EB"], [0.25, "#60A5FA"], [0.5, "#E6EDF7"], [0.75, "#F87171"], [1.0, "#DC2626"]]


def auto_amplification(result, target: float = 0.08) -> float:
    """Factor that makes the peak displayed deflection `target`*L (1 if the beam does not deflect)."""
    vmax = abs(result.v_max.value)
    if vmax <= 0:
        return 1.0
    amp = target * result.model.L / vmax
    return float(10 ** round(math.log10(amp), 1)) if amp > 1 else 1.0  # rounded to 1, 1.26, 1.58, ... x 10^k


def _perimeter(s, n_side: int = 9, n_circ: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Closed outer perimeter of the section as (y, z) arrays (section coordinates, from the centroid)."""
    if s.shape == "rect":
        hb, hh = s.b / 2, s.h / 2
        t = np.linspace(0, 1, n_side)[:-1]
        corners = [(hh, -hb), (hh, hb), (-hh, hb), (-hh, -hb), (hh, -hb)]
        ys, zs = [], []
        for (y0, z0), (y1, z1) in zip(corners[:-1], corners[1:]):
            ys.extend(y0 + (y1 - y0) * t)
            zs.extend(z0 + (z1 - z0) * t)
        ys.append(hh)
        zs.append(-hb)
        return np.array(ys), np.array(zs)
    R = s.depth / 2
    phi = np.linspace(0, 2 * np.pi, n_circ + 1)
    return R * np.sin(phi), R * np.cos(phi)


def _fields(result):
    """x, v, theta, u, sigma(y) callable from the result (u from the FEM frame solution when present)."""
    model = result.model
    s = model.section
    A, I = sections.area(s), sections.second_moment(s)
    x = np.asarray(result.x, float)
    u = np.zeros_like(x)
    f = result.fem
    if f is not None and getattr(f, "element", "") == "frame":
        u = np.interp(x, f.mesh.nodes, f.u_nodes)
    M, N = np.asarray(result.M, float), np.asarray(result.N, float)
    return x, np.asarray(result.v, float), np.asarray(result.theta, float), u, \
        (lambda y: (N[:, None] / A - M[:, None] * np.asarray(y)[None, :] / I))


def _surface(X, Y, Z, color, cmax, customdata, show_stress, name, opacity=1.0, showscale=False, hover=True):
    kw = dict(x=X, y=Y, z=Z, name=name, opacity=opacity, showlegend=False,
              lighting=dict(ambient=0.75, diffuse=0.6, specular=0.15, roughness=0.6), contours=dict())
    if show_stress:
        kw.update(surfacecolor=color, colorscale=STRESS_SCALE, cmin=-cmax, cmax=cmax, cmid=0.0,
                  showscale=showscale,
                  colorbar=dict(title=dict(text="σ (MPa)", side="right"), thickness=14, len=0.75, x=1.0,
                                tickfont=dict(color=MUTED)))
    else:
        kw.update(surfacecolor=np.zeros_like(X), colorscale=[[0, CYAN], [1, CYAN]], showscale=False)
    if hover:
        kw.update(customdata=customdata,
                  hovertemplate="x = %{customdata[0]:.4g} m<br>y = %{customdata[1]:.3g} mm<br>"
                                "σ = %{customdata[2]:.4g} MPa<br>v = %{customdata[3]:.4g} mm<extra></extra>")
    else:
        kw.update(hoverinfo="skip")
    return go.Surface(**kw)


def _beam_surfaces(result, amp, show_stress, cmax, opacity=1.0, hover=True, showscale=False):
    """Outer surface + two end caps of the (deformed by amp) beam."""
    s = result.model.section
    x, v, th, u, sigma_at = _fields(result)
    py, pz = _perimeter(s)
    sig = sigma_at(py) / 1e6  # (nx, np) MPa
    X = x[:, None] + amp * (u[:, None] - py[None, :] * th[:, None])
    Y = np.broadcast_to(pz[None, :], X.shape)
    Z = py[None, :] + amp * v[:, None]
    cd = np.stack([np.broadcast_to(x[:, None], X.shape), np.broadcast_to(py[None, :] * 1e3, X.shape), sig,
                   np.broadcast_to(v[:, None] * 1e3, X.shape)], axis=-1)
    out = [_surface(X, Y, Z, sig, cmax, cd, show_stress, "beam", opacity, showscale, hover)]

    inner = (s.d / s.D) if s.shape == "hollow_circ" else 0.0
    rr = np.linspace(inner, 1.0, 6)
    for i in (0, len(x) - 1):  # end caps, perimeter scaled towards the centroid (annulus for hollow)
        cy, cz = rr[:, None] * py[None, :], rr[:, None] * pz[None, :]
        csig = sigma_at(cy.ravel())[i].reshape(cy.shape) / 1e6
        CX = x[i] + amp * (u[i] - cy * th[i])
        CZ = cy + amp * v[i]
        ccd = np.stack([np.full(cy.shape, x[i]), cy * 1e3, csig, np.full(cy.shape, v[i] * 1e3)], axis=-1)
        out.append(_surface(CX, cz, CZ, csig, cmax, ccd, show_stress, "end", opacity, False, hover))
    return out


def _box(x0, x1, y0, y1, z0, z1, color, name, opacity=1.0):
    """Axis-aligned box as Mesh3d (plot axes X, Y(width), Z(up))."""
    X = [x0, x1, x1, x0, x0, x1, x1, x0]
    Y = [y0, y0, y1, y1, y0, y0, y1, y1]
    Z = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 4, 6, 7]
    return go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, color=color, opacity=opacity, name=name, flatshading=True,
                     hoverinfo="name", showlegend=False)


def _prism(x, ytop, size, half_w, color, name):
    """Triangular prism (pin symbol): apex at (x, ytop), base below, extruded across the width."""
    X = [x, x - 0.6 * size, x + 0.6 * size] * 2
    Y = [-half_w] * 3 + [half_w] * 3
    Z = [ytop, ytop - size, ytop - size] * 2
    i = [0, 3, 0, 0, 1, 1, 0, 0]
    j = [1, 4, 1, 4, 2, 5, 2, 5]
    k = [2, 5, 4, 3, 5, 4, 5, 3]
    return go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, color=color, flatshading=True, name=name,
                     hoverinfo="name", showlegend=False)


def _supports(model, glyph, half_w):
    s = model.section
    yb = -s.depth / 2
    traces = []
    for k, sp in enumerate(model.supports):
        label = f"Support {k + 1}: {sp.type} at x = {sp.x:.4g} m"
        if sp.type == "fixed":
            t = 0.12 * glyph
            x0, x1 = (sp.x - t, sp.x) if sp.x <= model.L / 2 else (sp.x, sp.x + t)
            H = s.depth / 2 + glyph
            traces.append(_box(x0, x1, -half_w * 1.4, half_w * 1.4, -H, H, MUTED, label))
        else:
            size = 0.8 * glyph if sp.type == "roller" else glyph
            traces.append(_prism(sp.x, yb, size, half_w, MUTED, label))
            if sp.type == "roller":
                n = 3
                xs = sp.x + np.linspace(-0.4, 0.4, n) * size
                traces.append(go.Scatter3d(x=xs, y=np.zeros(n), z=np.full(n, yb - size - 0.12 * size),
                                           mode="markers", marker=dict(size=5, color=MUTED, symbol="circle"),
                                           name=label, hoverinfo="name", showlegend=False))
            base = yb - size - (0.25 * size if sp.type == "roller" else 0.0)
            traces.append(_box(sp.x - 0.75 * size, sp.x + 0.75 * size, -half_w * 1.2, half_w * 1.2,
                               base - 0.08 * size, base, "#475569", label))
    return traces


def _arrow(tail, tip, head, label, hover):
    """Line + cone arrow in the x-y plane (Z up), Y = 0."""
    tail, tip = np.asarray(tail, float), np.asarray(tip, float)
    d = tip - tail
    n = np.linalg.norm(d)
    if n == 0:
        return []
    d /= n
    shaft_end = tip - d * head
    return [
        go.Scatter3d(x=[tail[0], shaft_end[0]], y=[0, 0], z=[tail[1], shaft_end[1]], mode="lines",
                     line=dict(color=AMBER, width=6), hoverinfo="text", hovertext=hover, showlegend=False),
        go.Cone(x=[tip[0]], y=[0], z=[tip[1]], u=[d[0]], v=[0], w=[d[1]], anchor="tip", sizemode="absolute",
                sizeref=head, colorscale=[[0, AMBER], [1, AMBER]], showscale=False, hoverinfo="text",
                hovertext=hover, name=label, showlegend=False),
    ]


def _fmt_force(F):
    return f"{F / 1e3:.3g} kN" if abs(F) >= 1e3 else f"{F:.3g} N"


def _loads(result, amp, glyph):
    """Load arrows attached to the displayed (deformed by amp) beam."""
    model = result.model
    s = model.section
    x, v, *_ = _fields(result)
    yc = lambda xx: amp * float(np.interp(xx, x, v))  # noqa: E731
    top, bot = s.depth / 2, -s.depth / 2
    La, head = 2.2 * glyph, 0.45 * glyph
    traces, labels = [], []

    for ld in model.point_loads:
        d = np.array([math.cos(math.radians(ld.angle_deg)), math.sin(math.radians(ld.angle_deg))])
        base = np.array([ld.x, yc(ld.x)])
        hover = f"Point load {_fmt_force(ld.magnitude)} at x = {ld.x:.4g} m, angle {ld.angle_deg:g}°"
        if abs(d[1]) >= abs(d[0]):  # mainly transverse: push on the surface
            tip = base + np.array([0, top if d[1] < 0 else bot])
            tail = tip - La * d
        else:  # mainly axial: pull away from / push into the nearer end
            outward = 1.0 if ld.x >= model.L / 2 else -1.0
            if d[0] * outward > 0:
                tail = base
                tip = base + La * d
            else:
                tip = base
                tail = base - La * d
        traces += _arrow(tail, tip, head, "point load", hover)
        labels.append((tail[0], tail[1], _fmt_force(ld.magnitude)))

    dls = model.distributed_loads
    qmax = max((max(abs(w.w1), abs(w.w2)) for w in dls), default=0.0)
    for w in dls:
        n = max(3, int(round(14 * (w.x2 - w.x1) / model.L)) + 1)
        xs = np.linspace(w.x1, w.x2, n)
        q = np.abs(w.w1 + (w.w2 - w.w1) * (xs - w.x1) / (w.x2 - w.x1))
        down = w.direction == "down"
        lens = np.where(q > 0, np.maximum(q / qmax, 0.12), 0.0) * 0.8 * La if qmax > 0 else np.zeros(n)
        hover = (f"Distributed load {w.w1 / 1e3:.3g} → {w.w2 / 1e3:.3g} kN/m ({w.direction}), "
                 f"x = {w.x1:.4g} to {w.x2:.4g} m")
        tails = []
        for xi, Li in zip(xs, lens):
            surf = yc(xi) + (top if down else bot)
            tail = np.array([xi, surf + (Li if down else -Li)])
            tails.append(tail)
            if Li > 0:
                traces += _arrow(tail, [xi, surf], 0.6 * head, "distributed load", hover)
        tails = np.array(tails)
        traces.append(go.Scatter3d(x=tails[:, 0], y=np.zeros(n), z=tails[:, 1], mode="lines",
                                   line=dict(color=AMBER, width=4), hoverinfo="text", hovertext=hover,
                                   showlegend=False))
        i = int(np.argmax(lens))
        labels.append((tails[i, 0], tails[i, 1], f"{max(abs(w.w1), abs(w.w2)) / 1e3:.3g} kN/m"))

    for mo in model.moment_loads:
        r = 1.1 * glyph
        c = np.array([mo.x, yc(mo.x)])
        sgn = 1.0 if mo.M > 0 else -1.0
        phi = np.radians(np.linspace(-45, 225, 40))[::int(sgn)]
        px, pz = c[0] + r * np.cos(phi), c[1] + r * np.sin(phi)
        sense = "CCW" if mo.M > 0 else "CW"
        hover = f"Moment {abs(mo.M) / 1e3:.3g} kN·m {sense} at x = {mo.x:.4g} m"
        traces.append(go.Scatter3d(x=px, y=np.zeros_like(px), z=pz, mode="lines", line=dict(color=AMBER, width=6),
                                   hoverinfo="text", hovertext=hover, showlegend=False))
        tang = np.array([px[-1] - px[-2], pz[-1] - pz[-2]])
        tang /= np.linalg.norm(tang)
        traces.append(go.Cone(x=[px[-1]], y=[0], z=[pz[-1]], u=[tang[0]], v=[0], w=[tang[1]], anchor="tip",
                              sizemode="absolute", sizeref=head, colorscale=[[0, AMBER], [1, AMBER]],
                              showscale=False, hoverinfo="text", hovertext=hover, showlegend=False))
        labels.append((c[0], c[1] + r, f"{abs(mo.M) / 1e3:.3g} kN·m {sense}"))

    if labels:
        lx, lz, lt = zip(*labels)
        traces.append(go.Scatter3d(x=lx, y=[0] * len(lx), z=lz, mode="text", text=lt, textposition="top center",
                                   textfont=dict(color=AMBER, size=12), hoverinfo="skip", showlegend=False))
    return traces


def beam_figure(result, amplification: float | None = None, show_undeformed: bool = True,
                show_deformed: bool = True, show_stress: bool = True, show_supports: bool = True,
                show_loads: bool = True, show_markers: bool = True, height: int = 560) -> go.Figure:
    """3D figure of the analysed beam. amplification None -> auto_amplification(result)."""
    model = result.model
    s = model.section
    amp = auto_amplification(result) if amplification is None else float(amplification)
    sig_all = np.concatenate([np.abs(result.sigma_top), np.abs(result.sigma_bottom)])
    cmax = max(float(sig_all.max()) / 1e6, 1e-12)
    glyph = max(1.2 * s.depth, 0.05 * model.L)
    half_w = max((s.b if s.shape == "rect" else s.depth) / 2, 0.25 * glyph)

    traces = []
    if show_undeformed:
        traces += _beam_surfaces(result, 0.0, show_stress and not show_deformed, cmax,
                                 opacity=0.18 if show_deformed else 1.0, hover=not show_deformed,
                                 showscale=show_stress and not show_deformed)
        traces.append(go.Scatter3d(x=[0, model.L], y=[0, 0], z=[0, 0], mode="lines", showlegend=False,
                                   line=dict(color=MUTED, width=2, dash="dash"), hoverinfo="skip"))
    if show_deformed:
        traces += _beam_surfaces(result, amp, show_stress, cmax, showscale=show_stress)
        x, v, *_ = _fields(result)
        traces.append(go.Scatter3d(
            x=x, y=np.zeros_like(x), z=amp * v, mode="lines", line=dict(color=CYAN, width=5),
            customdata=np.c_[v * 1e3, np.degrees(result.theta)], showlegend=False, name="deflection",
            hovertemplate="x = %{x:.4g} m<br>v = %{customdata[0]:.4g} mm<br>θ = %{customdata[1]:.4g}°"
                          "<extra>centreline</extra>"))
    if show_supports:
        traces += _supports(model, glyph, half_w)
    disp_amp = amp if show_deformed else 0.0
    if show_loads:
        traces += _loads(result, disp_amp, glyph)
    if show_markers:
        sm, vm = result.sigma_max, result.v_max
        x, v, th, u, _ = _fields(result)
        ys = s.depth / 2 if sm.where == "top" else -s.depth / 2
        i = int(np.argmin(np.abs(x - sm.x)))
        sx = sm.x + disp_amp * (u[i] - ys * th[i])
        sz = ys + disp_amp * v[i]
        traces.append(go.Scatter3d(
            x=[sx], y=[-half_w], z=[sz], mode="markers+text", marker=dict(size=7, color=RED, symbol="diamond"),
            text=[f"|σ|max {abs(sm.value) / 1e6:.4g} MPa"],
            textposition="bottom right" if sm.x < model.L / 2 else "bottom left",
            textfont=dict(color=RED, size=12), showlegend=False,
            hovertemplate=f"Max |σ| = {sm.value / 1e6:.4g} MPa ({sm.where} fibre) at x = {sm.x:.4g} m<extra></extra>"))
        traces.append(go.Scatter3d(
            x=[vm.x], y=[0], z=[disp_amp * vm.value], mode="markers+text",
            marker=dict(size=7, color=CYAN, symbol="circle", line=dict(color=TEXT, width=1)),
            text=[f"|v|max {abs(vm.value) * 1e3:.4g} mm"],
            textposition="top right" if vm.x < model.L / 2 else "top left",
            textfont=dict(color=CYAN, size=12), showlegend=False,
            hovertemplate=f"Max |v| = {vm.value * 1e3:.4g} mm at x = {vm.x:.4g} m<extra></extra>"))

    axis = dict(backgroundcolor="#0E1626", gridcolor="#1F2B42", zerolinecolor="#24324D", color=MUTED,
                showspikes=False)
    fig = go.Figure(traces)
    # True proportions, normalised so the longest axis is 1. Plotly's aspectmode="data" normalises by the
    # geometric mean instead, which makes a slender beam ~5 units long and puts the camera inside the scene.
    span = np.ones(3)
    for k, ax in enumerate("xyz"):
        vals = np.concatenate([np.ravel(np.asarray(getattr(t, ax), dtype=float)) for t in traces
                               if getattr(t, ax, None) is not None])
        span[k] = max(np.nanmax(vals) - np.nanmin(vals), 1e-12)
    ratio = span / span.max()
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(aspectmode="manual", aspectratio=dict(x=ratio[0], y=ratio[1], z=ratio[2]),
                   xaxis=dict(title="x (m)", **axis), yaxis=dict(title="z (m)", **axis),
                   zaxis=dict(title="y (m)", **axis),
                   camera=dict(eye=dict(x=0.2, y=-1.05, z=0.42), up=dict(x=0, y=0, z=1))),
        hoverlabel=dict(bgcolor="#16213A", font=dict(color=TEXT)),
    )
    return fig
