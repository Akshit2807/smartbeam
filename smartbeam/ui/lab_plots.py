"""Virtual Lab figures (Plotly on the page, Matplotlib in PDF lab reports) and the lab-report PDF glue.
OWNER: lab engineer. Every plotted number comes from a lab.ExperimentResult table.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure
from plotly.subplots import make_subplots

from smartbeam import lab, report
from smartbeam.ui.theme import COLORS, SERIES

_PDF_COLORS = ["#2D6CDF", "#B8975A", "#12355B", "#2E7D4F", "#7EA2E3", "#E06A5A"]


def _series(res: lab.ExperimentResult):
    """[(name, sub-table)] split by res.group (one unnamed series when there is no group)."""
    df = res.table
    if res.group:
        return [(str(g), df[df[res.group] == g]) for g in df[res.group].unique()]
    return [(None, df)]


def _xy(res, sub, ycol):
    xl, x = lab.display_column(res.x, sub[res.x].to_numpy())
    yl, y = lab.display_column(ycol, sub[ycol].to_numpy())
    if res.plot == "bar":
        x = [str(v) for v in x]
    return xl, np.asarray(x) if res.plot != "bar" else x, yl, np.asarray(y, float)


def experiment_figure(res: lab.ExperimentResult, height: int = 380) -> go.Figure:
    """One subplot per plotted column; log-log axes get the fitted slope of each series annotated."""
    ys = list(res.y)
    titles = [lab.display_column(c, [])[0] for c in ys]
    fig = make_subplots(1, len(ys), subplot_titles=titles, horizontal_spacing=0.1)
    for j, ycol in enumerate(ys, 1):
        for i, (name, sub) in enumerate(_series(res)):
            xl, x, yl, y = _xy(res, sub, ycol)
            color = SERIES[i % len(SERIES)] if res.group else SERIES[(j - 1) % len(SERIES)]
            label = name or yl
            hover = f"{xl} = %{{x}}<br>{yl} = %{{y:.5g}}<extra>{name or ''}</extra>"
            if res.plot == "bar":
                fig.add_trace(go.Bar(x=x, y=y, name=label, marker_color=color, showlegend=bool(name) and j == 1,
                                     legendgroup=label, hovertemplate=hover), 1, j)
            else:
                if res.log_axes:
                    slope = lab.loglog_slope(x, y)
                    label = f"{label} · slope {slope:.3f}"
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=label, legendgroup=label,
                                         line=dict(color=color, width=2.4), marker=dict(size=7),
                                         hovertemplate=hover), 1, j)
            fig.update_xaxes(title_text=xl, row=1, col=j)
            fig.update_yaxes(title_text=yl, row=1, col=j)
    if res.log_axes:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log", exponentformat="power")
    if res.plot == "bar":
        fig.update_layout(barmode="group")
    fig.update_layout(height=height, margin=dict(t=56, b=48), showlegend=res.log_axes or bool(res.group),
                      legend=dict(orientation="h", y=-0.22, x=0))
    fig.update_annotations(font=dict(color=COLORS["muted"], size=13))
    return fig


def report_figures(res: lab.ExperimentResult) -> list[Figure]:
    """The same plots as Matplotlib figures (light style) for the PDF report."""
    ys = list(res.y)
    if not ys or res.table.empty:
        return []
    fig = Figure(figsize=(7.0, 2.8 if len(ys) == 1 else 3.0), layout="constrained")
    axes = fig.subplots(1, len(ys), squeeze=False).ravel()
    series = _series(res)
    for j, (ax, ycol) in enumerate(zip(axes, ys)):
        width = 0.8 / len(series)
        for i, (name, sub) in enumerate(series):
            xl, x, yl, y = _xy(res, sub, ycol)
            color = _PDF_COLORS[(i if res.group else j) % len(_PDF_COLORS)]
            if res.plot == "bar":
                cats = list(dict.fromkeys(str(v) for v in lab.display_column(res.x, res.table[res.x].to_numpy())[1]))
                pos = np.array([cats.index(v) for v in x]) + (i - (len(series) - 1) / 2) * width
                ax.bar(pos, y, width=width, color=color, label=name)
                ax.set_xticks(range(len(cats)), cats, rotation=20 if max(map(len, cats)) > 10 else 0,
                              ha="right" if max(map(len, cats)) > 10 else "center", fontsize=7)
            else:
                lbl = name or None
                if res.log_axes:
                    lbl = f"{name + ' · ' if name else ''}slope {lab.loglog_slope(x, y):.3f}"
                ax.plot(x, y, "o-", color=color, linewidth=1.5, markersize=3.5, label=lbl)
            ax.set_xlabel(xl if res.plot != "bar" else "", fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
        if res.log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.grid(True, color="#EFE8D6", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if res.group or res.log_axes:
            ax.legend(fontsize=7, frameon=False)
    return [fig]


def lab_pdf(res: lab.ExperimentResult, meta: report.ReportMeta | None = None) -> bytes:
    """PDF lab report via report.build_lab_report: objective, initial conditions, equations, parameter changes,
    results (display units), graph, conclusions."""
    title = f"Lab report: {res.title}"
    meta = dataclasses.replace(meta, title=title) if meta else report.ReportMeta(title=title)
    objective = res.objective + (f" Hypothesis: {res.hypothesis}" if res.hypothesis else "")
    return report.build_lab_report(
        title=res.title, objective=objective, initial_conditions=res.initial_conditions, equations=res.equations,
        parameter_changes=res.parameter_changes or res.parameter, results_df=lab.display_table(res.table),
        figures=report_figures(res), conclusions=[res.conclusion, f"Theory: {res.explanation}"], meta=meta)
