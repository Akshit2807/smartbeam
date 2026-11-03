# SmartBeam 2.0

**Understand. Simulate. Predict. Optimize.**

> **Live app:** https://smartbeam2.streamlit.app · **Reviewers:** start with [TESTING.md](TESTING.md)

SmartBeam 2.0 is a beam-analysis application for a second-year Solid Mechanics course. It is a single local
Streamlit app with 14 pages. Every number and plot comes from a Python calculation engine (`smartbeam/`) that is kept
separate from the UI (`pages/`, `smartbeam/ui/`). The engine solves each beam in two independent ways:

- **Classical solid mechanics:** statics, then numerical integration of EI v'' = M.
- **1D finite elements:** Euler–Bernoulli, Timoshenko and planar frame elements.

The two solvers share no code, so when they agree, that agreement is a real cross-check.

Team: `[Team members]` · Institute: `[Institute]` · Guide: `[Guide]` · Course: `[Course – Solid Mechanics]`
(You can also edit these placeholders on the About page. The Reports page copies them into the PDF.)

---

## Features by module

| # | Page | What it does |
|---|---|---|
| 1 | **Home** | Branding, a live 3D view of the current beam, key results, capability cards that link to every page, a recent-analyses table with "load into workspace". |
| 2 | **Design Studio** | Rectangular, solid circular and hollow circular sections; it shows only the inputs the chosen section needs. Editable material library (structural steel, mild steel, aluminium 6061-T6, copper, brass; values labelled approximate) plus custom materials. Cantilever and simply supported presets. Loads: point (down/up/axial/inclined), UDL, linearly varying load, CW/CCW moments, with unit selectors and validation on every row. JSON import/export, and named snapshots. |
| 3 | **Analysis** | Reactions from ΣFx = ΣFy = ΣM = 0, with equilibrium residuals and dM/dx = V, dV/dx = q checks. SFD, BMD and axial-force diagrams. Fibre stress against ±σy. σ, ε and τ through the depth at any x. Slope and deflection. Textbook comparison (only for the 4 recognised cases). CSV export. |
| 4 | **Numerical FEM** | 1–64 elements, Euler–Bernoulli or Timoshenko. Mesh and connectivity, the element stiffness matrix (LaTeX and numbers), a heatmap and table of the global K, nodal displacements, reactions and element end forces. FEM vs analytical, and a 2→32-element convergence study. |
| 5 | **Beam Theories** | Euler–Bernoulli vs Timoshenko assumptions, both theories run on your beam (κ, G, φ = 12EI/κGAL²), and an L/h sweep that shows the shear contribution falling with slope −2 on log–log axes. |
| 6 | **3D Visualization** | Plotly 3D: undeformed and deformed beam (from the computed v(x) and θ(x)), σ = N/A − My/I colour map, supports, loads, max σ and max v markers, hover values. The deformation amplification (auto, ×1 or custom) is clearly labelled. The page states that this is **not** a 3D continuum solution. |
| 7 | **Safety & Failure** | FOS = σy/abs(σ)max and the first-yield load factor. von Mises at the critical point (assumptions stated). Euler buckling with K = 2.0 / 1.0 / 0.699 / 0.5, Pcr, slenderness and mode shape. Stress-life fatigue (Basquin S–N, Goodman), shown **only** when you enter σf′, b and Se. Status SAFE / WARNING / YIELD PREDICTED / BUCKLING PREDICTED against a minimum FOS you choose. |
| 8 | **Design Comparison** | Save designs (or one copy per library material). Comparison table, bar and radar charts, editable ₹/kg rates. One-variable studies with fitted log–log exponents. Thickness-vs-stiffness and geometry-vs-weight charts. |
| 9 | **Optimization** | SciPy SLSQP minimises volume, mass or cost subject to δmax ≤ δallow, FOS ≥ FOSmin, dimension bounds, and (for compressed members) buckling FOS. Every result is re-checked by an independent engine run. Infeasible designs are explained, never labelled "optimized". Pareto front of material vs stiffness vs strength. |
| 10 | **Design Advisor** | Rule-based, not a chatbot. Finds the limiting condition (deflection, stress, buckling or material excess). Each recommendation is sized from beam theory, re-analysed, and shown with a before/after table and an "Apply" button. |
| 11 | **Virtual Lab** | Live explorer (5 support layouts, 4 load types). 11 guided experiments, each computed from scratch. 5 auto-graded design challenges. An analytical-vs-numerical lab. Sensitivity analysis (tornado, normalised heatmap, log–log slopes such as δ ∝ L³ and δ ∝ h⁻³). PDF/CSV lab reports. |
| 12 | **Reports** | ReportLab PDF covering geometry, material, loads, supports, section properties, reactions, SFD/BMD, stress, strain, deflection, FEM, analytical vs FEM, mesh convergence, failure/FOS, optimization, advisor, and assumptions. Plots are embedded Matplotlib PNGs. Three CSV exports. |
| 13 | **Structural Playground** | Custom Streamlit component (plain HTML/JS/SVG, no build step). Drag-and-drop toolbox; move, resize, edit and delete items; undo/redo; snapping; zoom and pan; 6 presets; JSON save/load. **Analyse Structure** or Live mode. SFD/BMD/deflection/stress overlays drawn on the canvas, plus full result panels below. Mechanisms are rejected with an explanation. Indeterminate beams go to FEM, and axial or inclined loads go to planar frame elements. |
| – | **About & Methodology** | Editable project details, all governing equations, validation benchmarks computed live on each visit, technology versions read at runtime, and the limitations. |

### Solver routing (never a silent textbook formula)

| Model | Route |
|---|---|
| Invalid data | `ModelError` with every problem listed |
| Mechanism (rigid-body motion possible) | `UnstableStructureError` with the explanation; the analysis is rejected |
| Transverse loads, statically determinate | Statics + numerical integration (`analytical`) |
| Transverse loads, statically indeterminate | Euler–Bernoulli FEM (`fem-euler`); the page states that statics alone is insufficient |
| Any axial or inclined load | Planar frame elements (u, v, θ per node) (`fem-frame`) |

Textbook formulas are shown only for: a cantilever with an end load, a cantilever with a full UDL, a simply supported
beam with a central load, and a simply supported beam with a full UDL.

---

## Folder structure

```
smartbeam/                         project root (run everything from here)
├── app.py                         Streamlit entry: navigation + theme
├── requirements.txt
├── README.md
├── .streamlit/config.toml         dark navy theme
├── docs/
│   ├── REQUIREMENTS.md            consolidated project brief
│   └── ARCHITECTURE.md            engine API contract, sign convention, contract changes
├── pages/                         one file per page (UI only, calls the engine)
│   ├── home.py  design_studio.py  analysis.py  numerical_fem.py  beam_theories.py
│   ├── visualization_3d.py  safety_failure.py  design_comparison.py  optimization.py
│   └── design_advisor.py  virtual_lab.py  playground.py  reports.py  about.py
├── smartbeam/                     calculation engine (no Streamlit imports outside ui/ and playground/component.py)
│   ├── model.py                   data classes, validation, JSON, stability/determinacy
│   ├── sections.py                A, I, Z, r, Q(y), b(y), shear correction factor κ
│   ├── materials.py               5-material library (approximate values) + custom
│   ├── units.py                   SI ↔ display units
│   ├── metrics.py                 shared design metrics and constraint checks
│   ├── analytical.py              statics + integration, textbook checks, analyze() router
│   ├── fem.py                     Euler–Bernoulli / Timoshenko / frame FEM, convergence study
│   ├── safety.py                  yield, von Mises, Euler buckling, fatigue
│   ├── optimize.py                SLSQP sizing, re-verification, Pareto front
│   ├── advisor.py                 rule-based design advisor
│   ├── lab.py                     experiments, challenges, sensitivity
│   ├── report.py                  ReportLab PDF + CSV
│   ├── viz3d.py                   Plotly 3D beam figure
│   ├── playground/
│   │   ├── convert.py             canvas config ↔ BeamModel, presets, save/load
│   │   ├── component.py           Streamlit custom component wrapper
│   │   └── frontend/              index.html, playground.js, playground.css (SVG editor)
│   └── ui/
│       ├── theme.py  state.py     CSS/Plotly template, session state + cached analysis
│       └── plots_core.py  design_tools.py  lab_plots.py   shared page helpers
├── saved_configs/final_demo.json  Playground configuration for the viva demo
└── tests/                         316 pytest tests (conftest.py + test_*.py)
```

---

## Setup and run

All commands are run from the project root. You need Python 3.12.

### Option A: uv

```bash
cd smartbeam
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

### Option B: plain venv + pip

```bash
cd smartbeam
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. Navigate with the sidebar. Every page analyses the same shared "current
design".

### Tests

```bash
.venv/bin/python -m pytest -q        # or simply `pytest -q` inside the activated venv
```

Expected: `316 passed`.

---

## Viva demonstration walkthrough

Allow about 10 minutes. Each step names the page and what to point out.

1. **Home.** State the tagline and point to the live 3D view of the default beam (a steel cantilever, 30 × 60 mm,
   L = 1 m, 1 kN at the tip). Key results: δmax = 3.086 mm and σmax = 55.56 MPa, which match PL³/3EI and PL/Z.
2. **Structural Playground** (the signature feature):
   1. Click **Clear**.
   2. Click or drag from the toolbox: **Pinned** (lands at x = 0), **Roller** (lands at x = L), **Point ↓**,
      **Partial UDL**, **Moment ↻**.
   3. Select each item and edit it in the right panel. For example: point load 2 kN at x = 0.3 m, and moment
      0.5 kN·m CW.
   4. Drag a load along the beam with the mouse. It snaps to the grid.
   5. Press **▶ Analyse Structure**. The reaction arrows appear on the canvas, and the canvas toggles show the
      **SFD, BMD, Deflection and Stress** overlays.
   6. Scroll down to the results: metric cards, then tabs for Diagrams, Reactions & equilibrium (residuals ≈ 0),
      **FEM vs analytical**, Safety, and Theory.
   7. Tick **Live**, then drag a load or change a magnitude. Everything re-analyses after a short pause.
   - Prepared alternative: open *Configuration files · saved_configs/*, choose `final_demo`, click **Load into
     canvas**, then **Analyse**. This beam has a pin at 0, a roller at 1.6 m (overhang), 1.5 kN at the tip, a
     2 kN/m partial UDL on 0.4–1.2 m, and 0.8 kN·m CW at 1 m. The answers are R_A = −0.075 kN (downward),
     R_B = 3.175 kN and M(1.6) = −0.6 kN·m. Check them by hand: ΣM_A = 0 gives
     R_B = (1.6·0.8 + 1.5·2 + 0.8)/1.6.
3. **Show the robustness:**
   - Select the pin, press **Delete**, then **Analyse**. The canvas shows an "Unstable: … rigid-body …" banner and
     the status bar reads "Analysis rejected".
   - Press **Undo**, select the pin, and tick **θ** under *Restrained* (or set *Type* to Fixed). Analyse again. The
     beam is now statically indeterminate to degree 1, so it is solved by FEM (`fem-euler`), and the page says that
     equilibrium alone is insufficient.
   - Add an **Inclined** load. The beam is now solved with planar frame elements and an axial-force diagram appears.
4. Click **Use as current design** (or **Open in 3D Visualization**). The same beam now drives every other page.
5. **Analysis.** Show the reactions table, the equilibrium checks, SFD/BMD with the maxima marked, the stress
   distribution through the depth, and slope/deflection.
6. **Numerical FEM.** Show the mesh, the element stiffness matrix, the banded global K, and FEM vs analytical
   (% differences). Show the mesh-convergence study. Switch to Timoshenko and back.
7. **3D Visualization.** Rotate the view and change the amplification. Point out the stress colour map and the
   disclaimer that this is beam theory, not solid FEA.
8. **Safety & Failure.** Change the minimum FOS to see the status change. Show the buckling tab (for a compressed
   member) and the fatigue tab, which works only with your own data.
9. **Optimization → Design Advisor.** Run the optimization, show the constraint table (re-verified), and apply the
   result. Show how the advisor explains the limiting condition.
10. **Virtual Lab.** Run the "height vs stiffness" experiment and point out the fitted slope of −3.000 (δ ∝ h⁻³).
11. **Reports.** Fill in the title page, **Build PDF report**, download it, and show the embedded diagrams.
12. **About & Methodology → Validation benchmarks.** These benchmarks are recomputed live on each visit.

---

## Engineering assumptions and limitations

- **Beam theory:** linear elastic, small deflections, prismatic member, homogeneous isotropic material. Bending is
  about the horizontal centroidal axis. Euler–Bernoulli by default; Timoshenko is optional (κ: rectangle 5/6, solid
  circle 0.9, hollow circle by Cowper).
- **1D line elements only:** no 2D or 3D continuum FEA. The 3D view is a visualisation of beam-theory results.
- **Stress:** σ = N/A − M·y/I and τ = VQ/(Ib). No stress concentrations, residual stresses, contact or bearing
  stresses. von Mises is evaluated only from these σ and τ, as stated on the page.
- **Failure:** yield is not fracture. SmartBeam predicts the onset of yielding, not ultimate failure. Euler buckling
  is elastic and assumes ideal members. Stocky members, imperfections, local buckling and lateral–torsional
  buckling are not modelled, so optimized or challenge sections can come out thin.
- **Material data:** library values are approximate typical values; verify them against supplier or standard data.
  No fatigue properties are invented, so the fatigue results need your own σf′, b and Se.
- **Supports:** fixed (u, v, θ), pin (u, v) and roller (v) only. Springs, settlements and hinges are not modelled.
- **Loads:** static only. Distributed loads are transverse; axial or inclined effects use point loads.
- **FEM:** nodal values are exact for Euler–Bernoulli elements. Between nodes, under distributed loads, the cubic
  interpolation differs from the exact quartic by about 1e-5 relative. The FEM route reports equilibrium sums but no
  differential-equation residuals.
- **UI:**
  - The Playground canvas switches between library materials only; a custom material from Design Studio passes
    through unchanged.
  - Loads placed at the same x can have overlapping labels.
  - Over the canvas, Ctrl/⌘ + wheel zooms and a plain wheel scrolls the page.
- **Optimization section in the PDF:** it appears only when the Optimization page was run for the exact current
  model.

---

## Validation summary

The values below were produced by running the engine (`analytical.analyze` and `fem.solve` with 20 elements).
They are the same cases the test suite checks.

- **Beam:** structural steel E = 200 GPa, rectangle b × h = 30 × 60 mm (I = 5.4 × 10⁵ mm⁴, Z = 1.8 × 10⁴ mm³),
  L = 2 m.
- **Loads:** P = 1 kN, w = 0.5 kN/m.

| Benchmark | Closed form | δmax closed form (mm) | Engine δmax (mm) [route] | FEM δmax (mm) | Engine error % | FEM error % | Mmax closed / engine (N·m) | σmax closed / engine (MPa) |
|---|---|---|---|---|---|---|---|---|
| Cantilever, end load | PL³/3EI | 24.69136 | 24.69136 [analytical] | 24.69136 | 0 | 6e-10 | 2000 / 2000 | 111.111 / 111.111 |
| Cantilever, full UDL | wL⁴/8EI | 9.25926 | 9.25926 [analytical] | 9.25926 | 9e-14 | 6e-10 | 1000 / 1000 | 55.556 / 55.556 |
| Simply supported, central load | PL³/48EI | 1.54321 | 1.54321 [analytical] | 1.54321 | 6e-14 | 2e-10 | 500 / 500 | 27.778 / 27.778 |
| Simply supported, full UDL | 5wL⁴/384EI | 0.96451 | 0.96451 [analytical] | 0.96451 | 2e-14 | 2e-10 | 250 / 250 | 13.889 / 13.889 |
| Propped cantilever, full UDL (indeterminate) | (39+55√33)/65536 · wL⁴/EI | 0.40119 | 0.40119 [fem-euler] | 0.40119 | 6e-4 | 2e-3 | 250 / 250 | 13.889 / 13.889 |
| Fixed–fixed, central load (indeterminate) | PL³/192EI | 0.38580 | 0.38580 [fem-euler] | 0.38580 | 5e-10 | 3e-11 | 250 / 250 | 13.889 / 13.889 |

The propped-cantilever errors are larger because its maximum lies between mesh nodes (see "FEM" under the
limitations).

Further checks, all computed by the engine and asserted in `tests/`:

- **Timoshenko cantilever:** FEM δ = 24.70869 mm against PL³/3EI + PL/(κGA) = 24.70869 mm (8 × 10⁻¹⁰ %). The shear
  share falls as (L/h)⁻² (fitted log–log slope −2).
- **Euler buckling:** fixed–free, Ø50 mm steel, L = 2 m gives Pcr = 37.849 kN, equal to π²EI/(2L)².
- **Symbolic check:** SymPy solutions of EI v'' = M match the engine's deflection curve at every sample point.
- **Lab sensitivity exponents:** δ ∝ L³, h⁻³, b⁻¹, E⁻¹; σ ∝ h⁻² (to 1 × 10⁻⁶).
- **Random beams:**
  - 40 random beams: the analytical engine and FEM agree on reactions, moment and deflection.
  - 20 random beams: Timoshenko FEM equals Euler–Bernoulli plus the exact shear term.
- **Cross-page consistency:** for 5 test models, Analysis, Playground, 3D and the Reports summary CSV show the same
  δmax and σmax.

---

## Sign convention

- **Axes and loads:**
  - x runs from the left end and y is up.
  - Point-load angles are measured counter-clockwise from +x, so −90° is downward.
  - Applied moments are positive counter-clockwise.
- **Internal forces:**
  - V(x) = ΣFy to the left of the cut, so dV/dx = q.
  - M is positive when sagging, so dM/dx = V.
  - N is positive in tension.
- **Deflection and stress:** EI v'' = M, with v positive upward. σ = N/A − M·y/I, with tension positive. The full
  contract is in `docs/ARCHITECTURE.md`.
