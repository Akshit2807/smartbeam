# SmartBeam 2.0 — Consolidated Requirements

Source: student's 2nd-year Solid Mechanics engineering software project brief (3 documents, merged).
Final architecture: ONE integrated Streamlit application (NO separate HTML website). Home/Dashboard and
About/Methodology live inside Streamlit.

Tagline: **Understand. Simulate. Predict. Optimize.**

## Global rules (apply everywhere)
- Never fabricate results. Every number/plot comes from the real calculation engine.
- No decorative / non-functional controls.
- Engineering accuracy; SI units internally; convenient I/O units (mm, N, kN, MPa, GPa, N/m, kN/m, N·m, kN·m).
- Calculation engine fully separated from UI.
- Do not silently apply a textbook formula outside its valid configuration.
- Do NOT implement 2D continuum FEA (no triangles/quads, plane stress/strain, 2D meshing). Only 1D line elements
  (beam, Timoshenko beam, planar frame element with axial + bending DOFs).
- No paid APIs, no cloud, no DB. Local only. Avoid unnecessary dependencies.
- Do not invent personal info (team/institute/guide fields are editable placeholders).
- Do not invent fatigue properties for materials. Label reference material properties as approximate.
- Distinguish yielding from ultimate fracture; do not claim fracture prediction.
- Professional, visually impressive UI: dark navy / charcoal / white with careful accent colours.

Stack: Python 3.12 (venv at .venv), NumPy, SciPy, Pandas, SymPy (symbolic verification where useful), Plotly,
Matplotlib, Streamlit, ReportLab (PDF), Pandas (CSV), Pytest.

## Module 1 — Home Dashboard
Branding, intro, capabilities, quick navigation, example beam model (interactive Plotly), recent analysis
configurations, summary of tools.

## Module 2 — Beam Design Studio
- Sections: rectangular (b, h), solid circular (d), hollow circular (D, d). Show only relevant inputs.
- Material library (editable, labelled approximate): structural steel, mild steel, aluminium, copper, brass.
  Store E, yield strength, UTS where available, Poisson's ratio, density. Custom materials allowed.
- Supports: cantilever, simply supported (Playground extends this to arbitrary fixed/pin/roller placements).
- Loads: point load, UDL, applied moment; positions, directions, magnitudes. Validate all inputs.

## Module 3 — Classical Solid Mechanics Engine
A. Section properties: A, centroid, I, section modulus Z, radius of gyration.
   rect I=bh³/12 A=bh; solid circ I=πd⁴/64 A=πd²/4; hollow I=π(D⁴−d⁴)/64 A=π(D²−d²)/4.
B. Reactions from ΣFx=0, ΣFy=0, ΣM=0. Document sign convention. No hardcoded reactions.
C. SFD/BMD numeric, point + distributed + applied moments, max |V|, max |M|, locations. Validate equilibrium and
   dM/dx = V, dV/dx = −w relationships.
D. Bending stress σ = My/I: max tensile, max compressive, distribution across section. Axial + bending combined
   normal stress where axial load exists.
E. Strain ε = σ/E distribution.
F. Slope & deflection (Euler–Bernoulli). Textbook checks:
   cantilever end load θ=PL²/2EI, δ=PL³/3EI; cantilever UDL θ=wL³/6EI, δ=wL⁴/8EI;
   SS central load δ=PL³/48EI; SS full UDL δ=5wL⁴/384EI. Other configs: reliable numerical integration.

## Module 4 — Numerical 1D Beam FEM (independent of analytical engine)
2-node EB element, DOFs [v1,θ1,v2,θ2], Ke = EI/Le³·[[12,6L,−12,6L],[6L,4L²,−6L,2L²],[−12,−6L,12,−6L],[6L,2L²,−6L,4L²]].
Discretization, nodes, connectivity, local K, global assembly, equivalent nodal loads, point loads, consistent
loads for distributed loads, BCs, reduced solve (NumPy/SciPy), displacement recovery, element force/moment
recovery, stress, deflection reconstruction (Hermite shape functions). Conditioning check; detect unstable /
under-constrained systems. Mesh convergence study (2,4,8,16,32 elements) with plot + explanation.
Analytical vs numerical: max deflection, max moment, max stress, % difference, convergence. Don't manufacture
agreement; report discrepancies/limitations honestly.

## Module 5 — Advanced Beam Theories
EB assumptions explained. Timoshenko: shear deformation, section rotation, shear correction factor κ
(rect 5/6, solid circ 6/7 or 0.9 — document; thin hollow ~0.5), G=E/[2(1+ν)]. Mathematically consistent
Timoshenko FE formulation (e.g. exact/interdependent-interpolation element with φ=12EI/(κGAL²)) — no arbitrary
correction factors. Compare EB vs Timoshenko; short/deep beams; when each theory applies.

## Module 6 — Interactive 3D Visualization (Plotly)
Undeformed + deformed beam (from actual displacement field), loads, supports, deflection profile, bending stress
colour map (beam-theory σ = My/I across section depth), max stress & max deflection markers, hover values.
Rotate/zoom/pan, deformation amplification control (clearly labelled), show/hide toggles. State clearly it is NOT a
3D continuum stress solution.

## Module 7 — Failure Prediction & Safety
A. FOS = σy/|σmax|, first-yield load. B. Combined axial+bending with suitable criterion; von Mises only with stated
assumptions (e.g. σ from bending+axial, τ from VQ/It at the relevant point) — explain. C. Euler buckling
Pcr=π²EI/(K L)², K factors (fixed-free 2.0, pinned-pinned 1.0, fixed-pinned 0.699, fixed-fixed 0.5), applied
compressive load, buckling FOS, normalized mode shape, limitations (stocky members, imperfections, slenderness
check). D. Intro stress-life fatigue: σa, σm, S–N curve (Basquin from user-entered σf′, b or endurance data),
cycles-to-failure only when user supplies data; mean-stress correction (Goodman) with assumptions.
E. Status SAFE / WARNING / YIELD PREDICTED with user-defined minimum FOS.

## Module 8 — Design Comparison Studio
Save/compare designs (session state or local JSON): material, L, section, dims, δmax, σmax, Mmax, FOS, A, volume,
mass, estimated material cost (editable ₹/kg). One-variable-at-a-time studies; thickness-vs-stiffness and
geometry-vs-weight charts.

## Module 9 — Intelligent Design Optimization (SciPy)
Minimize V=bhL (or mass, or cost) s.t. δmax ≤ δallow, FOS ≥ FOSmin, bmin≤b≤bmax, hmin≤h≤hmax (and equivalents for
circular/hollow). Inputs: L, loads, material, section, δallow, FOSmin, ranges. Display initial vs optimized:
dims, volume, % reduction, δ, σ, FOS, constraint verification table. Independently re-verify constraints with the
engine after optimization; never label infeasible results optimized; explain infeasibility. Pareto front
(material vs stiffness vs strength) with trade-off explanation.

## Module 10 — Smart Design Advisor (rule-based, NO chatbot)
Identify limiting condition (deflection, stress, buckling, material excess). For each: what's wrong, which
quantity, recommended modification, what changes after it (actually recompute), does modified design satisfy
all constraints.

## Module 11 — Engineering Virtual Laboratory (replaces website)
1. Live Structural Behaviour Explorer — change dims/material/loads/supports, everything recalculates instantly.
2. Interactive Engineering Experiments (guided, each with objective + explanation, all real calcs):
   height vs stiffness, length vs deflection, steel vs aluminium, rect vs circular, solid vs hollow,
   point vs distributed load, load position, cantilever vs SS, EB vs Timoshenko, mesh refinement/convergence,
   material volume vs performance.
3. Structural Design Challenges — problems with strength/stiffness/weight/safety constraints; user submits a design,
   engine auto-grades it (pass/fail per constraint, score).
4. Analytical vs Numerical Laboratory — vary element count, compare, convergence.
5. Structural Sensitivity Analysis — effect of each geometric/material parameter on δ, σ, stiffness, mass, FOS;
   sensitivity graphs (e.g. normalized ±% sweeps, tornado chart, log-log slopes showing δ∝L³, δ∝h⁻³) + explanation.
6. Automatic Laboratory Reports — objectives, initial conditions, governing equations, parameter changes, results,
   graphs, conclusions (PDF + CSV).

## Module 12 — Engineering Report Generator
ReportLab PDF: title, geometry, material, loading, supports, section props, reactions, SFD, BMD, stress, strain,
deflection, FEM results, analytical vs numerical, mesh convergence, failure, FOS, optimization, recommendations,
assumptions & limitations. Embedded graphs (Matplotlib PNGs). CSV export of numerical data.

## Module 13 — Interactive Structural Playground (SIGNATURE FEATURE)
Graphical canvas with genuine drag-and-drop, bidirectionally connected to Python engine (custom Streamlit
component, plain HTML/JS/SVG/canvas is fine — no npm build required).
- Toolbox: beam (adjustable length + section); supports fixed / pinned / roller (placeable along beam);
  forces: vertical down, vertical up, horizontal axial, inclined, multiple; distributed: full UDL, partial UDL,
  linearly varying, triangular (start, end, magnitudes, direction); moments CW / CCW (magnitude, position).
- Drag from toolbox → drop on beam → move later → select to edit → delete → multiple components → save/reopen.
- Visuals: point load arrows, distributed = arrow series, moments = curved arrows, fixed = wall hatch,
  pinned = triangle, roller = triangle + circles. Beam length, position coordinates, grid snapping, measurement
  scale.
- Properties panel (right): point load (magnitude N/kN, position m/mm, direction, angle); distributed (start, end,
  intensity N/m or kN/m, type, direction); moment (N·m/kN·m, position, CW/CCW); support (type, position, restrained
  DOFs); beam (length, section, material). All editable.
- Analyse Structure button: read config → internal model → validate → reactions, V, M, σ, ε, slope, deflection,
  FEM → update all panels. Optional live-analysis mode (recalc after short interaction pause).
- After analysis toggle: configuration / SFD / BMD / deflection / stress. Link to 3D visualization of same beam.
- Toolbar: undo, redo, delete selected, clear, save config, load config, duplicate load, change length, snapping,
  zoom & pan, reset workspace, presets: cantilever end load, cantilever UDL, SS central load, SS full UDL,
  SS partial UDL, multiple concentrated loads.
- Layout: left toolbox, large central canvas, right properties panel, top toolbar, bottom results.
- Solver: transverse beam bending for the basic case; axial/inclined loads → planar frame line element (u, v, θ per
  node). Do NOT apply transverse EB equations to axial/frame loading silently. Reject unstable configs with clear
  explanation (rigid-body check). Statically indeterminate (redundant restraints) → stiffness-based solution
  (FEM), explain that statics alone is insufficient.
- Final demo must work: pinned + roller supports, arbitrary point load, partial UDL, applied moment, change
  magnitudes/positions, Analyse, reactions + SFD + BMD, deformation + stress, numerical comparison, modify & update.

## Testing (mandatory, pytest)
Benchmarks: (1) cantilever end load — reactions, Mmax, σmax, θmax, δmax; (2) cantilever full UDL — reactions, Vmax,
Mmax, δmax; (3) SS central load — reactions, Mmax, σmax, δmax; (4) SS full UDL — reactions, Mmax, δmax.
Also: section properties, element stiffness matrix, global assembly, boundary conditions, FEM vs analytical,
Timoshenko → EB as L/h grows, buckling, optimization constraint satisfaction, playground presets vs analytical,
component editing/positioning, load direction, multiple loads, distributed loads, support movement, invalid
configs, save/restore configs, live recalculation logic.
(Original brief section 15 was truncated after "Boun…" — assume boundary-condition tests and similar.)

## Deliverables
Folder structure, README with exact local run instructions, requirements.txt, tests passing, app running.
