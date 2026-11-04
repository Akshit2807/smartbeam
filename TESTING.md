# SmartBeam 2.0 — Tester's Verification Guide

This guide is for a reviewer who wants to check that SmartBeam works and that its numbers are right.
Plan on about 15 minutes for the automated checks and 30–45 minutes for the manual walkthrough.

- **Live app:** https://smartbeam.onrender.com, or run it locally (section 1).
- **Feature list:** `README.md` → "Features by module".
- **Limitations:** section 5 of this guide.

---

## 1. Run it yourself

You need Python 3.12.

```bash
git clone https://github.com/Akshit2807/smartbeam.git
cd smartbeam
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                            # expected: 316 passed
streamlit run app.py                 # opens http://localhost:8501
```

---

## 2. Automated checks (pytest)

`pytest -q` must end with **316 passed** and no failures. Many tests are parametrized, so there are more test
cases than `def test_` functions.

| File | What it proves |
|---|---|
| `tests/test_validation.py` | The 4 textbook benchmarks (cantilever end load, cantilever UDL, simply supported central load, simply supported UDL): reactions, V, M, σ, θ and δ, analytical **and** FEM. Also indeterminate beams, random-beam cross-checks, and a SymPy symbolic deflection check. |
| `tests/test_fem.py` | Element stiffness matrix (values, symmetry, rank 2), global assembly, boundary conditions, mechanism detection, Timoshenko and frame elements, mesh convergence. |
| `tests/test_analytical.py` | Statics, SFD/BMD jumps, equilibrium and dM/dx = V checks, routing of indeterminate or axial cases away from textbook formulas. |
| `tests/test_foundation.py` | Section properties (A, I, Z, r, Q), material library, model validation, JSON round trip. |
| `tests/test_safety.py` | FOS, status thresholds, von Mises, Euler buckling K factors, fatigue refusing to run without user data. |
| `tests/test_optimize.py` | Optimized designs satisfy every constraint when re-checked; infeasible problems are reported as infeasible. |
| `tests/test_advisor.py` | Each recommended fix, when re-analysed, actually passes. |
| `tests/test_lab.py` | Experiments, challenge grading, sensitivity exponents (δ ∝ L³, δ ∝ h⁻³, σ ∝ h⁻²). |
| `tests/test_playground.py` | Canvas config ↔ engine model: presets, positions, edits, direction flips, multiple and distributed loads, support movement, invalid configs, save/restore, live action. |
| `tests/test_report.py` | PDF bytes start with `%PDF`; CSV contents. |
| `tests/test_viz3d.py` | The 3D figure uses the computed displacement field. |

To check that the tests are not trivially passing, change one formula and confirm that tests fail. For example, in
`smartbeam/sections.py`, change the rectangle's `/ 12` to `/ 10` and rerun `pytest -q`. Then revert it.

---

## 3. Hand-check the numbers

Use these inputs and compare the app against your own hand calculation.

### 3a. Default beam (Home or Analysis page)

Structural steel (E = 200 GPa), rectangle 30 × 60 mm, cantilever, L = 1 m, 1 kN at the free end.

| Quantity | Formula | Expected |
|---|---|---|
| I | bh³/12 | 5.4 × 10⁵ mm⁴ |
| Max deflection | PL³/3EI | **3.086 mm** |
| Max bending stress | PL·(h/2)/I | **55.56 MPa** |
| Fixed-end reaction | ΣFy = 0, ΣM = 0 | 1 kN, 1 kN·m |

### 3b. Benchmarks (L = 2 m, same section, P = 1 kN, w = 0.5 kN/m)

Set these up in **Design Studio**, then open **Analysis** and **Numerical FEM**.

| Case | Formula | Expected δmax | Expected Mmax |
|---|---|---|---|
| Cantilever, end load | PL³/3EI | 24.691 mm | 2000 N·m |
| Cantilever, full UDL | wL⁴/8EI | 9.259 mm | 1000 N·m |
| Simply supported, central load | PL³/48EI | 1.543 mm | 500 N·m |
| Simply supported, full UDL | 5wL⁴/384EI | 0.9645 mm | 250 N·m |

Analytical and FEM should agree to better than 0.01 %.

### 3c. Playground demo beam (overhang, partial UDL, moment)

On **Structural Playground**, open *Configuration files · saved_configs/*, choose `final_demo`,
click **Load into canvas**, then **▶ Analyse Structure**.

Beam: pin at x = 0, roller at x = 1.6 m, 1.5 kN down at the tip (x = 2 m), 2 kN/m on 0.4–1.2 m, 0.8 kN·m CW at x = 1 m.

| Quantity | Hand check | Expected |
|---|---|---|
| R_B (roller) | ΣM_A = 0: (1.6·0.8 + 1.5·2 + 0.8)/1.6 | **3.175 kN** up |
| R_A (pin) | ΣFy = 0: 1.6 + 1.5 − 3.175 | **−0.075 kN** (acts downward) |
| M at roller | −1.5 × 0.4 | **−0.6 kN·m** (hogging) |

---

## 4. Manual feature checklist

Tick each item. "Expected" says what correct behaviour looks like.

### Structural Playground (signature feature)
- [ ] Drag **Pinned**, **Roller**, **Point ↓**, **Partial UDL** and **Moment ↻** from the toolbox onto the beam. Each lands where dropped and snaps to the grid.
- [ ] Drag a placed load along the beam. Its x coordinate updates.
- [ ] Drag the end handles of a distributed load. Its start and end change.
- [ ] Click an item. The right panel shows editable fields (magnitude, position, units, direction, angle).
- [ ] Undo, Redo, Duplicate, Delete, Clear and Reset all work. Keyboard Delete and Ctrl/⌘+Z also work.
- [ ] Load each of the 6 presets and analyse. Values match the table in 3b for the matching cases.
- [ ] **▶ Analyse Structure** draws reaction arrows. The SFD, BMD, Deflection and Stress overlays toggle on the canvas.
- [ ] Tick **Live**, then drag a load. Results update after a short pause without pressing Analyse.
- [ ] Delete the pin and analyse. Expected: rejected with an "Unstable … rigid-body" message, not a wrong answer.
- [ ] Make the pin fixed (or tick θ) and analyse. Expected: solved by FEM (`fem-euler`), with a note that statics alone is not enough.
- [ ] Add an **Inclined** load. Expected: solved with frame elements (`fem-frame`) and an axial-force diagram appears.
- [ ] Save a configuration, reload the page, load it again. The canvas is restored.
- [ ] **Use as current design** / **Open in 3D Visualization**. The same beam appears on the other pages.

### Other pages
- [ ] **Design Studio:** switching section type shows only the relevant dimensions. Invalid inputs (negative length, inner diameter ≥ outer) show an error.
- [ ] **Analysis:** reactions, equilibrium residuals ≈ 0, SFD/BMD with maxima marked, stress through depth, slope and deflection.
- [ ] **Numerical FEM:** change the element count. The convergence plot (2→32) and FEM vs analytical % differences are shown.
- [ ] **Beam Theories:** for a short, deep beam (e.g. L = 0.3 m, h = 100 mm), Timoshenko deflection is noticeably larger than Euler–Bernoulli. For a long, slender beam they converge.
- [ ] **3D Visualization:** rotate, zoom, change amplification. The amplification factor is labelled. Max σ and max δ markers are shown.
- [ ] **Safety & Failure:** raise the minimum FOS above the current FOS. Status changes from SAFE to WARNING. Fatigue shows "insufficient data" until you enter σf′, b and Se.
- [ ] **Design Comparison:** save two designs and compare them in the table and charts.
- [ ] **Optimization:** run it. Every row of the constraint table passes. Set an impossible limit (e.g. tiny δallow with small max dimensions). Expected: reported as infeasible with a reason, never "optimized".
- [ ] **Design Advisor:** with an overloaded beam, it names the limiting quantity, proposes a fix, and shows before/after values.
- [ ] **Virtual Lab:** run "Beam height vs stiffness". The fitted slope is about −3.000. Submit a design in a challenge and get a per-constraint grade. Generate a lab report PDF.
- [ ] **Reports:** Build PDF report and download it. It contains SFD/BMD plots and tables. CSV downloads open in a spreadsheet.
- [ ] **About & Methodology:** equations render; validation benchmarks are computed on page load.

---

## 5. Known limitations

These are deliberate scope limits, not bugs. Report anything else you find.

- **Beam theory only:** linear elastic, small deflections, prismatic, homogeneous isotropic material. No 2D or 3D continuum FEA. The 3D view visualises beam-theory results.
- **Yield, not fracture:** SmartBeam predicts the onset of yielding. It does not predict ultimate fracture.
- **Buckling:** Euler elastic buckling of ideal members only. No lateral–torsional or local buckling, no imperfections. Optimized sections can therefore come out thin.
- **Stress:** no stress concentrations, residual stresses or contact stresses. von Mises uses only beam-theory σ and τ.
- **Materials:** library values are approximate. Fatigue needs your own material data; none is invented.
- **Supports:** fixed, pin and roller only. No springs, settlements or internal hinges.
- **Loads:** static only. Distributed loads are transverse; axial or inclined effects use point loads.
- **Playground UI:** the canvas material menu lists library materials only. Labels of loads at the same x can overlap. Over the canvas, Ctrl/⌘ + wheel zooms and a plain wheel scrolls the page.
- **Hosted version:** saved configurations on the hosted deployment are not permanent; they reset when the app restarts. Use the JSON download to keep them.

---

## 6. Where to look in the code

| Concern | File |
|---|---|
| Beam model, validation, stability check | `smartbeam/model.py` |
| Section properties | `smartbeam/sections.py` |
| Statics + integration, solver routing | `smartbeam/analytical.py` |
| Finite elements (EB, Timoshenko, frame) | `smartbeam/fem.py` |
| Yield, von Mises, buckling, fatigue | `smartbeam/safety.py` |
| Optimization + re-verification | `smartbeam/optimize.py` |
| Design advisor | `smartbeam/advisor.py` |
| Virtual Lab calculations | `smartbeam/lab.py` |
| PDF/CSV reports | `smartbeam/report.py` |
| Playground canvas (HTML/JS/SVG) | `smartbeam/playground/frontend/` |
| Canvas ↔ engine conversion | `smartbeam/playground/convert.py` |
| Sign convention, API contract | `docs/ARCHITECTURE.md` |

## 7. Reporting a problem

Open a GitHub issue with: the page, the inputs (or the saved JSON configuration), what you expected with your hand
calculation, and what the app showed.
