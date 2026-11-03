"""About & Methodology: purpose, governing equations, live validation benchmarks, stack, team placeholders, limits."""
import sys

import numpy as np
import pandas as pd
import plotly
import scipy
import streamlit as st

from smartbeam import fem
from smartbeam.analytical import analyze
from smartbeam.materials import get_material
from smartbeam.model import BeamModel, DistributedLoad, PointLoad, Section, Support
from smartbeam.ui.theme import COLORS, page_header

C = COLORS
page_header("About & Methodology", "What SmartBeam computes, how, and how far the results can be trusted.")

# --------------------------------------------------------------------------- project details (editable)
with st.container(border=True):
    st.markdown("##### Project details")
    st.caption("Editable placeholders: enter your own details (kept for this session and available to reports). "
               "SmartBeam never invents names.")
    a, b = st.columns(2)
    a.text_input("Team members", value="[Team members]", key="about_team", persist_state="session")
    b.text_input("Institute / department", value="[Institute]", key="about_institute", persist_state="session")
    a.text_input("Project guide", value="[Guide]", key="about_guide", persist_state="session")
    b.text_input("Course", value="[Course · Solid Mechanics]", key="about_course", persist_state="session")

tabs = st.tabs(["Purpose", "Governing equations", "Validation benchmarks", "Technology", "Limitations"])

with tabs[0]:
    st.markdown(f"""
**SmartBeam 2.0** is a second-year Solid Mechanics project. It turns the beam chapter of the course into an
interactive engineering tool: **Understand. Simulate. Predict. Optimize.**

- **Understand:** reactions, shear force and bending moment diagrams, bending and shear stress, strain, slope and
  deflection, with sign conventions and equations shown next to the results.
- **Simulate:** an independent 1D finite-element solver (Euler–Bernoulli, Timoshenko and planar frame elements)
  that is checked against the classical solution, with mesh-convergence studies.
- **Predict:** onset of yielding, combined stress (von Mises), Euler buckling and stress-life fatigue.
- **Optimize:** constrained sizing (SciPy SLSQP), Pareto trade-offs and a rule-based design advisor.

**Solid Mechanics topics covered:** equilibrium of rigid bodies and free-body diagrams; statical determinacy; shear
force and bending moment; flexure formula and the neutral axis; section properties (A, I, Z, r, Q); shear stress
distribution; Hooke's law and strain; the elastic curve (double integration); shear deformation; failure theories
and factor of safety; column buckling; fatigue.

**Sign convention used throughout:** x from the left end, y up; loads +x/+y; moments counter-clockwise positive;
V = ΣFy to the left of the cut (dV/dx = q); M sagging positive (dM/dx = V); N tension positive; EI v″ = M with v up;
σ = N/A − My/I (tension positive).
""")

with tabs[1]:
    st.markdown("##### Section properties")
    st.latex(r"\text{rect: }A=bh,\ I=\frac{bh^3}{12}\qquad \text{solid: }A=\frac{\pi d^2}{4},\ I=\frac{\pi d^4}{64}"
             r"\qquad \text{hollow: }A=\frac{\pi(D^2-d^2)}{4},\ I=\frac{\pi(D^4-d^4)}{64}")
    st.latex(r"Z=\frac{I}{c},\qquad r=\sqrt{I/A},\qquad Q(y)=\int_{y}^{c} \eta\, b(\eta)\, d\eta")
    st.markdown("##### Equilibrium and internal forces")
    st.latex(r"\sum F_x=0,\quad \sum F_y=0,\quad \sum M=0,\qquad \frac{dV}{dx}=q(x)=-w(x),\qquad \frac{dM}{dx}=V")
    st.markdown("##### Euler–Bernoulli beam")
    st.latex(r"\sigma=\frac{N}{A}-\frac{My}{I},\qquad \varepsilon=\frac{\sigma}{E},\qquad \tau=\frac{VQ}{Ib},\qquad "
             r"EI\frac{d^2v}{dx^2}=M,\qquad EI\frac{d^4v}{dx^4}=q")
    st.markdown("##### Timoshenko beam")
    st.latex(r"M=EI\frac{d\psi}{dx},\qquad V=\kappa GA\left(\frac{dv}{dx}-\psi\right),\qquad G=\frac{E}{2(1+\nu)},"
             r"\qquad \phi=\frac{12EI}{\kappa G A L_e^2}")
    st.markdown("##### Finite-element formulation (1D line elements only)")
    st.latex(r"\mathbf{k}^{EB}_e=\frac{EI}{L_e^3}\begin{bmatrix}12&6L_e&-12&6L_e\\6L_e&4L_e^2&-6L_e&2L_e^2\\"
             r"-12&-6L_e&12&-6L_e\\6L_e&2L_e^2&-6L_e&4L_e^2\end{bmatrix}\quad"
             r"\mathbf{k}^{T}_e=\frac{EI}{(1+\phi)L_e^3}\begin{bmatrix}12&6L_e&-12&6L_e\\6L_e&(4+\phi)L_e^2&-6L_e&"
             r"(2-\phi)L_e^2\\-12&-6L_e&12&-6L_e\\6L_e&(2-\phi)L_e^2&-6L_e&(4+\phi)L_e^2\end{bmatrix}")
    st.latex(r"\mathbf{k}^{frame}_e=\underbrace{\frac{EA}{L_e}\begin{bmatrix}1&-1\\-1&1\end{bmatrix}}_{(u_1,\,u_2)}"
             r"\;\oplus\;\underbrace{\mathbf{k}^{bend}_e}_{(v_1,\theta_1,v_2,\theta_2)}\quad(6\times6),\qquad "
             r"\mathbf{f}_e=\int_0^{L_e}\mathbf{N}^T q\,dx \;\overset{\text{UDL}}{=}\;\frac{qL_e}{12}"
             r"\begin{bmatrix}6\\L_e\\6\\-L_e\end{bmatrix}")
    st.latex(r"\mathbf{K}=\sum_e \mathbf{A}_e^T\mathbf{k}_e\mathbf{A}_e,\qquad \mathbf{K}_{ff}\mathbf{U}_f=\mathbf{F}_f,"
             r"\qquad \mathbf{R}=\mathbf{K}\mathbf{U}-\mathbf{F},\qquad v(\xi)=N_1v_1+N_2\theta_1+N_3v_2+N_4\theta_2")
    st.caption("Hermite cubic shape functions N₁…N₄ for EB; interdependent interpolation for Timoshenko (exact for "
               "prismatic beams, free of shear locking). A mechanism is detected from the rank/condition number of "
               "K_ff (limit 10¹²).")
    st.markdown("##### Failure criteria")
    st.latex(r"\text{FOS}=\frac{\sigma_y}{|\sigma|_{\max}},\qquad \sigma_{vm}=\sqrt{\sigma^2+3\tau^2},\qquad "
             r"P_{cr}=\frac{\pi^2EI_{\min}}{(KL)^2}")
    st.latex(r"\sigma_a=\tfrac{\sigma_{\max}-\sigma_{\min}}{2},\quad \sigma_m=\tfrac{\sigma_{\max}+\sigma_{\min}}{2},"
             r"\quad \frac{\sigma_a}{S_e}+\frac{\sigma_m}{S_u}=\frac{1}{n}\ (\text{Goodman}),\quad "
             r"\sigma_{ar}=\sigma_f'(2N)^b\ (\text{Basquin})")
    st.markdown("##### Optimization")
    st.latex(r"\min_{\mathbf{d}}\; f(\mathbf{d})=A(\mathbf{d})\,L\ \ (\text{or } \rho AL,\ c\rho AL)\quad"
             r"\text{s.t.}\quad |v|_{\max}\le\delta_{allow},\quad \frac{\sigma_y}{|\sigma|_{\max}}\ge \text{FOS}_{\min},"
             r"\quad \mathbf{d}_{lo}\le\mathbf{d}\le\mathbf{d}_{hi}")
    st.caption("Solved with SciPy SLSQP from several starting points; every candidate is re-analysed by the engine and "
               "only a design that passes all constraints is labelled OPTIMIZED. A Euler-buckling constraint is "
               "added automatically when the member is in axial compression.")


# --------------------------------------------------------------------------- live validation
@st.cache_data(show_spinner="Running validation benchmarks…")
def benchmarks() -> pd.DataFrame:
    steel, sec, L, P, w = get_material("structural_steel"), Section.rect(0.030, 0.060), 2.0, 1000.0, 500.0
    cases = {
        "Cantilever, end load P": ([Support("fixed", 0.0)], [PointLoad(L, P)]),
        "Cantilever, full UDL w": ([Support("fixed", 0.0)], [DistributedLoad(0.0, L, w, w)]),
        "Simply supported, central P": ([Support("pin", 0.0), Support("roller", L)], [PointLoad(L / 2, P)]),
        "Simply supported, full UDL w": ([Support("pin", 0.0), Support("roller", L)], [DistributedLoad(0.0, L, w, w)]),
    }
    rows = []
    for name, (sup, loads) in cases.items():
        m = BeamModel(L, sec, steel, sup, loads, name)
        r = analyze(m)
        f = fem.solve(m, n_elements=8)
        tb = r.textbook
        fem_vals = {"R": max(abs(x.Fy) for x in f.reactions), "V_max": abs(f.V_max.value),
                    "M_max": abs(f.M_max.value), "theta_max": abs(f.theta_max.value),
                    "delta_max": abs(f.v_max.value), "sigma_max": abs(f.sigma_max.value)}
        for k in ("R", "M_max", "delta_max", "sigma_max"):
            rows.append({"Benchmark": name, "Quantity": k, "Formula": tb.formulas[k], "Textbook": tb.textbook[k],
                         "Analytical err (%)": 100 * tb.rel_error[k],
                         "FEM (8 el.) err (%)": 100 * abs(fem_vals[k] - tb.textbook[k]) / tb.textbook[k]})
    # Indeterminate check (not a textbook-guarded case, solved by FEM only): propped cantilever, full UDL.
    m = BeamModel(L, sec, steel, [Support("fixed", 0.0), Support("roller", L)], [DistributedLoad(0.0, L, w, w)],
                  "Propped cantilever, UDL")
    r = analyze(m)
    for k, formula, exact, num in [("R_roller", "3wL/8", 3 * w * L / 8, abs(r.reactions[1].Fy)),
                                   ("M_fixed", "wL²/8", w * L**2 / 8, abs(r.reactions[0].M))]:
        rows.append({"Benchmark": "Propped cantilever, full UDL (indeterminate)", "Quantity": k, "Formula": formula,
                     "Textbook": exact, "Analytical err (%)": np.nan,
                     "FEM (8 el.) err (%)": 100 * abs(num - exact) / exact})
    return pd.DataFrame(rows)


with tabs[2]:
    st.markdown("Computed live by the engine on every visit (structural steel, rect 30 × 60 mm, L = 2 m, "
                "P = 1 kN, w = 0.5 kN/m). Values in SI (N, N·m, m, Pa).")
    df = benchmarks()
    st.dataframe(df, hide_index=True, column_config={
        "Textbook": st.column_config.NumberColumn(format="%.6g"),
        "Analytical err (%)": st.column_config.NumberColumn(format="%.1e"),
        "FEM (8 el.) err (%)": st.column_config.NumberColumn(format="%.1e")})
    worst = np.nanmax(df[["Analytical err (%)", "FEM (8 el.) err (%)"]].to_numpy())
    st.success(f"Largest relative error across all benchmarks: {worst:.1e} % (acceptance target 0.1 %).",
               icon=":material/verified:")
    st.caption("The automated test suite (pytest) repeats these checks and adds section properties, element stiffness "
               "symmetry/rank, assembly, boundary conditions, Timoshenko → Euler–Bernoulli as L/h grows, frame axial "
               "response, buckling, fatigue, optimizer constraint satisfaction and advisor fixes.")

with tabs[3]:
    st.dataframe(pd.DataFrame([
        {"Component": "Python", "Version": sys.version.split()[0], "Role": "language"},
        {"Component": "NumPy", "Version": np.__version__, "Role": "arrays, linear algebra, integration"},
        {"Component": "SciPy", "Version": scipy.__version__, "Role": "sparse-free linear solve, SLSQP optimizer"},
        {"Component": "Pandas", "Version": pd.__version__, "Role": "tables, CSV export"},
        {"Component": "Plotly", "Version": plotly.__version__, "Role": "interactive 2D/3D charts"},
        {"Component": "Streamlit", "Version": st.__version__, "Role": "user interface"},
        {"Component": "Matplotlib + ReportLab", "Version": "—", "Role": "PDF report figures and layout"},
        {"Component": "pytest", "Version": "—", "Role": "automated verification"},
    ]), hide_index=True)
    st.markdown("Runs entirely locally: no cloud services, databases or paid APIs. The calculation engine "
                "(`smartbeam/`) has no user-interface code; the pages only call it and display its results.")

with tabs[4]:
    st.markdown("""
- **Linear elastic, small deflection** theory throughout; no plasticity beyond first yield, no large-displacement or
  geometric-stiffness effects (except the separate Euler buckling check).
- **Prismatic, straight, homogeneous beams**: constant section and material along the span.
- **1D line elements only** (beam, Timoshenko beam, planar frame). There is no 2D/3D continuum FEA, so local effects
  such as stress concentrations at holes, notches, supports and load points are not captured.
- **Bending about one axis** (vertical loads in the plane of symmetry); no torsion, no lateral-torsional buckling and
  no local/wall buckling of thin tubes.
- **Shear stress** from elementary theory τ = VQ/(Ib); for circular sections this is the chord average.
- **Buckling** is the ideal Euler load with K factors for ideal end conditions; imperfections lower the real capacity,
  and stocky members are flagged (Johnson estimate shown).
- **Fatigue** is an introductory stress-life estimate that uses only the data you enter (no notch, size, surface or
  reliability factors).
- **Yield is not fracture**: SmartBeam predicts the onset of yielding, not the ultimate/collapse load.
- **Material data** in the library are approximate typical values; costs are rough INR/kg estimates. Both can be edited.
- The 3D view is a visualisation of beam theory, not a continuum stress solution.
""")
