"""Numerical 1D FEM: mesh, element/global stiffness, nodal results, FEM vs analytical, mesh convergence."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smartbeam import fem, sections
from smartbeam.ui.plots_core import beam_schematic, fem_convergence, fem_solve, model_caption, require_result
from smartbeam.ui.state import get_model
from smartbeam.ui.theme import COLORS, page_header

C = COLORS
page_header("Numerical FEM", "An independent 1D finite-element solution: discretise, assemble, apply supports, "
            "solve K·U = F, recover forces, and check it against the classical engine.")
model = get_model()
ref = require_result(model)  # validates + gives the reference solution
st.markdown(model_caption(model))

c1, c2, c3 = st.columns([2, 1.2, 1.4], vertical_alignment="bottom")
n_req = c1.select_slider("Number of elements (requested)", [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64],
                         value=8, key="fem_n")
theory = c2.segmented_control("Element theory", ["euler", "timoshenko"], default="euler", required=True,
                              key="fem_theory", format_func={"euler": "Euler–Bernoulli", "timoshenko": "Timoshenko"}.get)
f = fem_solve(model.to_json(indent=None), n_req, theory)
frame = f.element == "frame"
c3.markdown(f"Element type: **{'planar frame (u, v, θ)' if frame else 'beam (v, θ)'}**  \n"
            f"<span style='color:{C['muted']};font-size:0.85rem'>chosen automatically: "
            f"{'axial/inclined loads present' if frame else 'transverse loads only'}</span>", unsafe_allow_html=True)
for w in f.warnings:
    st.warning(w, icon=":material/info:")

k = st.columns(5)
k[0].metric("Elements", f.mesh.n_elements, help="Nodes are always placed at supports and load points, so the "
                                                 "mesh can be finer than requested.")
k[1].metric("Nodes", f.mesh.n_nodes)
k[2].metric("DOFs (total / free)", f"{f.n_dof} / {f.n_free}")
k[3].metric("cond(K_ff)", f"{f.condition_number:.3g}", help="2-norm condition number of the Jacobi-scaled reduced "
                                                             "stiffness matrix; > 1e12 is treated as a mechanism.")
k[4].metric("Max |v| (FEM)", f"{abs(f.v_max.value) * 1e3:.5g} mm")

tabs = st.tabs(["Mesh & matrices", "Nodal results", "FEM vs analytical", "Mesh convergence"])
s, mat = model.section, model.material
E, I, A = mat.E, sections.second_moment(s), sections.area(s)
dpn = f.dofs_per_node
dof_names = ["u", "v", "θ"] if frame else ["v", "θ"]

# --------------------------------------------------------------------------- mesh & matrices
with tabs[0]:
    st.plotly_chart(beam_schematic(model, mesh=f.mesh, height=250), theme=None, key="fem_mesh")
    a, b = st.columns([1, 1.3], gap="large")
    with a:
        st.markdown("##### Connectivity")
        conn = pd.DataFrame({"Element": np.arange(1, f.mesh.n_elements + 1), "Node i": f.mesh.elements[:, 0] + 1,
                             "Node j": f.mesh.elements[:, 1] + 1, "Le (m)": f.mesh.lengths})
        if theory == "timoshenko" and f.phi is not None:
            conn["φ = 12EI/(κGA Le²)"] = f.phi
        st.dataframe(conn, hide_index=True, height=260, column_config={
            "Le (m)": st.column_config.NumberColumn(format="%.5g"),
            "φ = 12EI/(κGA Le²)": st.column_config.NumberColumn(format="%.4g")})
        fixed = fem.restrained_dofs(model, f.mesh, frame)
        st.markdown("##### Boundary conditions")
        st.caption(", ".join(f"node {d // dpn + 1}: {dof_names[d % dpn]} = 0" for d in sorted(fixed)) or "none")
    with b:
        st.markdown("##### Element stiffness matrix")
        if theory == "euler":
            st.latex(r"\mathbf{k}_e=\frac{EI}{L_e^3}\begin{bmatrix}12&6L_e&-12&6L_e\\6L_e&4L_e^2&-6L_e&2L_e^2\\"
                     r"-12&-6L_e&12&-6L_e\\6L_e&2L_e^2&-6L_e&4L_e^2\end{bmatrix},\quad "
                     r"\mathbf{d}_e=[v_1,\theta_1,v_2,\theta_2]^T")
        else:
            st.latex(r"\mathbf{k}_e=\frac{EI}{(1+\phi)L_e^3}\begin{bmatrix}12&6L_e&-12&6L_e\\"
                     r"6L_e&(4+\phi)L_e^2&-6L_e&(2-\phi)L_e^2\\-12&-6L_e&12&-6L_e\\"
                     r"6L_e&(2-\phi)L_e^2&-6L_e&(4+\phi)L_e^2\end{bmatrix},\quad \phi=\frac{12EI}{\kappa G A L_e^2}")
        if frame:
            st.caption("Frame element: the axial block EA/Le·[[1, −1], [−1, 1]] on (u₁, u₂) is added to the bending "
                       "block, giving a 6×6 matrix on [u₁, v₁, θ₁, u₂, v₂, θ₂].")
        Le0 = float(f.mesh.lengths[0])
        if theory == "euler":
            kb = fem.eb_element_stiffness(E, I, Le0)
        else:
            kb = fem.timoshenko_element_stiffness(E, I, Le0, mat.G, A, sections.shear_correction_factor(s, mat.nu))
        ke = fem.frame_element_stiffness(E, A, I, Le0, phi=float(f.phi[0]) if f.phi is not None else 0.0) if frame else kb
        lab = (["u₁", "v₁", "θ₁", "u₂", "v₂", "θ₂"] if frame else ["v₁", "θ₁", "v₂", "θ₂"])
        st.caption(f"Element 1 (Le = {Le0:.4g} m), SI units (N/m, N, N·m):")
        st.dataframe(pd.DataFrame(ke, index=lab, columns=lab).style.format("{:.4e}"))

    st.markdown("##### Global stiffness matrix K and load vector F (before boundary conditions)")
    K, F = fem.assemble(model, f.mesh, theory=theory, frame=frame)
    # Display only: entries that should cancel (e.g. ±6EI/Le² from neighbouring elements) leave ~1e-8 round-off
    # next to ~1e9 terms; show them as the exact zeros they represent.
    K = np.where(np.abs(K) < 1e-12 * np.abs(K).max(), 0.0, K)
    labels = [f"{dof_names[i % dpn]}{i // dpn + 1}" for i in range(len(F))]
    mag = np.where(K != 0, np.log10(np.abs(K) + 1e-300), np.nan)
    hm = go.Figure(go.Heatmap(z=mag, x=labels, y=labels, colorscale="Cividis", hoverongaps=False,
                              customdata=K, hovertemplate="K[%{y}, %{x}] = %{customdata:.4e}<extra></extra>",
                              colorbar=dict(title="log₁₀|K|")))
    hm.update_layout(height=420 if len(F) <= 40 else 520, yaxis=dict(autorange="reversed", scaleanchor="x"),
                     title=f"{len(F)}×{len(F)} global K — banded because each element couples only its two nodes")
    st.plotly_chart(hm, theme=None, key="fem_K")
    if len(F) <= 24:
        st.dataframe(pd.DataFrame(K, index=labels, columns=labels).style.format("{:.3e}"))
        st.dataframe(pd.DataFrame({"DOF": labels, "F (N or N·m)": F}).set_index("DOF").T.style.format("{:.4g}"))
    else:
        st.caption(f"Numeric table shown for meshes with ≤ 24 DOFs (this one has {len(F)}); hover the heatmap for "
                   "individual entries.")

# --------------------------------------------------------------------------- nodal results
with tabs[1]:
    nod = pd.DataFrame({"Node": np.arange(1, f.mesh.n_nodes + 1), "x (m)": f.mesh.nodes,
                        "v (mm)": f.v_nodes * 1e3, "θ (mrad)": f.theta_nodes * 1e3})
    if frame:
        nod.insert(2, "u (mm)", f.u_nodes * 1e3)
    st.dataframe(nod, hide_index=True, height=320,
                 column_config={c: st.column_config.NumberColumn(format="%.6g") for c in nod.columns[1:]})
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Reactions (R = K·U − F at restrained DOFs)")
        st.dataframe(pd.DataFrame([{"Support": r.support_index + 1, "Type": r.type, "x (m)": r.x,
                                    "Rx (kN)": r.Fx / 1e3, "Ry (kN)": r.Fy / 1e3, "M (kN·m)": r.M / 1e3}
                                   for r in f.reactions]), hide_index=True)
    with b:
        st.markdown("##### Element end forces (k·d − f, global axes)")
        cols = [f"{n}{e}" for e in ("₁", "₂") for n in (["Fx", "Fy", "M"] if frame else ["Fy", "M"])]
        st.dataframe(pd.DataFrame(f.element_end_forces, columns=cols,
                                  index=pd.Index(np.arange(1, f.mesh.n_elements + 1), name="Element")
                                  ).style.format("{:.4g}"), height=240)
    st.download_button("Download nodal results (CSV)", nod.to_csv(index=False), file_name="smartbeam_fem_nodes.csv",
                       mime="text/csv", icon=":material/download:")

# --------------------------------------------------------------------------- FEM vs analytical
with tabs[2]:
    independent = ref.method == "analytical"
    if independent:
        st.success("Reference: the classical engine (statics + integration). It shares no code with the FEM solver, "
                   "so agreement is a genuine cross-check.", icon=":material/verified:")
    else:
        st.info(f"No independent classical solution exists for this configuration ({ref.stability.message}) The "
                f"reference below is the {ref.method} solution with 40 elements, i.e. the same method on a finer "
                "mesh: differences only show mesh dependence.", icon=":material/info:")
    if theory == "timoshenko":
        st.warning("You are comparing Timoshenko elements with the Euler–Bernoulli reference: part of the "
                   "difference is physical shear deformation, not numerical error.", icon=":material/science:")
    fig = go.Figure([
        go.Scatter(x=ref.x, y=ref.v * 1e3, mode="lines", name="reference", line=dict(color=C["muted"], width=5)),
        go.Scatter(x=f.x, y=f.v * 1e3, mode="lines", name=f"FEM ({f.mesh.n_elements} el.)",
                   line=dict(color=C["cyan"], width=2, dash="dash")),
        go.Scatter(x=f.mesh.nodes, y=f.v_nodes * 1e3, mode="markers", name="FEM nodes",
                   marker=dict(color=C["amber"], size=8))])
    fig.update_layout(height=320, title="Deflection v (mm)", xaxis_title="x (m)", yaxis_title="v (mm)",
                      legend=dict(orientation="h", y=1.15, x=1, xanchor="right"))
    st.plotly_chart(fig, theme=None, key="fem_v")
    fig = go.Figure([
        go.Scatter(x=ref.x, y=ref.M / 1e3, mode="lines", name="reference", line=dict(color=C["muted"], width=5)),
        go.Scatter(x=f.x, y=f.M / 1e3, mode="lines", name="FEM", line=dict(color=C["amber"], width=2, dash="dash"))])
    fig.update_layout(height=300, title="Bending moment M (kN·m)", xaxis_title="x (m)", yaxis_title="M (kN·m)",
                      legend=dict(orientation="h", y=1.15, x=1, xanchor="right"))
    st.plotly_chart(fig, theme=None, key="fem_M")

    def pct(a, b):
        return 100 * abs(abs(a) - abs(b)) / abs(b) if b else (0.0 if a == 0 else np.inf)

    rmax = lambda rs: max((abs(r.Fy) for r in rs), default=0.0)  # noqa: E731
    rows = [("Max |v| (mm)", f.v_max.value * 1e3, ref.v_max.value * 1e3),
            ("Max |θ| (mrad)", f.theta_max.value * 1e3, ref.theta_max.value * 1e3),
            ("Max |M| (kN·m)", f.M_max.value / 1e3, ref.M_max.value / 1e3),
            ("Max |V| (kN)", f.V_max.value / 1e3, ref.V_max.value / 1e3),
            ("Max |σ| (MPa)", f.sigma_max.value / 1e6, ref.sigma_max.value / 1e6),
            ("Max |Ry| (kN)", rmax(f.reactions) / 1e3, rmax(ref.reactions) / 1e3)]
    cmp_df = pd.DataFrame([{"Quantity": q, "FEM": abs(a), "Reference": abs(b), "Difference (%)": pct(a, b)}
                           for q, a, b in rows])
    st.dataframe(cmp_df, hide_index=True, column_config={
        "FEM": st.column_config.NumberColumn(format="%.6g"), "Reference": st.column_config.NumberColumn(format="%.6g"),
        "Difference (%)": st.column_config.NumberColumn(format="%.3e")})
    worst = cmp_df["Difference (%)"].max()
    if theory == "euler":
        st.caption(
            f"Largest difference: {worst:.2e} %. Euler–Bernoulli elements with consistent loads give exact nodal "
            "values for prismatic beams, so differences at round-off level are expected. A peak that lies between "
            "nodes is read from the cubic interpolation, and the slope there can differ slightly on coarse meshes "
            "under distributed loads. That is a real interpolation error and it shrinks as the mesh is refined.")

# --------------------------------------------------------------------------- convergence
with tabs[3]:
    cv = fem_convergence(model.to_json(indent=None), theory)
    n = np.array(cv.n_elements)
    series = [("v_max", "|v|max", C["cyan"]), ("theta_max", "|θ|max", C["violet"]), ("M_max", "|M|max (recovered)",
              C["amber"]), ("sigma_max", "|σ|max", C["red"]), ("M_fe", "|EI v″|max (raw field)", C["green"])]
    fig = go.Figure()
    if cv.error_pct:
        floor = 1e-13
        for key, name, col in series:
            e = np.maximum(np.asarray(cv.error_pct[key]), floor)
            fig.add_trace(go.Scatter(x=n, y=e, mode="lines+markers", name=name, line=dict(color=col, width=2.2),
                                     hovertemplate="%{x} elements: %{y:.3e} %<extra>" + name + "</extra>"))
        fig.update_layout(title=f"Error vs closed-form reference ({cv.theory})", yaxis_type="log",
                          yaxis_title="error (%)", yaxis_exponentformat="power")
        fig.add_hrect(y0=floor / 10, y1=1e-9, fillcolor=C["muted"], opacity=0.08, line_width=0,
                      annotation_text="round-off level", annotation_position="bottom right",
                      annotation_font_color=C["muted"])
    else:
        vals = {"v_max": cv.v_max, "theta_max": cv.theta_max, "M_max": cv.M_max, "sigma_max": cv.sigma_max,
                "M_fe": cv.M_fe_max}
        for key, name, col in series:
            y = np.asarray(vals[key], float)
            fig.add_trace(go.Scatter(x=n, y=y / y[-1] if y[-1] else y, mode="lines+markers", name=name,
                                     line=dict(color=col, width=2.2)))
        fig.update_layout(title="Values normalised by the finest mesh (no closed-form reference applies)",
                          yaxis_title="value / value(finest)")
    fig.update_layout(height=400, xaxis=dict(type="log", tickvals=n, ticktext=[str(i) for i in n],
                                             title="number of elements"),
                      legend=dict(orientation="h", y=-0.22))
    st.plotly_chart(fig, theme=None, key="fem_conv")
    tab = pd.DataFrame({"Requested": cv.n_requested, "Elements": n, "|v|max (mm)": cv.v_max * 1e3,
                        "|θ|max (mrad)": cv.theta_max * 1e3, "|M|max (kN·m)": cv.M_max / 1e3,
                        "|EI v″|max (kN·m)": cv.M_fe_max / 1e3, "|σ|max (MPa)": cv.sigma_max / 1e6})
    st.dataframe(tab, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.8g")
                                                      for c in tab.columns[2:]})
    if cv.reference:
        st.caption("Closed-form reference: |v| = {:.6g} mm, |θ| = {:.6g} mrad, |M| = {:.6g} kN·m, |σ| = {:.6g} MPa."
                   .format(cv.reference["v_max"] * 1e3, cv.reference["theta_max"] * 1e3,
                           cv.reference["M_max"] / 1e3, cv.reference["sigma_max"] / 1e6))
    with st.expander("How to read this convergence study", expanded=True):
        st.markdown(cv.note)
