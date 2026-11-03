"""Beam Design Studio: section, material library, supports and loads -> shared BeamModel (auto-saved when valid)."""
import math

import pandas as pd
import streamlit as st

from smartbeam import materials, units
from smartbeam.analytical import analyze
from smartbeam.model import (BeamModel, DistributedLoad, Material, ModelError, MomentLoad, PointLoad, Section,
                             Support, UnstableStructureError, check_stability, validate)
from smartbeam.ui.plots_core import (METHOD_LABEL, beam_schematic, section_figure, section_props_metrics)
from smartbeam.ui.state import default_model, get_model, save_design, saved_designs, set_model
from smartbeam.ui.theme import page_header

page_header("Beam Design Studio", "Define the beam once; every other page analyses this design. "
            "Valid changes are saved to the workspace automatically.")

SS = st.session_state
SUPPORT_PRESETS = {
    "cant_left": ("Cantilever · fixed at x = 0", lambda L: [Support("fixed", 0.0)]),
    "cant_right": ("Cantilever · fixed at x = L", lambda L: [Support("fixed", L)]),
    "ss": ("Simply supported · pin at 0, roller at L", lambda L: [Support("pin", 0.0), Support("roller", L)]),
}
CUSTOM_SUP = "custom"
LOAD_TYPES = ["Point load", "Distributed", "Moment"]
DIRECTIONS = ["Down", "Up", "Axial →", "Axial ←", "Inclined", "CW", "CCW"]
CUSTOM_MAT = "Custom material…"
LIB_COLS = ["Material", "E (GPa)", "Yield σy (MPa)", "UTS (MPa)", "Poisson ν", "Density (kg/m³)", "Cost (₹/kg)"]


def r10(v):
    return None if v is None else float(f"{v:.10g}")


# --------------------------------------------------------------------------- material library (session copy)
def library_df() -> pd.DataFrame:
    rows = [[m.name, m.E / 1e9, m.yield_strength / 1e6, m.uts / 1e6 if m.uts else None, m.nu, m.density,
             m.cost_per_kg] for m in materials.MATERIALS.values()]
    return pd.DataFrame(rows, columns=LIB_COLS, index=list(materials.MATERIALS)).astype({c: float for c in LIB_COLS[1:]})


def row_material(row) -> Material:
    uts = row["UTS (MPa)"]
    cost = row["Cost (₹/kg)"]
    return Material(name=str(row["Material"]), E=float(row["E (GPa)"]) * 1e9,
                    yield_strength=float(row["Yield σy (MPa)"]) * 1e6, nu=float(row["Poisson ν"]),
                    density=float(row["Density (kg/m³)"]), uts=None if pd.isna(uts) else float(uts) * 1e6, cost_per_kg=None if pd.isna(cost) else float(cost),
                    approximate=True, note=materials.APPROX_NOTE)


if "ds_lib" not in SS:
    SS["ds_lib"] = library_df()


# --------------------------------------------------------------------------- model -> widget state
def load_table(model: BeamModel, u) -> pd.DataFrame:
    rows = []
    for ld in model.loads:
        if isinstance(ld, PointLoad):
            a = ld.angle_deg % 360
            dirn, ang = {270.0: ("Down", None), 90.0: ("Up", None), 0.0: ("Axial →", None),
                         180.0: ("Axial ←", None)}.get(round(a, 9), ("Inclined", ld.angle_deg))
            rows.append(["Point load", units.from_si(ld.x, u["len"]), None, units.from_si(ld.magnitude, u["force"]),
                         None, dirn, ang])
        elif isinstance(ld, DistributedLoad):
            rows.append(["Distributed", units.from_si(ld.x1, u["len"]), units.from_si(ld.x2, u["len"]),
                         units.from_si(ld.w1, u["dist"]),
                         None if ld.w1 == ld.w2 else units.from_si(ld.w2, u["dist"]),
                         ld.direction.capitalize(), None])
        else:
            rows.append(["Moment", units.from_si(ld.x, u["len"]), None, units.from_si(abs(ld.M), u["mom"]), None,
                         "CCW" if ld.M > 0 else "CW", None])
    rows = [[r[0]] + [r10(v) if isinstance(v, float) else v for v in r[1:]] for r in rows]
    return pd.DataFrame(rows, columns=["Type", "Position / start", "End", "Magnitude", "End magnitude",
                                       "Direction", "Angle (°)"]).astype(
        {"Position / start": float, "End": float, "Magnitude": float, "End magnitude": float, "Angle (°)": float})


def sync_from(model: BeamModel) -> None:
    """Push the shared model into this page's widget state."""
    s, m = model.section, model.material
    SS["ds_name"], SS["ds_L"], SS["ds_shape"] = model.name, r10(model.L), s.shape
    depth_mm = s.depth * 1e3
    SS["ds_b"] = r10(s.b * 1e3) if s.b else r10(depth_mm / 2)
    SS["ds_h"] = r10(s.h * 1e3) if s.h else r10(depth_mm)
    SS["ds_d"] = r10(s.d * 1e3) if s.shape == "solid_circ" else r10(depth_mm)
    SS["ds_D"] = r10(s.D * 1e3) if s.D else r10(depth_mm)
    SS["ds_di"] = r10(s.d * 1e3) if s.shape == "hollow_circ" else r10(0.8 * depth_mm)

    lib = SS["ds_lib"]
    match = [k for k, row in lib.iterrows()
             if row["Material"] == m.name and math.isclose(row["E (GPa)"] * 1e9, m.E, rel_tol=1e-9)
             and math.isclose(row["Yield σy (MPa)"] * 1e6, m.yield_strength, rel_tol=1e-9)]
    SS["ds_mat"] = lib.loc[match[0], "Material"] if match else CUSTOM_MAT
    SS["ds_c_name"] = m.name if not match else "My material"
    SS["ds_c_E"], SS["ds_c_sy"] = r10(m.E / 1e9), r10(m.yield_strength / 1e6)
    SS["ds_c_uts"] = r10(m.uts / 1e6) if m.uts else None
    SS["ds_c_nu"], SS["ds_c_rho"], SS["ds_c_cost"] = r10(m.nu), r10(m.density), m.cost_per_kg

    sup = sorted(model.supports, key=lambda sp: sp.x)
    SS["ds_sup"] = CUSTOM_SUP
    for key, (_, make) in SUPPORT_PRESETS.items():
        ref = make(model.L)
        if len(ref) == len(sup) and all(a.type == b.type and abs(a.x - b.x) <= 1e-9 * max(1, model.L)
                                        for a, b in zip(ref, sup)):
            SS["ds_sup"] = key
    SS["ds_custom_supports"] = [(sp.type, sp.x) for sp in model.supports]
    SS["ds_loads_base"] = load_table(model, units_now())
    SS["ds_ver"] = SS.get("ds_ver", 0) + 1
    SS["ds_synced"] = model.to_json()


def units_now() -> dict:
    return {"len": SS.get("ds_u_len", "m"), "force": SS.get("ds_u_force", "kN"),
            "dist": SS.get("ds_u_dist", "kN/m"), "mom": SS.get("ds_u_mom", "kN·m")}


shared = get_model()
if SS.get("ds_synced") != shared.to_json() or "ds_L" not in SS:
    sync_from(shared)

# --------------------------------------------------------------------------- upload / reset toolbar
with st.container(border=True):
    t1, t2, t3 = st.columns([1, 1, 1.6], vertical_alignment="bottom")
    if t1.button("Reset to default beam", icon=":material/restart_alt:", width="stretch"):
        set_model(default_model())
        st.rerun()
    t2.download_button("Download config (JSON)", shared.to_json(), file_name=f"{shared.name or 'beam'}.json",
                       mime="application/json", icon=":material/download:", width="stretch")
    up = t3.file_uploader("Load config (JSON)", type=["json"], label_visibility="collapsed")
    if up is not None and SS.get("ds_uploaded") != up.file_id:
        SS["ds_uploaded"] = up.file_id
        try:
            loaded = BeamModel.from_json(up.getvalue().decode("utf-8"))
            errs = validate(loaded)
            if errs:
                st.error("Config not loaded: " + "; ".join(errs))
            else:
                set_model(loaded)
                st.rerun()
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Config not loaded: not a valid SmartBeam JSON file ({exc}).")

left, right = st.columns([1.25, 1], gap="large")
errors: list[str] = []

with left:
    tab_geo, tab_mat, tab_sup, tab_load = st.tabs(["Geometry & section", "Material", "Supports", "Loads"])

    # ------------------------------------------------------------------ geometry & section
    with tab_geo:
        c1, c2 = st.columns([2, 1])
        c1.text_input("Design name", key="ds_name")
        c2.number_input("Span L (m)", min_value=0.01, max_value=100.0, step=0.1, format="%.4g", key="ds_L")
        st.segmented_control("Cross-section", ["rect", "solid_circ", "hollow_circ"], key="ds_shape", required=True,
                             format_func={"rect": "Rectangular", "solid_circ": "Solid circular",
                                          "hollow_circ": "Hollow circular"}.get)
        shape = SS["ds_shape"]
        dim = dict(min_value=0.1, max_value=5000.0, step=1.0, format="%.4g")
        if shape == "rect":
            a, b = st.columns(2)
            a.number_input("Width b (mm)", key="ds_b", **dim)
            b.number_input("Depth h (mm)", key="ds_h", help="Bending is about the horizontal axis; h is vertical.",
                           **dim)
            section = Section.rect(SS["ds_b"] / 1e3, SS["ds_h"] / 1e3)
        elif shape == "solid_circ":
            st.number_input("Diameter d (mm)", key="ds_d", **dim)
            section = Section.solid_circ(SS["ds_d"] / 1e3)
        else:
            a, b = st.columns(2)
            a.number_input("Outer diameter D (mm)", key="ds_D", **dim)
            b.number_input("Inner diameter d (mm)", key="ds_di", **dim)
            section = Section.hollow_circ(SS["ds_D"] / 1e3, SS["ds_di"] / 1e3)
            if SS["ds_di"] >= SS["ds_D"]:
                errors.append("Hollow section: inner diameter must be smaller than outer diameter.")
        if not errors:
            s1, s2 = st.columns([1, 2.2], vertical_alignment="center")
            s1.plotly_chart(section_figure(section, height=190), theme=None, key="ds_secfig")
            with s2:
                section_props_metrics(section, 0.3)
                st.caption("Section properties about the horizontal centroidal (bending) axis.")

    # ------------------------------------------------------------------ material
    with tab_mat:
        st.caption("Library values are **approximate typical figures** for each grade (edit them to match your "
                   "supplier data). Costs are rough INR/kg estimates. Fatigue properties are never pre-filled.")
        edited = st.data_editor(
            SS["ds_lib"], key=f"ds_lib_editor_{SS.get('ds_lib_ver', 0)}", num_rows="fixed", hide_index=True,
            disabled=["Material"],
            column_config={
                "E (GPa)": st.column_config.NumberColumn(min_value=0.001, format="%.4g", required=True),
                "Yield σy (MPa)": st.column_config.NumberColumn(min_value=0.001, format="%.4g", required=True),
                "UTS (MPa)": st.column_config.NumberColumn(min_value=0.001, format="%.4g"),
                "Poisson ν": st.column_config.NumberColumn(min_value=0.0, max_value=0.499, format="%.3f",
                                                           required=True),
                "Density (kg/m³)": st.column_config.NumberColumn(min_value=1.0, format="%.5g", required=True),
                "Cost (₹/kg)": st.column_config.NumberColumn(min_value=0.0, format="%.4g"),
            })
        SS["ds_lib"] = edited
        if st.button("Restore library defaults", icon=":material/history:"):
            SS["ds_lib"] = library_df()
            SS["ds_lib_ver"] = SS.get("ds_lib_ver", 0) + 1
            st.rerun()

        names = list(edited["Material"]) + [CUSTOM_MAT]
        if SS.get("ds_mat") not in names:
            SS["ds_mat"] = names[0]
        st.selectbox("Material for this beam", names, key="ds_mat")
        material = None
        if SS["ds_mat"] == CUSTOM_MAT:
            with st.container(border=True):
                st.text_input("Material name", key="ds_c_name")
                a, b, c = st.columns(3)
                a.number_input("E (GPa)", min_value=0.0, step=1.0, format="%.4g", key="ds_c_E")
                b.number_input("Yield strength σy (MPa)", min_value=0.0, step=5.0, format="%.4g", key="ds_c_sy")
                c.number_input("UTS (MPa, optional)", min_value=0.0, step=5.0, format="%.4g", key="ds_c_uts")
                a.number_input("Poisson's ratio ν", min_value=0.0, max_value=0.499, step=0.01, format="%.3f",
                               key="ds_c_nu")
                b.number_input("Density (kg/m³)", min_value=0.0, step=10.0, format="%.5g", key="ds_c_rho")
                c.number_input("Cost (₹/kg, optional)", min_value=0.0, step=5.0, format="%.4g", key="ds_c_cost")
            try:
                material = materials.custom_material(
                    SS["ds_c_name"].strip() or "Custom material", (SS["ds_c_E"] or 0) * 1e9,
                    (SS["ds_c_sy"] or 0) * 1e6, SS["ds_c_nu"] or 0.0, SS["ds_c_rho"] or 0.0,
                    uts=None if SS["ds_c_uts"] is None else SS["ds_c_uts"] * 1e6, cost_per_kg=SS["ds_c_cost"])
            except ValueError as exc:
                errors.append(f"Custom material: {exc}.")
        else:
            row = edited[edited["Material"] == SS["ds_mat"]].iloc[0]
            if row.isna()[["E (GPa)", "Yield σy (MPa)", "Poisson ν", "Density (kg/m³)"]].any():
                errors.append(f"Library entry '{SS['ds_mat']}' has an empty required value.")
            else:
                material = row_material(row)
                if material.uts is not None and material.uts < material.yield_strength:
                    errors.append("Material: UTS must not be below the yield strength.")
        if material is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("E", f"{material.E / 1e9:.4g} GPa")
            m2.metric("G = E/2(1+ν)", f"{material.G / 1e9:.4g} GPa")
            m3.metric("Yield σy", f"{material.yield_strength / 1e6:.4g} MPa")
            m4.metric("UTS", f"{material.uts / 1e6:.4g} MPa" if material.uts else "—")

    # ------------------------------------------------------------------ supports
    with tab_sup:
        L = SS["ds_L"]
        opts = list(SUPPORT_PRESETS) + ([CUSTOM_SUP] if SS["ds_sup"] == CUSTOM_SUP else [])
        st.radio("Support arrangement", opts, key="ds_sup",
                 format_func=lambda k: SUPPORT_PRESETS[k][0] if k in SUPPORT_PRESETS
                 else "Custom arrangement (from Playground / loaded config)")
        if SS["ds_sup"] == CUSTOM_SUP:
            supports = [Support(t, x) for t, x in SS["ds_custom_supports"]]
            st.dataframe(pd.DataFrame([{"Type": s.type, "x (m)": s.x, "Restrains": ", ".join(s.restrained)}
                                       for s in supports]), hide_index=True)
            st.caption("Arbitrary support placement is edited in the Structural Playground; switch to a preset "
                       "above to replace it.")
        else:
            supports = SUPPORT_PRESETS[SS["ds_sup"]][1](L)
        st.caption("Fixed restrains u, v, θ · pin restrains u, v · roller restrains v only.")

    # ------------------------------------------------------------------ loads
    with tab_load:
        u_cols = st.columns(4)
        u_cols[0].selectbox("Length unit", ["m", "mm"], key="ds_u_len")
        u_cols[1].selectbox("Force unit", ["kN", "N"], key="ds_u_force")
        u_cols[2].selectbox("Distributed unit", ["kN/m", "N/m"], key="ds_u_dist")
        u_cols[3].selectbox("Moment unit", ["kN·m", "N·m"], key="ds_u_mom")
        u = units_now()
        if SS.get("ds_units_synced") != u:
            SS["ds_units_synced"] = u
            SS["ds_loads_base"] = load_table(shared, u)
            SS["ds_ver"] = SS.get("ds_ver", 0) + 1
        st.caption(f"Positions in **{u['len']}** from the left end. Magnitudes are positive; the Direction column "
                   f"sets the sense (Down/Up/Axial/Inclined for forces and distributed loads, CW/CCW for moments). "
                   f"Distributed: leave *End magnitude* blank for a UDL, fill it for a linearly varying or "
                   f"triangular load. *Angle* (CCW from +x, −90° = down) is used only for Inclined point loads.")
        table = st.data_editor(
            SS["ds_loads_base"], key=f"ds_loads_{SS['ds_ver']}", num_rows="dynamic", hide_index=True,
            column_config={
                "Type": st.column_config.SelectboxColumn(options=LOAD_TYPES, required=True, default="Point load"),
                "Position / start": st.column_config.NumberColumn(f"Position / start ({u['len']})", format="%.4g"),
                "End": st.column_config.NumberColumn(f"End ({u['len']})", format="%.4g",
                                                     help="Distributed loads only"),
                "Magnitude": st.column_config.NumberColumn(
                    f"Magnitude ({u['force']} | {u['dist']} | {u['mom']})", min_value=0.0, format="%.4g"),
                "End magnitude": st.column_config.NumberColumn(f"End magnitude ({u['dist']})", min_value=0.0,
                                                               format="%.4g"),
                "Direction": st.column_config.SelectboxColumn(options=DIRECTIONS, required=True, default="Down"),
                "Angle (°)": st.column_config.NumberColumn(min_value=-360.0, max_value=360.0, format="%.4g"),
            })

        loads = []
        Lm = SS["ds_L"]
        for i, row in table.reset_index(drop=True).iterrows():
            tag = f"Load row {i + 1}"
            if row.drop("Type").isna().all() or pd.isna(row["Type"]):
                continue
            typ, dirn = row["Type"], row["Direction"]
            pos, mag = row["Position / start"], row["Magnitude"]
            if pd.isna(pos) or pd.isna(mag):
                errors.append(f"{tag}: position and magnitude are required.")
                continue
            x = units.to_si(float(pos), u["len"])
            if typ == "Point load":
                angles = {"Down": -90.0, "Up": 90.0, "Axial →": 0.0, "Axial ←": 180.0}
                if dirn == "Inclined":
                    if pd.isna(row["Angle (°)"]):
                        errors.append(f"{tag}: an Inclined point load needs an angle.")
                        continue
                    ang = float(row["Angle (°)"])
                elif dirn in angles:
                    ang = angles[dirn]
                else:
                    errors.append(f"{tag}: a point load's direction must be Down, Up, Axial or Inclined (not {dirn}).")
                    continue
                if mag <= 0:
                    errors.append(f"{tag}: magnitude must be greater than zero.")
                    continue
                loads.append(PointLoad(x, units.to_si(float(mag), u["force"]), ang))
            elif typ == "Distributed":
                if dirn not in ("Down", "Up"):
                    errors.append(f"{tag}: a distributed load acts Down or Up (axial distributed loads are not "
                                  f"supported).")
                    continue
                if pd.isna(row["End"]):
                    errors.append(f"{tag}: a distributed load needs an end position.")
                    continue
                w1 = units.to_si(float(mag), u["dist"])
                w2 = w1 if pd.isna(row["End magnitude"]) else units.to_si(float(row["End magnitude"]), u["dist"])
                if w1 == 0 and w2 == 0:
                    errors.append(f"{tag}: both intensities are zero.")
                    continue
                loads.append(DistributedLoad(x, units.to_si(float(row["End"]), u["len"]), w1, w2, dirn.lower()))
            else:
                if dirn not in ("CW", "CCW"):
                    errors.append(f"{tag}: a moment's direction must be CW or CCW.")
                    continue
                if mag <= 0:
                    errors.append(f"{tag}: magnitude must be greater than zero.")
                    continue
                M = units.to_si(float(mag), u["mom"])
                loads.append(MomentLoad.ccw(x, M) if dirn == "CCW" else MomentLoad.cw(x, M))
        if not loads and not errors:
            st.info("No loads: the beam is unloaded (all results will be zero).")

# --------------------------------------------------------------------------- assemble, validate, auto-save
candidate = None
if not errors and material is not None:
    candidate = BeamModel(L=float(SS["ds_L"]), section=section, material=material, supports=supports, loads=loads,
                          name=SS["ds_name"].strip())
    errors += validate(candidate)

with right:
    st.subheader("Preview")
    if errors:
        for e in errors:
            st.error(e, icon=":material/error:")
        st.warning("Not saved: the workspace still holds the last valid design.", icon=":material/block:")
        preview = shared
    else:
        stab = check_stability(candidate)
        if candidate.to_json() != SS["ds_synced"]:
            if candidate.to_dict() != shared.to_dict():  # value compare: entry re-sync only turns 7850 into 7850.0
                st.toast("Design saved to workspace", icon=":material/check_circle:")
            set_model(candidate)
            SS["ds_synced"] = candidate.to_json()
        preview = candidate
        (st.success if stab.stable else st.error)(stab.message)
    st.plotly_chart(beam_schematic(preview, height=240), theme=None, key="ds_schematic")
    if not errors:
        try:
            res = analyze(preview)
            q1, q2, q3 = st.columns(3)
            q1.metric("|v|max", f"{abs(res.v_max.value) * 1e3:.4g} mm")
            q2.metric("|σ|max", f"{abs(res.sigma_max.value) / 1e6:.4g} MPa")
            sig = abs(res.sigma_max.value)
            q3.metric("Yield FOS", f"{preview.material.yield_strength / sig:.3g}" if sig else "∞")
            st.caption(f"Solver route: {METHOD_LABEL[res.method]}")
        except (ModelError, UnstableStructureError) as exc:
            st.error(str(exc))
    st.page_link("pages/analysis.py", label="Go to full analysis", icon=":material/functions:")
    st.page_link("pages/visualization_3d.py", label="View in 3D", icon=":material/view_in_ar:")
    st.page_link("pages/playground.py", label="Place supports anywhere (Playground)", icon=":material/draw:")

    with st.container(border=True):
        n1, n2 = st.columns([1.6, 1], vertical_alignment="bottom")
        save_name = n1.text_input("Save a snapshot for Design Comparison", placeholder="Design name")
        if n2.button("Save snapshot", icon=":material/bookmark_add:", width="stretch",
                     disabled=bool(errors) or not save_name.strip()):
            named = candidate.copy()
            named.name = save_name.strip()
            save_design(named.name, named)
            st.toast(f"Saved '{named.name}' ({len(saved_designs())} saved designs).")
