"""Shared Streamlit session state: current model, cached analysis, recent analyses, saved designs.

Keys in st.session_state (do not write them directly from pages; use these helpers):
    'model'          dict  current BeamModel.to_dict()
    'recent'         list  newest-first summaries (max RECENT_MAX)
    'saved_designs'  dict  name -> BeamModel.to_dict()
"""
from __future__ import annotations

import time

import streamlit as st

from smartbeam.materials import get_material
from smartbeam.model import BeamModel, PointLoad, Section, Support, classify

RECENT_MAX = 10


def default_model() -> BeamModel:
    """Steel rectangular cantilever, 1 m, 1 kN downward end load (fixed at x = 0)."""
    return BeamModel(
        L=1.0,
        section=Section.rect(b=0.030, h=0.060),
        material=get_material("structural_steel"),
        supports=[Support("fixed", 0.0)],
        loads=[PointLoad(x=1.0, magnitude=1000.0, angle_deg=-90.0)],
        name="Default cantilever",
    )


def get_model() -> BeamModel:
    """Fresh BeamModel copy of the current design (edit it, then set_model())."""
    if "model" not in st.session_state:
        st.session_state["model"] = default_model().to_dict()
    return BeamModel.from_dict(st.session_state["model"])


def set_model(model: BeamModel) -> None:
    st.session_state["model"] = model.to_dict()


def reset_model() -> None:
    set_model(default_model())


@st.cache_data(max_entries=64, show_spinner=False)
def _analyze_cached(model_json: str, n_points: int, n_elements: int):
    from smartbeam.analytical import analyze
    return analyze(BeamModel.from_json(model_json), n_points=n_points, n_elements=n_elements)


def analyze_model(model: BeamModel | None = None, n_points: int = 501, n_elements: int = 40, record: bool = True):
    """analytical.analyze() with caching keyed on the model JSON. Exceptions (ModelError,
    UnstableStructureError) propagate so pages can show them. record=True logs it in recent analyses."""
    model = model or get_model()
    result = _analyze_cached(model.to_json(indent=None), n_points, n_elements)
    if record:
        add_recent(model, result)
    return result


def add_recent(model: BeamModel, result) -> None:
    recent = st.session_state.setdefault("recent", [])
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "name": model.name or "Untitled",
        "config": classify(model),
        "method": result.method,
        "L_m": model.L,
        "delta_max_m": abs(result.v_max.value),
        "sigma_max_Pa": abs(result.sigma_max.value),
        "model": model.to_dict(),
    }
    if recent and recent[0]["model"] == entry["model"]:
        recent[0] = entry
    else:
        recent.insert(0, entry)
    del recent[RECENT_MAX:]


def recent_analyses() -> list[dict]:
    return list(st.session_state.get("recent", []))


def saved_designs() -> dict[str, BeamModel]:
    return {k: BeamModel.from_dict(v) for k, v in st.session_state.get("saved_designs", {}).items()}


def save_design(name: str, model: BeamModel) -> None:
    st.session_state.setdefault("saved_designs", {})[name] = model.to_dict()


def delete_design(name: str) -> None:
    st.session_state.get("saved_designs", {}).pop(name, None)
