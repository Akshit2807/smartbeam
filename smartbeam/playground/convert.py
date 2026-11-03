"""Playground canvas config  <->  BeamModel, presets, result payloads and config files.

No Streamlit in here (engine side). The canvas config is the BeamModel JSON schema (docs/ARCHITECTURE.md §5,
SI units) plus a few UI-only keys that the converter strips:

    {"version":1, "name", "L", "section":{...}, "material":{...},
     "supports":[{"id","type","x"}], "loads":[{"id","kind",...}],
     "material_key": "structural_steel" | null, "units": {"force","length","dist","moment"}}
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import fields
from pathlib import Path

import numpy as np

from smartbeam.materials import MATERIALS, get_material
from smartbeam.model import (BeamModel, DistributedLoad, Material, ModelError, MomentLoad, PointLoad, Section,
                             Support, validate)

SAVED_DIR = Path(__file__).resolve().parents[2] / "saved_configs"
DEFAULT_UNITS = {"force": "kN", "length": "m", "dist": "kN/m", "moment": "kN·m"}
UNIT_CHOICES = {"force": ("N", "kN"), "length": ("m", "mm"), "dist": ("N/m", "kN/m"), "moment": ("N·m", "kN·m")}

_LOAD_KEYS = {"point": ("x", "magnitude", "angle_deg"), "distributed": ("x1", "x2", "w1", "w2", "direction"),
              "moment": ("x", "M")}
_MATERIAL_KEYS = tuple(f.name for f in fields(Material))
_SECTION_KEYS = ("shape", "b", "h", "d", "D")


# --------------------------------------------------------------------------- config <-> model
def material_key_of(material: Material) -> str | None:
    """Library key whose material equals `material` (None for custom materials)."""
    return next((k for k, m in MATERIALS.items() if m == material), None)


def model_to_config(model: BeamModel, units: dict | None = None) -> dict:
    """BeamModel -> canvas config (adds stable ids, units and the material library key)."""
    cfg = model.to_dict()
    for i, s in enumerate(cfg["supports"]):
        s["id"] = f"s{i + 1}"
    for i, ld in enumerate(cfg["loads"]):
        ld["id"] = f"l{i + 1}"
    cfg["material_key"] = material_key_of(model.material)
    cfg["units"] = {**DEFAULT_UNITS, **(units or {})}
    return cfg


def _num(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ModelError(f"{what} must be a number (got {v!r}).")
    try:
        f = float(v)
    except ValueError:
        raise ModelError(f"{what} must be a number (got {v!r}).") from None
    if not math.isfinite(f):
        raise ModelError(f"{what} must be finite (got {v!r}).")
    return f


def config_to_model(config: dict) -> BeamModel:
    """Canvas config -> validated BeamModel. Raises ModelError with every problem found.

    Stability (mechanisms) is not checked here; analytical.analyze() does that."""
    if not isinstance(config, dict):
        raise ModelError("Configuration must be a JSON object.")
    try:
        sec = config["section"]
        section = Section(**{k: (None if sec.get(k) is None else (sec[k] if k == "shape" else _num(sec[k], f"section {k}")))
                             for k in _SECTION_KEYS})
        mat = config["material"]
        material = Material(**{k: mat[k] for k in _MATERIAL_KEYS if k in mat})
        supports = [Support(str(s["type"]), _num(s["x"], f"support {i + 1} position"))
                    for i, s in enumerate(config.get("supports", []))]
        loads = []
        for i, ld in enumerate(config.get("loads", [])):
            kind = ld.get("kind")
            if kind not in _LOAD_KEYS:
                raise ModelError(f"Load {i + 1}: unknown kind {kind!r}.")
            kw = {k: (ld[k] if k == "direction" else _num(ld[k], f"load {i + 1} {k}")) for k in _LOAD_KEYS[kind]}
            loads.append({"point": PointLoad, "distributed": DistributedLoad, "moment": MomentLoad}[kind](**kw))
        model = BeamModel(L=_num(config["L"], "beam length"), section=section, material=material,
                          supports=supports, loads=loads, name=str(config.get("name", "") or "Playground beam"))
    except KeyError as e:
        raise ModelError(f"Configuration is missing the field {e.args[0]!r}.") from None
    except (TypeError, AttributeError) as e:
        raise ModelError(f"Malformed configuration: {e}") from None
    errors = validate(model)
    if errors:
        raise ModelError(" ".join(errors))
    return model


def accept_message(state, value) -> str | None:
    """Record a canvas message {config, action, nonce, version} in the mapping `state` (Streamlit session state:
    keys playground_config / playground_seen / playground_nonce). Each nonce is accepted once, because the component
    keeps returning its last value on every rerun. Returns 'analyse' | 'live' | 'edit', or None to ignore."""
    if (not isinstance(value, dict) or not isinstance(value.get("config"), dict)
            or value.get("nonce") is None or value.get("nonce") == state.get("playground_nonce")):
        return None
    state["playground_nonce"] = value["nonce"]
    state["playground_config"] = value["config"]
    state["playground_seen"] = int(value.get("version") or 0)
    action = value.get("action")
    return action if action in ("analyse", "live") else "edit"


# --------------------------------------------------------------------------- presets
PRESET_L = 2.0  # m
PRESET_SECTION = (0.040, 0.080)  # rect b x h, m


def _preset(name: str, supports, loads) -> BeamModel:
    return BeamModel(L=PRESET_L, section=Section.rect(*PRESET_SECTION), material=get_material("structural_steel"),
                     supports=supports, loads=loads, name=name)


def _ss():
    return [Support("pin", 0.0), Support("roller", PRESET_L)]


PRESETS = {
    "cantilever_end": ("Cantilever · end load", lambda: _preset(
        "Cantilever with end load", [Support("fixed", 0.0)], [PointLoad(PRESET_L, 1000.0, -90.0)])),
    "cantilever_udl": ("Cantilever · full UDL", lambda: _preset(
        "Cantilever with UDL", [Support("fixed", 0.0)], [DistributedLoad(0.0, PRESET_L, 1000.0, 1000.0)])),
    "ss_central": ("Simply supported · central load", lambda: _preset(
        "SS beam with central load", _ss(), [PointLoad(PRESET_L / 2, 2000.0, -90.0)])),
    "ss_udl": ("Simply supported · full UDL", lambda: _preset(
        "SS beam with UDL", _ss(), [DistributedLoad(0.0, PRESET_L, 2000.0, 2000.0)])),
    "ss_partial_udl": ("Simply supported · partial UDL", lambda: _preset(
        "SS beam with partial UDL", _ss(), [DistributedLoad(0.5, 1.25, 3000.0, 3000.0)])),
    "ss_multi_point": ("Simply supported · multiple point loads", lambda: _preset(
        "SS beam with three point loads", _ss(),
        [PointLoad(0.4, 1000.0, -90.0), PointLoad(1.0, 2000.0, -90.0), PointLoad(1.6, 1500.0, -90.0)])),
}


def preset_model(key: str) -> BeamModel:
    return PRESETS[key][1]()


def preset_configs(units: dict | None = None) -> dict:
    """{key: {"label", "config"}} for the canvas Presets menu."""
    return {k: {"label": label, "config": model_to_config(build(), units)} for k, (label, build) in PRESETS.items()}


# --------------------------------------------------------------------------- result payload for the canvas
def _sig(a, n: int = 6) -> list[float]:
    return [float(f"{v:.{n}g}") for v in np.asarray(a, dtype=float)]


def _ext(e) -> dict:
    return {"value": float(e.value), "x": float(e.x), "where": e.where}


def result_payload(result) -> dict:
    """JSON-safe subset of an AnalysisResult for the canvas overlays (SI units)."""
    return {
        "method": result.method,
        "x": _sig(result.x), "V": _sig(result.V), "M": _sig(result.M), "N": _sig(result.N),
        "v": _sig(result.v), "theta": _sig(result.theta),
        "sigma_top": _sig(result.sigma_top), "sigma_bottom": _sig(result.sigma_bottom),
        "reactions": [{"x": r.x, "type": r.type, "Fx": float(r.Fx), "Fy": float(r.Fy), "M": float(r.M)}
                      for r in result.reactions],
        "max": {k: _ext(getattr(result, k)) for k in ("V_max", "M_max", "N_max", "theta_max", "v_max", "sigma_max")},
        "stability": result.stability.message,
        "warnings": list(result.warnings),
    }


# --------------------------------------------------------------------------- config files
def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    if not s:
        raise ValueError("Enter a file name (letters, digits, '-' or '_').")
    return s[:80]


def save_config(config: dict, name: str, directory: Path = SAVED_DIR) -> Path:
    """Validate and write `config` as <directory>/<name>.json (BeamModel schema + 'playground' UI block)."""
    model = config_to_model(config)
    data = model.to_dict()
    data["playground"] = {"units": {**DEFAULT_UNITS, **config.get("units", {})}}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slug(name)}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def list_configs(directory: Path = SAVED_DIR) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.json")) if directory.is_dir() else []


def load_config(name: str, directory: Path = SAVED_DIR) -> dict:
    """Read <directory>/<name>.json (plain BeamModel JSON also accepted) -> validated canvas config."""
    data = json.loads((directory / f"{_slug(name)}.json").read_text())
    units = (data.get("playground") or {}).get("units")
    return model_to_config(config_to_model(data), units)
