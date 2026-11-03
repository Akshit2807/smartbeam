"""Streamlit wrapper for the SVG beam editor in ./frontend (plain HTML/JS, no build step)."""
from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_canvas = components.declare_component("smartbeam_canvas", path=str(Path(__file__).parent / "frontend"))


def beam_canvas(*, config: dict, config_rev: int, seen_version: int, results: dict | None, results_current: bool,
                error: str | None, presets: dict, materials: dict, key: str, on_change=None, height: int = 760):
    """Render the editor. Returns the last value it sent: {config, action: 'analyse'|'live'|'edit', nonce, version}.

    config_rev: bump to make the canvas replace its local state with `config` (preset/file load from Python).
    seen_version: the canvas edit counter Python last received (lets the canvas tell if results are stale).
    """
    return _canvas(config=config, config_rev=config_rev, seen_version=seen_version, results=results,
                   results_current=results_current, error=error, presets=presets, materials=materials,
                   height=height, key=key, on_change=on_change, default=None)
