"""Structural Playground: converter, presets vs closed form, editing/positioning, invalid configs, files, messages."""
import json

import numpy as np
import pytest

from smartbeam import fem, sections
from smartbeam.analytical import analyze
from smartbeam.model import ModelError, UnstableStructureError
from smartbeam.playground import convert

TB_CASE = {"cantilever_end": "cantilever-end-load", "cantilever_udl": "cantilever-udl",
           "ss_central": "ss-central-load", "ss_udl": "ss-udl"}


def cfg_of(key):
    return convert.preset_configs()[key]["config"]


def run(cfg):
    return analyze(convert.config_to_model(cfg))


def EI(model):
    return model.material.E * sections.second_moment(model.section)


def rel(a, b):
    return abs(a - b) / abs(b)


# --------------------------------------------------------------------------- presets vs textbook / closed form
def test_six_presets_exist():
    assert set(convert.PRESETS) == {"cantilever_end", "cantilever_udl", "ss_central", "ss_udl", "ss_partial_udl",
                                    "ss_multi_point"}


@pytest.mark.parametrize("key", list(TB_CASE))
def test_textbook_presets(key):
    m = convert.preset_model(key)
    r = run(cfg_of(key))
    L, ei, Z = m.L, EI(m), sections.section_modulus(m.section)
    ld = m.loads[0]
    P = abs(getattr(ld, "fy", 0.0))
    w = getattr(ld, "w1", 0.0)
    expected = {
        "cantilever_end": dict(R=P, M=P * L, d=P * L**3 / (3 * ei), t=P * L**2 / (2 * ei)),
        "cantilever_udl": dict(R=w * L, M=w * L**2 / 2, d=w * L**4 / (8 * ei), t=w * L**3 / (6 * ei)),
        "ss_central": dict(R=P / 2, M=P * L / 4, d=P * L**3 / (48 * ei), t=P * L**2 / (16 * ei)),
        "ss_udl": dict(R=w * L / 2, M=w * L**2 / 8, d=5 * w * L**4 / (384 * ei), t=w * L**3 / (24 * ei)),
    }[key]
    assert r.method == "analytical" and r.textbook.case == TB_CASE[key]
    assert max(abs(re.Fy) for re in r.reactions) == pytest.approx(expected["R"], rel=1e-9)
    assert abs(r.M_max.value) == pytest.approx(expected["M"], rel=1e-6)
    assert abs(r.v_max.value) == pytest.approx(expected["d"], rel=1e-6)
    assert abs(r.theta_max.value) == pytest.approx(expected["t"], rel=1e-6)
    assert abs(r.sigma_max.value) == pytest.approx(expected["M"] / Z, rel=1e-6)
    assert r.v_max.value < 0  # loads act downward -> deflection down


def test_partial_udl_preset_closed_form():
    m = convert.preset_model("ss_partial_udl")
    r = run(cfg_of("ss_partial_udl"))
    ld, L = m.loads[0], m.L
    a, b, w = ld.x1, ld.x2, ld.w1
    W = w * (b - a)
    RA = W * (L - (a + b) / 2) / L
    x0 = a + RA / w  # V = 0 inside the loaded length
    Mmax = RA * x0 - w * (x0 - a) ** 2 / 2
    ra, rb = sorted(r.reactions, key=lambda q: q.x)
    assert ra.Fy == pytest.approx(RA, rel=1e-12) and rb.Fy == pytest.approx(W - RA, rel=1e-12)
    assert r.M_max.value == pytest.approx(Mmax, rel=1e-5)
    fr = fem.solve(m, n_elements=40)  # independent stiffness solution, exact at nodes
    assert abs(r.v_max.value) == pytest.approx(abs(fr.v_max.value), rel=1e-4)


def test_multi_point_preset_superposition():
    m = convert.preset_model("ss_multi_point")
    r = run(cfg_of("ss_multi_point"))
    L, ei = m.L, EI(m)

    def v_at(x):  # SS beam, sum of PL-type closed forms (downward loads -> negative)
        tot = 0.0
        for ld in m.loads:
            P, a = abs(ld.fy), ld.x
            bb = L - a
            if x <= a:
                tot -= P * bb * x * (L**2 - bb**2 - x**2) / (6 * L * ei)
            else:
                tot -= P * a * (L - x) * (L**2 - a**2 - (L - x) ** 2) / (6 * L * ei)
        return tot

    RA = sum(abs(ld.fy) * (L - ld.x) for ld in m.loads) / L
    assert min(r.reactions, key=lambda q: q.x).Fy == pytest.approx(RA, rel=1e-12)
    for x in (0.4, 1.0, 1.6):
        i = int(np.argmin(np.abs(r.x - x)))
        assert r.v[i] == pytest.approx(v_at(x), rel=1e-6)


# --------------------------------------------------------------------------- converter
def test_round_trip_and_ui_keys_stripped():
    cfg = cfg_of("ss_multi_point")
    assert cfg["material_key"] == "structural_steel" and cfg["units"] == convert.DEFAULT_UNITS
    assert all("id" in o for o in cfg["supports"] + cfg["loads"])
    m = convert.config_to_model(json.loads(json.dumps(cfg)))
    assert m.to_dict() == convert.preset_model("ss_multi_point").to_dict() | {"name": m.name}


def test_editing_positions_direction_and_support_movement():
    cfg = cfg_of("ss_central")
    cfg["loads"][0]["x"] = 0.5  # move the load
    ra, rb = sorted(run(cfg).reactions, key=lambda q: q.x)
    assert (ra.Fy, rb.Fy) == pytest.approx((1500.0, 500.0))
    cfg["loads"][0]["angle_deg"] = 90.0  # flip direction -> reactions flip
    ra, rb = sorted(run(cfg).reactions, key=lambda q: q.x)
    assert (ra.Fy, rb.Fy) == pytest.approx((-1500.0, -500.0))
    cfg["loads"][0]["angle_deg"] = -90.0
    cfg["supports"][1]["x"] = 1.0  # move the roller to mid-span: load at 0.5 now between supports
    ra, rb = sorted(run(cfg).reactions, key=lambda q: q.x)
    assert (ra.Fy, rb.Fy) == pytest.approx((1000.0, 1000.0))


def test_multiple_and_distributed_loads_superpose():
    base = cfg_of("ss_udl")
    tri = json.loads(json.dumps(base))
    tri["loads"] = [{"id": "t", "kind": "distributed", "x1": 0.0, "x2": 2.0, "w1": 0.0, "w2": 3000.0, "direction": "down"}]
    r_tri = run(tri)
    ra, rb = sorted(r_tri.reactions, key=lambda q: q.x)
    assert (ra.Fy, rb.Fy) == pytest.approx((1000.0, 2000.0))  # wL/6, wL/3
    assert r_tri.M_max.value == pytest.approx(3000 * 4 / (9 * np.sqrt(3)), rel=1e-5)
    both = json.loads(json.dumps(base))
    both["loads"] += tri["loads"]
    r_both, r_udl = run(both), run(base)
    assert r_both.v_max.value == pytest.approx(np.interp(r_both.v_max.x, r_udl.x, r_udl.v)
                                               + np.interp(r_both.v_max.x, r_tri.x, r_tri.v), rel=1e-6)


def test_routes_for_indeterminate_inclined_and_moment():
    cfg = cfg_of("ss_udl")
    cfg["supports"].append({"id": "s3", "type": "roller", "x": 1.0})
    r = run(cfg)
    assert r.method == "fem-euler" and not r.stability.determinate
    assert sorted(q.Fy for q in r.reactions) == pytest.approx(sorted([3 / 8 * 2000, 10 / 8 * 2000, 3 / 8 * 2000]), rel=1e-6)
    cfg = cfg_of("ss_central")
    cfg["loads"][0]["angle_deg"] = -30.0
    r = run(cfg)
    assert r.method == "fem-frame"
    assert sum(q.Fx for q in r.reactions) == pytest.approx(-2000 * np.cos(np.radians(-30)), rel=1e-9)
    cfg = cfg_of("ss_central")
    cfg["loads"] = [{"id": "m", "kind": "moment", "x": 1.0, "M": -1000.0}]  # CW couple at mid-span
    ra, rb = sorted(run(cfg).reactions, key=lambda q: q.x)
    assert (ra.Fy, rb.Fy) == pytest.approx((-500.0, 500.0))


@pytest.mark.parametrize("mutate, msg", [
    (lambda c: c["loads"][0].update(x=5.0), "off the beam"),
    (lambda c: c["loads"].append({"kind": "distributed", "x1": 1.5, "x2": 0.5, "w1": 1, "w2": 1, "direction": "down"}),
     "x1 < x2"),
    (lambda c: c["loads"][0].update(magnitude="abc"), "must be a number"),
    (lambda c: c["loads"].append({"kind": "spring", "x": 1}), "unknown kind"),
    (lambda c: c.pop("section"), "missing the field"),
    (lambda c: c["section"].update(h=-0.01), "must be positive"),
    (lambda c: c.update(supports=[]), "At least one support"),
])
def test_invalid_configs_rejected(mutate, msg):
    cfg = cfg_of("ss_central")
    mutate(cfg)
    with pytest.raises(ModelError, match=msg):
        convert.config_to_model(cfg)


def test_mechanism_rejected_with_explanation():
    cfg = cfg_of("ss_central")
    cfg["supports"] = [{"id": "s1", "type": "roller", "x": 0.0}]
    with pytest.raises(UnstableStructureError, match="rigid-body"):
        run(cfg)
    cfg["supports"] = [{"type": "roller", "x": 0.0}, {"type": "roller", "x": 2.0}]
    cfg["loads"][0]["angle_deg"] = -45.0  # two rollers cannot carry the axial component
    with pytest.raises(UnstableStructureError, match="Rollers cannot resist"):
        run(cfg)


# --------------------------------------------------------------------------- files + canvas messages
def test_save_and_restore(tmp_path):
    cfg = cfg_of("ss_partial_udl")
    cfg["units"] = {**cfg["units"], "length": "mm"}
    path = convert.save_config(cfg, "my beam/../x", tmp_path)
    assert path.parent == tmp_path and path.name == "my_beam_x.json"  # slugged, cannot escape the directory
    assert convert.list_configs(tmp_path) == ["my_beam_x"]
    back = convert.load_config("my_beam_x", tmp_path)
    assert back["units"]["length"] == "mm"
    assert convert.config_to_model(back).to_dict() == convert.config_to_model(cfg).to_dict()
    with pytest.raises(ModelError):
        convert.save_config({**cfg, "L": -1}, "bad", tmp_path)


def test_accept_message_handles_each_nonce_once():
    state = {"playground_nonce": None}
    cfg = cfg_of("ss_udl")
    msg = {"config": cfg, "action": "live", "nonce": "1-1", "version": 3}
    assert convert.accept_message(state, msg) == "live"
    assert state["playground_seen"] == 3 and state["playground_config"] is cfg
    assert convert.accept_message(state, msg) is None  # same value returned on the next rerun
    assert convert.accept_message(state, {**msg, "nonce": "1-2", "action": "edit"}) == "edit"
    assert convert.accept_message(state, {**msg, "nonce": "1-3", "action": "analyse"}) == "analyse"
    assert convert.accept_message(state, None) is None
    assert convert.accept_message(state, {"nonce": "x", "config": "nope"}) is None


def test_result_payload_is_json_safe():
    r = run(cfg_of("cantilever_end"))
    p = json.loads(json.dumps(convert.result_payload(r)))
    assert p["method"] == "analytical" and len(p["x"]) == len(p["M"])
    assert p["reactions"][0]["M"] == pytest.approx(2000.0)  # +PL, CCW
    assert p["max"]["v_max"]["value"] == pytest.approx(r.v_max.value, rel=1e-6)
