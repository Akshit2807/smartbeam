"""3D view must draw the engine's numbers: deflected centreline = amp * v(x), surface colour = sigma(x, y)."""
import numpy as np

from smartbeam import viz3d
from smartbeam.analytical import analyze


def test_beam_figure_uses_engine_fields(cantilever_end_load):
    r = analyze(cantilever_end_load)
    amp = viz3d.auto_amplification(r)
    assert 0.05 * r.model.L <= amp * abs(r.v_max.value) <= 0.13 * r.model.L

    fig = viz3d.beam_figure(r, amplification=amp)
    line = next(t for t in fig.data if t.name == "deflection")
    assert np.allclose(line.z, amp * r.v)

    beam = next(t for t in fig.data if t.type == "surface" and t.name == "beam" and t.opacity == 1.0)
    colour = np.asarray(beam.surfacecolor)  # MPa, perimeter starts on the top fibre (y = +h/2)
    assert np.allclose(colour[:, 0], r.sigma_top / 1e6)
    assert np.isclose(beam.cmax, np.abs(np.r_[r.sigma_top, r.sigma_bottom]).max() / 1e6)

    bare = viz3d.beam_figure(r, show_loads=False, show_supports=False, show_markers=False, show_undeformed=False)
    assert all(t.type in ("surface", "scatter3d") for t in bare.data)
