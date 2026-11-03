"""Shared fixtures: the four textbook benchmark models (steel, rectangular 30 x 60 mm, L = 2 m)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartbeam.materials import get_material  # noqa: E402
from smartbeam.model import BeamModel, DistributedLoad, PointLoad, Section, Support  # noqa: E402

L, P, W = 2.0, 1000.0, 500.0  # m, N (downward), N/m (downward)


@pytest.fixture
def steel():
    return get_material("structural_steel")


@pytest.fixture
def rect_section():
    return Section.rect(b=0.030, h=0.060)


def _model(supports, loads, name):
    return BeamModel(L=L, section=Section.rect(0.030, 0.060), material=get_material("structural_steel"),
                     supports=supports, loads=loads, name=name)


@pytest.fixture
def cantilever_end_load():
    return _model([Support("fixed", 0.0)], [PointLoad(L, P, -90.0)], "cantilever end load")


@pytest.fixture
def cantilever_udl():
    return _model([Support("fixed", 0.0)], [DistributedLoad(0.0, L, W, W)], "cantilever UDL")


@pytest.fixture
def ss_central_load():
    return _model([Support("pin", 0.0), Support("roller", L)], [PointLoad(L / 2, P, -90.0)], "SS central load")


@pytest.fixture
def ss_udl():
    return _model([Support("pin", 0.0), Support("roller", L)], [DistributedLoad(0.0, L, W, W)], "SS UDL")
