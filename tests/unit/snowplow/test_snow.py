"""Snow layer, clouds and clearing (snowplow/snow.py)."""

import numpy as np

from snowplow.config import SnowConfig
from snowplow.road_map import RoadMap
from snowplow.road_network_configs import road_network_1
from snowplow.snow import Cloud, SnowField

ROAD_MAP = RoadMap.from_config(road_network_1)


def field(**kwargs):
    return SnowField(ROAD_MAP, SnowConfig(**kwargs))


def test_initial_snow_is_reproducible_and_only_on_roads():
    a, b = field(), field()
    a.reset(np.random.default_rng(3)), b.reset(np.random.default_rng(3))
    assert (a.snow == b.snow).all() and a.snow.any()
    assert a.snow[~ROAD_MAP.passable].sum() == 0
    assert a.snow.max() <= 3


def test_cloud_adds_snow_to_roads_below_it_capped_at_three():
    f = field(initial_clouds=0, cloud_spawn_prob=0.0)
    cells = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # (x, y). On map 1 only (1,0) is a building
    f.clouds.append(Cloud(cells=cells, ttl=10))
    for _ in range(5):
        f.snowfall_step(np.random.default_rng(0), step=1)
    assert f.snow[0, 0] == 3 and f.snow[1, 0] == 3 and f.snow[1, 1] == 3  # road cells hit the cap
    assert f.snow[0, 1] == 0  # (x=1, y=0) is a building: no snow


def test_clouds_expire():
    f = field(initial_clouds=0, cloud_spawn_prob=0.0)
    f.clouds.append(Cloud(cells=np.array([[0, 0]]), ttl=2))
    f.age_clouds()
    assert len(f.clouds) == 1
    f.age_clouds()
    assert f.clouds == []


def test_clear_removes_all_layers():
    f = field()
    f.snow[1, 0] = 3
    assert f.clear(0, 1) == 3 and f.snow[1, 0] == 0
    assert f.clear(0, 1) == 0


def test_plow_step_single_and_team_times():
    f = field()
    f.snow[1, 0] = 3
    for _ in range(9):  # one plow: 10 steps for level 3
        assert f.plow_step({(0, 1): 1}) == {}
    assert f.plow_step({(0, 1): 1}) == {(0, 1): 3} and f.snow[1, 0] == 0
    f.snow[1, 0] = 3
    for _ in range(2):  # two plows: 3 steps for level 3
        assert f.plow_step({(0, 1): 2}) == {}
    assert f.plow_step({(0, 1): 2}) == {(0, 1): 3} and f.plow_progress[1, 0] == 0


def test_storm_steps_stops_new_clouds():
    f = field(initial_clouds=0, cloud_spawn_prob=1.0, storm_steps=3)
    rng = np.random.default_rng(0)
    f.snowfall_step(rng, step=3)
    assert len(f.clouds) == 1
    f.clouds.clear()
    f.snowfall_step(rng, step=4)  # the storm is over
    assert f.clouds == []


def test_cloud_shape_is_irregular_and_within_nine_by_nine():
    f = field()
    rng = np.random.default_rng(1)
    sizes = set()
    for _ in range(50):
        cloud = f._new_cloud(rng, on_road=False)
        xs, ys = cloud.cells[:, 0], cloud.cells[:, 1]
        assert xs.max() - xs.min() < 9 and ys.max() - ys.min() < 9
        sizes.add(len(cloud.cells))
    assert len(sizes) > 5  # many different shapes
