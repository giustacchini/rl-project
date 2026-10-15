"""Configured traffic maps preserve terrain during random agent placement."""

from copy import deepcopy

import numpy as np
import pytest

from snowplow import road_map as rm
from snowplow.config import SnowplowConfig
from snowplow.env import SnowPlowEnv
from snowplow.maps import road_network_1


def make_traffic_env(seed, road_network):
    env = SnowPlowEnv(
        road_network=road_network,
        config=SnowplowConfig(n_plows=2, max_steps=50),
    )
    env.reset(seed=seed)
    return env


def test_traffic_map_contains_all_cell_categories():
    env = make_traffic_env(42, road_network_1)
    expected = np.array([
        list(row) for row in (
            "RWWRWWWRWWWRWWWRWWR",
            "RRRRWWWRRRRRRRRRRRR",
            "RWWRWWWRWWWWWWWRWWR",
            "RWWRWWWRWWWWWWWRWWR",
            "IHHIHHHIHHHIHHHIHHI",
            "IHHIHHHIHHHIHHHIHHI",
            "RWWRWWWRWWWRWWWRWWR",
            "RWWRWWWRWWWRWWWRWWR",
            "RRRRRRRRRRRRRRRRRRR",
            "RWWRWWWRWWWRWWWRWWR",
        )
    ])
    for symbol, terrain in (
        ("R", rm.ROAD), ("H", rm.HIGHWAY),
        ("I", rm.INTERSECTION), ("W", rm.OBSTACLE),
    ):
        np.testing.assert_array_equal(env.road_map.codes == terrain, expected == symbol)
    for y in range(env.height):
        for x in range(env.width):
            cell = env.grid.get(x, y)
            assert cell.terrain == env.road_map.codes[y, x]
            assert cell.can_overlap() == (expected[y, x] != "W")
    frame = env.get_frame(highlight=False, tile_size=16)
    assert frame.shape == (160, 304, 3)


@pytest.mark.parametrize("seed", range(10))
def test_traffic_agents_spawn_on_distinct_preserved_walkable_cells(seed):
    env = make_traffic_env(seed, road_network_1)
    terrain_before = env.road_map.codes.copy()
    positions = [tuple(agent.state.pos) for agent in env.agents]
    assert len(set(positions)) == 2
    for position in positions:
        cell = env.grid.get(*position)
        assert cell is not None
        assert cell.can_overlap()
    env.reset(seed=seed)
    assert [tuple(agent.state.pos) for agent in env.agents] == positions
    np.testing.assert_array_equal(env.road_map.codes, terrain_before)


def test_traffic_map_rejects_rectangles_outside_dimensions():
    invalid_map = deepcopy(road_network_1)
    invalid_map["size"] = (8, 8)
    with pytest.raises(ValueError, match="does not fit"):
        SnowPlowEnv(road_network=invalid_map)
