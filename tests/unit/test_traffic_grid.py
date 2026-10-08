"""Configured traffic maps preserve terrain during random agent placement."""

from collections import Counter

import pytest

from Enums import GridCellTypes
from snow_plow_env import SnowPlowEnv
from Traffic_configs import road_network_1


def make_traffic_env(seed):
    rows, columns = road_network_1["size"]
    env = SnowPlowEnv(
        agents=2,
        width=columns,
        height=rows,
        max_steps=50,
        road_network=road_network_1,
    )
    env.reset(seed=seed)
    return env


def test_traffic_map_contains_all_cell_categories():
    env = make_traffic_env(42)
    counts = Counter(cell.cell_type for cell in env.grid.grid)
    assert counts == {
        GridCellTypes.ROAD: 69,
        GridCellTypes.HIGHWAY: 26,
        GridCellTypes.INTERSECTION: 12,
        GridCellTypes.OBSTACLE: 83,
    }
    for cell in env.grid.grid:
        assert cell.can_overlap() == (cell.cell_type != GridCellTypes.OBSTACLE)
    frame = env.get_frame(highlight=False, tile_size=16)
    assert frame.shape == (160, 304, 3)


@pytest.mark.parametrize("seed", range(10))
def test_traffic_agents_spawn_on_distinct_preserved_walkable_cells(seed):
    env = make_traffic_env(seed)
    positions = [tuple(agent.state.pos) for agent in env.agents]
    assert len(set(positions)) == 2
    for position in positions:
        cell = env.grid.get(*position)
        assert cell is not None
        assert cell.can_overlap()
    env.reset(seed=seed)
    assert [tuple(agent.state.pos) for agent in env.agents] == positions


def test_traffic_map_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="Road network size"):
        SnowPlowEnv(
            agents=2, width=8, height=8, max_steps=50,
            road_network=road_network_1,
        )
