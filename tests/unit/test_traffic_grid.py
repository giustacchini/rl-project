"""Configured traffic maps preserve terrain during random agent placement."""

from collections import Counter

import pytest

from Enums import GridCellTypes
from snow_plow_env import SnowPlowEnv
from Traffic_configs import road_network_1


def make_traffic_env(seed, road_network):
    env = SnowPlowEnv(
        agents=2,
        max_steps=50,
        road_network=road_network,
    )
    env.reset(seed=seed)
    return env


def get_traffic_cell_counts(road_network):
    counts = Counter()
    for category, cell_type in {grid_type.value: grid_type for grid_type in GridCellTypes}.items():
        for rectangle in road_network[category]:
            row_count, column_count = rectangle["size"]
            counts[cell_type] += row_count * column_count
    return counts


def test_traffic_map_contains_all_cell_categories():
    env = make_traffic_env(42, road_network_1)
    counts = Counter(cell.cell_type for cell in env.grid.grid)

    # The counts of each cell type in the grid should match the counts specified in the road network configuration.
    assert counts == get_traffic_cell_counts(road_network_1)
    for cell in env.grid.grid:
        assert cell.can_overlap() == (cell.cell_type != GridCellTypes.OBSTACLE)
    frame = env.get_frame(highlight=False, tile_size=16)
    assert frame.shape == (160, 304, 3)


@pytest.mark.parametrize("seed", range(10))
def test_traffic_agents_spawn_on_distinct_preserved_walkable_cells(seed):
    env = make_traffic_env(seed, road_network_1)
    positions = [tuple(agent.state.pos) for agent in env.agents]
    assert len(set(positions)) == 2
    for position in positions:
        cell = env.grid.get(*position)
        assert cell is not None
        assert cell.can_overlap()
    env.reset(seed=seed)
    assert [tuple(agent.state.pos) for agent in env.agents] == positions


def test_traffic_map_dimensions_come_from_road_network():
    env = make_traffic_env(42, road_network_1)

    assert env.width == road_network_1["size"][1]
    assert env.height == road_network_1["size"][0]
