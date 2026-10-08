"""Configured traffic maps preserve terrain during random agent placement."""

from collections import Counter

import pytest

from Enums import GridCellTypes
from snow_plow_env import SnowPlowEnv
from Traffic_configs import road_network_1


<<<<<<< HEAD
def make_traffic_env(seed, road_network):
    env = SnowPlowEnv(
        agents=2,
        max_steps=50,
        road_network=road_network,
=======
def make_traffic_env(seed):
    rows, columns = road_network_1["size"]
    env = SnowPlowEnv(
        agents=2,
        width=columns,
        height=rows,
        max_steps=50,
        road_network=road_network_1,
>>>>>>> 0b6c6e5 (evol(visualization): applied modifications in order to see the environment in human render mode)
    )
    env.reset(seed=seed)
    return env


<<<<<<< HEAD
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
=======
def test_traffic_map_contains_all_cell_categories():
    env = make_traffic_env(42)
    counts = Counter(cell.cell_type for cell in env.grid.grid)
    assert counts == {
        GridCellTypes.ROAD: 69,
        GridCellTypes.HIGHWAY: 26,
        GridCellTypes.INTERSECTION: 12,
        GridCellTypes.OBSTACLE: 83,
    }
>>>>>>> 0b6c6e5 (evol(visualization): applied modifications in order to see the environment in human render mode)
    for cell in env.grid.grid:
        assert cell.can_overlap() == (cell.cell_type != GridCellTypes.OBSTACLE)
    frame = env.get_frame(highlight=False, tile_size=16)
    assert frame.shape == (160, 304, 3)


@pytest.mark.parametrize("seed", range(10))
def test_traffic_agents_spawn_on_distinct_preserved_walkable_cells(seed):
<<<<<<< HEAD
    env = make_traffic_env(seed, road_network_1)
=======
    env = make_traffic_env(seed)
>>>>>>> 0b6c6e5 (evol(visualization): applied modifications in order to see the environment in human render mode)
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
<<<<<<< HEAD
            agents=2,
            width=8,
            height=8,
            max_steps=50,
=======
            agents=2, width=8, height=8, max_steps=50,
>>>>>>> 0b6c6e5 (evol(visualization): applied modifications in order to see the environment in human render mode)
            road_network=road_network_1,
        )
