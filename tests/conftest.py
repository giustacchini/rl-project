"""Shared test helpers (fixtures). `pytest.ini` puts src/ on the import path."""

import numpy as np
import pytest

from snowplow.config import SnowConfig, SnowplowConfig
from snowplow.env import SnowPlowEnv

# No clouds at all: tests place snow by hand so every outcome is deterministic.
DRY = SnowConfig(initial_clouds=0, cloud_spawn_prob=0.0)


def open_map(height=8, width=8):
    """One big two-way road without obstacles."""
    return {
        "size": (height, width),
        "roads": [{"position": (0, 0), "size": (height, width)}],
        "highways": [],
        "intersections": [],
        "obstacles": [],
    }


def place(env, positions):
    """Put the plows on given (x, y) cells."""
    for agent, pos in zip(env.agents, positions):
        agent.state.pos = np.array(pos, dtype=int)


def set_snow(env, cells):
    """Put snow on {(x, y): level}; also marks that snow has fallen (so 'all clean' can mean success)."""
    for (x, y), level in cells.items():
        env.snow_field.snow[y, x] = level
    env.snow_field.total_fallen = max(env.snow_field.total_fallen, 1)


@pytest.fixture
def make_env():
    """make_env(positions, road_network=None, **config) -> reset env with plows at `positions`, no snow."""

    def factory(positions, road_network=None, max_steps=100, snow=DRY, **config):
        cfg = SnowplowConfig(n_plows=len(positions), max_steps=max_steps, snow=snow, **config)
        env = SnowPlowEnv(road_network or open_map(), cfg)
        env.reset(seed=0)
        place(env, positions)
        return env

    return factory
