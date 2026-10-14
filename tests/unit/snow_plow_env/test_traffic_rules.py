import numpy as np

from Enums import MovementActions
from snow_plow_env import SnowPlowEnv


def make_road_network():
    return {
        "size": (3, 3),
        "roads": [{"position": (0, 0), "size": (3, 3)}],
        "highways": [],
        "intersections": [],
        "obstacles": [],
    }


def make_env():
    env = SnowPlowEnv(
        agents=1,
        max_steps=10,
        road_network=make_road_network(),
    )
    env.reset(seed=0)
    env.agents[0].state.pos = np.array((1, 1), dtype=int)
    return env


def test_allowed_traffic_action_moves_agent():
    env = make_env()
    env.grid.get(1, 1).movement_actions = [
        MovementActions.RIGHT,
        MovementActions.DO_NOTHING,
    ]

    env.handle_actions({0: MovementActions.RIGHT})

    assert tuple(env.agents[0].state.pos) == (2, 1)
    assert env.last_movement_results[0]["status"] == "moved"


def test_disallowed_traffic_action_is_blocked():
    env = make_env()
    env.grid.get(1, 1).movement_actions = [
        MovementActions.RIGHT,
        MovementActions.DO_NOTHING,
    ]

    env.handle_actions({0: MovementActions.UP})

    assert tuple(env.agents[0].state.pos) == (1, 1)
    assert env.last_movement_results[0]["status"] == "blocked_invalid_action"


def test_do_nothing_is_allowed_by_traffic_rules():
    env = make_env()
    env.grid.get(1, 1).movement_actions = [MovementActions.DO_NOTHING]

    env.handle_actions({0: MovementActions.DO_NOTHING})

    assert tuple(env.agents[0].state.pos) == (1, 1)
    assert env.last_movement_results[0]["status"] == "waited"
