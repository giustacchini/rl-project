# test_snow_plow_env.py

from types import SimpleNamespace

import numpy as np

from Enums import MovementActions
from snow_plow_env import SnowPlowEnv


class TestSnowPlowEnv(SnowPlowEnv):
    def __init__(self, agents, width, height, max_steps):
        super().__init__(agents=agents, width=width, height=height, max_steps=max_steps)


def make_env(positions, width=8, height=8):
    env = TestSnowPlowEnv(agents=[], width=width, height=height, max_steps=100)

    env.agents = [
        SimpleNamespace(
            index=agent_id,
            state=SimpleNamespace(pos=np.array(position, dtype=int)),
        )
        for agent_id, position in enumerate(positions)
    ]

    return env


def position(env, agent_id):
    return tuple(env.agents[agent_id].state.pos)


def status(env, agent_id):
    return env.last_movement_results[agent_id]["status"]


def test_single_agent_moves_into_empty_cell():
    env = make_env([(1, 1)])

    rewards = env.handle_actions(
        {
            0: MovementActions.RIGHT,
        }
    )

    assert position(env, 0) == (2, 1)
    assert status(env, 0) == "moved"
    assert rewards == {0: 0}


def test_single_agent_do_nothing():
    env = make_env([(1, 1)])

    rewards = env.handle_actions(
        {
            0: MovementActions.DO_NOTHING,
        }
    )

    assert position(env, 0) == (1, 1)
    assert status(env, 0) == "waited"
    assert rewards == {0: 0}


def test_two_agents_request_same_empty_cell():
    env = make_env(
        [
            (1, 1),
            (2, 2),
        ]
    )

    env.handle_actions(
        {
            0: MovementActions.RIGHT,
            1: MovementActions.UP,
        }
    )

    assert position(env, 0) == (1, 1)
    assert position(env, 1) == (2, 2)
    assert status(env, 0) == "blocked_same_destination"
    assert status(env, 1) == "blocked_same_destination"


def test_two_agents_cannot_swap_cells():
    env = make_env(
        [
            (1, 1),
            (2, 1),
        ]
    )

    env.handle_actions(
        {
            0: MovementActions.RIGHT,
            1: MovementActions.LEFT,
        }
    )

    assert position(env, 0) == (1, 1)
    assert position(env, 1) == (2, 1)
    assert status(env, 0) == "blocked_swap"
    assert status(env, 1) == "blocked_swap"


def test_agent_cannot_enter_cell_of_stationary_agent():
    env = make_env(
        [
            (1, 1),
            (2, 1),
        ]
    )
