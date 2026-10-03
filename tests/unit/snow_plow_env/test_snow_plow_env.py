
    
  
# test_snow_plow_env.py

from types import SimpleNamespace

import numpy as np

from snow_plow_env import SnowPlowAction, SnowPlowEnv


class TestSnowPlowEnv(SnowPlowEnv):
    def _gen_grid(self, width, height):
        pass


def make_env(positions, width=8, height=8):
    env = TestSnowPlowEnv.__new__(TestSnowPlowEnv)

    env.width = width
    env.height = height

    env.agents = [
        SimpleNamespace(
            index=agent_id,
            state=SimpleNamespace(
                pos=np.array(position, dtype=int)
            ),
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

    rewards = env.handle_actions({
        0: SnowPlowAction.right,
    })

    assert position(env, 0) == (2, 1)
    assert status(env, 0) == "moved"
    assert rewards == {0: 0}


def test_single_agent_do_nothing():
    env = make_env([(1, 1)])

    rewards = env.handle_actions({
        0: SnowPlowAction.do_nothing,
    })

    assert position(env, 0) == (1, 1)
    assert status(env, 0) == "waited"
    assert rewards == {0: 0}


def test_two_agents_request_same_empty_cell():
    env = make_env([
        (1, 1),
        (2, 2),
    ])

    env.handle_actions({
        0: SnowPlowAction.right,
        1: SnowPlowAction.up,
    })

    assert position(env, 0) == (1, 1)
    assert position(env, 1) == (2, 2)
    assert status(env, 0) == "blocked_same_destination"
    assert status(env, 1) == "blocked_same_destination"


def test_two_agents_cannot_swap_cells():
    env = make_env([
        (1, 1),
        (2, 1),
    ])

    env.handle_actions({
        0: SnowPlowAction.right,
        1: SnowPlowAction.left,
    })

    assert position(env, 0) == (1, 1)
    assert position(env, 1) == (2, 1)
    assert status(env, 0) == "blocked_swap"
    assert status(env, 1) == "blocked_swap"


def test_agent_cannot_enter_cell_of_stationary_agent():
    env = make_env([
        (1, 1),
        (2, 1),
    ])

