# test_snow_plow_env.py

import numpy as np

from Enums import EventType, MovementActions
from snow_plow_env import SnowPlowEnv


def make_road_network(width=8, height=8, obstacles=None):
    obstacles = obstacles or []
    return {
        "size": (height, width),
        "roads": [{"position": (0, 0), "size": (height, width)}],
        "highways": [],
        "intersections": [],
        "obstacles": obstacles,
    }


def make_env(positions, width=8, height=8, obstacles=None):
    env = SnowPlowEnv(
        agents=len(positions),
        max_steps=100,
        road_network=make_road_network(width, height, obstacles),
    )
    env.reset(seed=0)
    for agent, position in zip(env.agents, positions):
        agent.state.pos = np.array(position, dtype=int)

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
    assert rewards == {0: -1.0}
    assert [event.type for event in env.event_log.events()] == [EventType.MOVED]


def test_single_agent_do_nothing():
    env = make_env([(1, 1)])

    rewards = env.handle_actions(
        {
            0: MovementActions.DO_NOTHING,
        }
    )

    assert position(env, 0) == (1, 1)
    assert status(env, 0) == "waited"
    assert rewards == {0: -2.0}
    assert [event.type for event in env.event_log.events()] == [EventType.DO_NOTHING]


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
    assert [event.type for event in env.event_log.events()] == [
        EventType.AGENT_COLLISION,
        EventType.AGENT_COLLISION,
    ]


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

    env.handle_actions(
        {
            0: MovementActions.RIGHT,
            1: MovementActions.DO_NOTHING,
        }
    )

    assert position(env, 0) == (1, 1)
    assert position(env, 1) == (2, 1)
    assert status(env, 0) == "blocked_occupied_cell"
    assert status(env, 1) == "waited"


def test_agent_cannot_move_out_of_bounds():
    env = make_env([(0, 0)])

    env.handle_actions({0: MovementActions.UP})

    assert position(env, 0) == (0, 0)
    assert status(env, 0) == "blocked_invalid_action"
