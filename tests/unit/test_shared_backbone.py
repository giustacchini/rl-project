# python
# test_shared_backbone.py
"""
Issue #5: tests for reset/seeding, movement-rule conflicts, and
rendering -- run against the REAL SnowPlowEnv / mosaic_multigrid stack
(not the SimpleNamespace stub used in test_snow_plow_env.py). This is
what originally caught the `agent.state.pos.copy()` bug: the stub's
`pos` was a numpy array, but the real `AgentState.pos` returns a plain
tuple for a single agent, so `.copy()` only fails against the real lib.
"""

import numpy as np
import pytest

from Enums import MovementActions
from shared.rendering import ascii_grid
from snow_plow_env import SnowPlowEnv


def make_env(num_agents=1, width=8, height=8, max_steps=50, seed=1):
    env = SnowPlowEnv(agents=num_agents, width=width, height=height, max_steps=max_steps)
    env.reset(seed=seed)
    return env


def set_positions(env, positions):
    for agent, pos in zip(env.agents, positions):
        agent.state.pos = np.array(pos, dtype=int)


def status_of(env, agent_id):
    return env.last_movement_results[agent_id]["status"]


# ---------------------------------------------------------------------
# Reset / seeding (issue #5: "tests for reset and seeding")
# ---------------------------------------------------------------------


def test_reset_returns_valid_agent_positions():
    env = make_env(num_agents=3)
    for agent in env.agents:
        x, y = agent.state.pos
        assert 0 <= x < env.width
        assert 0 <= y < env.height
        # Agents must not start on the border wall.
        assert env.grid.get(x, y) is None or env.grid.get(x, y).can_overlap()


def test_same_seed_is_reproducible():
    e1 = SnowPlowEnv(agents=4, width=10, height=10, max_steps=50)
    e2 = SnowPlowEnv(agents=4, width=10, height=10, max_steps=50)
    e1.reset(seed=123)
    e2.reset(seed=123)
    pos1 = [tuple(a.state.pos) for a in e1.agents]
    pos2 = [tuple(a.state.pos) for a in e2.agents]
    assert pos1 == pos2


def test_different_seed_usually_differs():
    e1 = SnowPlowEnv(agents=4, width=10, height=10, max_steps=50)
    e2 = SnowPlowEnv(agents=4, width=10, height=10, max_steps=50)
    e1.reset(seed=1)
    e2.reset(seed=2)
    pos1 = [tuple(a.state.pos) for a in e1.agents]
    pos2 = [tuple(a.state.pos) for a in e2.agents]
    assert pos1 != pos2


# ---------------------------------------------------------------------
# Movement conflicts against the real env (issue #5: same-cell / swap /
# out-of-bounds tests)
# ---------------------------------------------------------------------


def test_same_destination_conflict_blocks_both():
    env = make_env(num_agents=2)
    set_positions(env, [(1, 1), (2, 2)])
    env.handle_actions({0: MovementActions.RIGHT, 1: MovementActions.UP})
    assert status_of(env, 0) == "blocked_same_destination"
    assert status_of(env, 1) == "blocked_same_destination"


def test_swap_conflict_blocks_both():
    env = make_env(num_agents=2)
    set_positions(env, [(1, 1), (2, 1)])
    env.handle_actions({0: MovementActions.RIGHT, 1: MovementActions.LEFT})
    assert status_of(env, 0) == "blocked_swap"
    assert status_of(env, 1) == "blocked_swap"


def test_move_into_border_wall_is_blocked():
    env = make_env(num_agents=1, width=8, height=8)
    set_positions(env, [(1, 0)])  # row 0 sits just inside the border wall
    env.handle_actions({0: MovementActions.UP})  # would move onto y=-1 -> wall row
    assert status_of(env, 0) == "blocked_invalid_action"
    assert tuple(env.agents[0].state.pos) == (1, 0)


def test_move_into_wall_cell_is_blocked():
    env = make_env(num_agents=1, width=8, height=8)
    set_positions(env, [(1, 1)])
    env.handle_actions({0: MovementActions.LEFT})  # (0,1) is still inside border wall
    assert status_of(env, 0) == "blocked_invalid_action"
    assert tuple(env.agents[0].state.pos) == (1, 1)


# ---------------------------------------------------------------------
# Rendering (issue #5: "two generic agents can be visualized moving")
# ---------------------------------------------------------------------


def test_ascii_render_shows_every_agent():
    env = make_env(num_agents=2, width=6, height=6)
    set_positions(env, [(1, 1), (3, 3)])
    text = ascii_grid(env)
    assert "0" in text
    assert "1" in text
    assert text.count("\n") == env.height - 1


def test_rgb_frame_renders_without_crashing():
    env = make_env(num_agents=2, width=6, height=6)
    frame = env.get_frame(tile_size=16)
    assert frame.shape[0] > 0 and frame.shape[1] > 0 and frame.shape[2] == 3


# ---------------------------------------------------------------------
# Many-episode stability (issue #5: "confirm environment can run
# repeatedly without crashing")
# ---------------------------------------------------------------------


def test_many_random_episodes_do_not_crash():
    env = SnowPlowEnv(agents=2, width=8, height=8, max_steps=20)
    rng = np.random.default_rng(0)
    for episode in range(20):
        env.reset(seed=episode)
        for _ in range(20):
            actions = {agent.index: list(MovementActions)[rng.integers(0, len(MovementActions))] for agent in env.agents}
            env.handle_actions(actions)
