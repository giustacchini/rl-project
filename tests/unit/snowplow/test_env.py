"""SnowPlowEnv behaviour. Each test maps to an item of the checklist in RL_ENVIRONMENT_TASK_SPLIT.md."""

import numpy as np
import pytest
from conftest import place, set_snow

from Enums import ACTIONS, EventType, MovementActions, SnowplowEvent
from snowplow.config import SnowConfig, SnowplowConfig
from snowplow.env import GRID_CHANNELS, SnowPlowEnv
from snowplow.road_network_configs import MAPS, road_network_1
from snowplow.snow import Cloud

UP, DOWN, RIGHT, LEFT, WAIT = (
    MovementActions.UP,
    MovementActions.DOWN,
    MovementActions.RIGHT,
    MovementActions.LEFT,
    MovementActions.DO_NOTHING,
)


def pos(env, i=0):
    return tuple(int(v) for v in env.agents[i].state.pos)


def status(env, i=0):
    return env.last_movement_results[i]["status"]


# ------------------------------------------------------------------ reset / seed
def test_reset_returns_observation_for_every_plow():
    env = SnowPlowEnv(road_network_1, SnowplowConfig(n_plows=3))
    obs, info = env.reset(seed=1)
    assert set(obs) == {0, 1, 2} and set(info) == {0, 1, 2}
    for o in obs.values():
        assert o["grid"].shape == (len(GRID_CHANNELS), env.height, env.width)
        assert o["action_mask"].shape == (len(ACTIONS),)


@pytest.mark.parametrize("seed", range(5))
def test_plows_start_on_distinct_drivable_cells(seed):
    env = SnowPlowEnv(road_network_1, SnowplowConfig(n_plows=4))
    env.reset(seed=seed)
    cells = [pos(env, i) for i in range(4)]
    assert len(set(cells)) == 4
    assert all(env.road_map.passable[y, x] for x, y in cells)


def test_same_seed_same_episode_different_seed_different_start():
    def rollout(seed):
        env = SnowPlowEnv(road_network_1, SnowplowConfig())
        env.reset(seed=seed)
        rng, trace = np.random.default_rng(0), []
        for _ in range(60):
            mask = env.get_action_mask(0)
            _, r, *_ = env.step({0: int(rng.choice(np.flatnonzero(mask)))})
            trace.append((r[0], pos(env)))
        return trace, env.snow_field.snow.copy()

    (t1, s1), (t2, s2), (t3, s3) = rollout(7), rollout(7), rollout(8)
    assert t1 == t2 and (s1 == s2).all()
    assert t1 != t3 or not (s1 == s3).all()


# --------------------------------------------------------------- movement rules
def test_move_and_wait_rewards(make_env):
    env = make_env([(1, 1)])
    _, rewards, *_ = env.step({0: RIGHT})
    assert pos(env) == (2, 1) and status(env) == "moved" and rewards == {0: -1.0}
    _, rewards, *_ = env.step({0: WAIT})
    assert pos(env) == (2, 1) and status(env) == "waited" and rewards == {0: 0.0}


def test_leaving_the_map_is_blocked_and_penalised(make_env):
    env = make_env([(0, 0)])
    _, rewards, *_ = env.step({0: UP})
    assert pos(env) == (0, 0) and status(env) == "blocked_invalid_action" and rewards == {0: 0.0}
    assert [e.type for e in env.event_log.events()] == [EventType.INVALID_MOVE]


# The three blocking rules below only apply with exclusive cells (allow_shared_cells=False).
# With shared cells (the default, for cooperative plowing) see test_cooperative_plowing.py.
def test_two_plows_requesting_the_same_cell_both_stay(make_env):
    env = make_env([(1, 1), (2, 2)], allow_shared_cells=False)
    _, rewards, *_ = env.step({0: RIGHT, 1: UP})  # both want (2, 1)
    assert pos(env, 0) == (1, 1) and pos(env, 1) == (2, 2)
    assert status(env, 0) == status(env, 1) == "blocked_same_destination"
    assert rewards[0] == rewards[1] == 0.0
    assert sum(rewards.values()) == 0.0


def test_plows_cannot_swap(make_env):
    env = make_env([(1, 1), (2, 1)], allow_shared_cells=False)
    env.step({0: RIGHT, 1: LEFT})
    assert pos(env, 0) == (1, 1) and pos(env, 1) == (2, 1)
    assert status(env, 0) == status(env, 1) == "blocked_swap"


def test_plow_cannot_enter_cell_of_a_waiting_plow(make_env):
    env = make_env([(1, 1), (2, 1)], allow_shared_cells=False)
    env.step({0: RIGHT, 1: WAIT})
    assert pos(env, 0) == (1, 1) and status(env, 0) == "blocked_occupied_cell" and status(env, 1) == "waited"


def test_missing_action_means_wait_and_malformed_action_raises(make_env):
    env = make_env([(1, 1), (3, 3)])
    env.step({0: RIGHT})  # plow 1 sent nothing
    assert status(env, 1) == "waited"
    with pytest.raises(ValueError):
        env.step({0: "north", 1: WAIT})
    with pytest.raises(ValueError):
        env.step({0: 17, 1: WAIT})


def test_integer_actions_index_the_actions_tuple(make_env):
    env = make_env([(1, 1)])
    env.step({0: ACTIONS.index(DOWN)})
    assert pos(env) == (1, 2)


def test_wrong_way_on_the_highway_is_rejected():
    env = SnowPlowEnv(road_network_1, SnowplowConfig(snow=SnowConfig(initial_clouds=0, cloud_spawn_prob=0)))
    env.reset(seed=0)
    place(env, [(5, 4)])  # northern lane: flows west
    env.step({0: RIGHT})
    assert pos(env) == (5, 4) and status(env) == "blocked_invalid_action"
    env.step({0: LEFT})
    assert pos(env) == (4, 4) and status(env) == "moved"
    place(env, [(5, 5)])  # southern lane: flows east
    env.step({0: LEFT})
    assert pos(env) == (5, 5) and status(env) == "blocked_invalid_action"


def test_action_mask_matches_traffic_rules():
    # No snow: a plow on a snowy cell would be busy (only DO_NOTHING allowed), see test_cooperative_plowing.py.
    env = SnowPlowEnv(road_network_1, SnowplowConfig(snow=SnowConfig(initial_clouds=0, cloud_spawn_prob=0)))
    env.reset(seed=0)
    place(env, [(5, 4)])
    mask = env.get_action_mask(0)
    assert mask[ACTIONS.index(WAIT)] and mask[ACTIONS.index(LEFT)]
    assert mask.sum() == 2  # wait + the single legal direction
    # Taking ANY masked-in action never produces an invalid move, on every cell of the map.
    for y in range(env.height):
        for x in range(env.width):
            if not env.road_map.passable[y, x]:
                continue
            for i in np.flatnonzero(env.road_map.allowed[y, x]):
                place(env, [(x, y)])
                env.step({0: int(i)})
                assert status(env) in ("moved", "waited")
                env._finished = False


# ------------------------------------------------------------------------- snow
def test_plow_cleans_a_level_2_cell_in_4_steps_and_is_rewarded(make_env):
    env = make_env([(1, 1)])
    set_snow(env, {(2, 1): 2, (5, 5): 1})  # a second snowy cell keeps the episode running
    _, rewards, *_ = env.step({0: RIGHT})  # arrival = first working step
    assert env.snow_field.snow[1, 2] == 2 and rewards[0] == -1  # not done yet, only the move cost
    for _ in range(2):
        env.step({0: WAIT})
    assert env.snow_field.snow[1, 2] == 2
    _, rewards, *_ = env.step({0: WAIT})  # 4th step: done, all snow removed at once
    assert env.snow_field.snow[1, 2] == 0
    assert rewards[0] == 2  # one reward point per snow layer


def test_a_plow_standing_on_snow_plows_it(make_env):
    env = make_env([(1, 1)])
    set_snow(env, {(1, 1): 1, (5, 5): 1})
    env.step({0: WAIT})  # level 1 needs a single step
    assert env.snow_field.snow[1, 1] == 0


# Street completion is still reported as an event (for the metrics) but no longer rewarded.
def test_secondary_street_event_without_reward():
    two_streets = {  # street A = the top row, street B = the two lower rows
        "size": (3, 3),
        "roads": [{"position": (0, 0), "size": (1, 3)}, {"position": (1, 0), "size": (2, 3)}],
        "highways": [],
        "intersections": [],
        "obstacles": [],
    }
    env = SnowPlowEnv(two_streets, SnowplowConfig(snow=SnowConfig(initial_clouds=0, cloud_spawn_prob=0)))
    env.reset(seed=0)
    place(env, [(0, 0)])
    set_snow(env, {(1, 0): 1, (1, 2): 3})  # the only snow of street A is on (1, 0)
    _, rewards, *_ = env.step({0: RIGHT})  # finishes street A only
    assert rewards[0] == -1 + 1  # move cost + level-1 cell, no street bonus
    assert SnowplowEvent.SECONDARY_STREET_CLEARED in [e.type for e in env.event_log.events()]


def test_main_street_reward():
    env = SnowPlowEnv(road_network_small, SnowplowConfig(snow=SnowConfig(initial_clouds=0, cloud_spawn_prob=0)))
    env.reset(seed=0)
    place(env, [(3, 2)])  # northern highway lane, flows west
    set_snow(env, {(2, 2): 1, (0, 6): 3})
    _, rewards, *_ = env.step({0: LEFT})
    assert rewards[0] == -1 + 1  # move cost + level-1 cell, no street bonus
    assert SnowplowEvent.MAIN_STREET_CLEARED in [e.type for e in env.event_log.events()]


def test_clouds_add_snow_but_are_invisible(make_env):
    env = make_env([(0, 0)])
    before = env.get_observable_grid()
    env.snow_field.clouds.append(Cloud(cells=np.array([[4, 4], [5, 4]]), ttl=3))
    assert (env.get_observable_grid() == before).all()  # a cloud alone changes nothing the plow sees
    obs, *_ = env.step({0: WAIT})
    assert env.snow_field.snow[4, 4] == 1 and env.snow_field.snow[4, 5] == 1  # it snowed
    for o in obs.values():  # observations expose snow, never cloud positions
        assert o["grid"].shape[0] == len(GRID_CHANNELS) and "clouds" not in o


# ------------------------------------------------------------ termination / reset
def test_cleaning_everything_does_not_terminate_before_max_steps(make_env):
    env = make_env([(1, 1)])
    set_snow(env, {(2, 1): 1})
    obs, rewards, terminated, truncated, info = env.step({0: RIGHT})
    assert terminated == {0: False} and truncated == {0: False}
    assert rewards[0] == -1 + 1  # move + level-1 cell (street completion is no longer rewarded)
    env.step({0: WAIT})  # episodes terminate only at max_steps


def test_episode_truncates_at_max_steps(make_env):
    env = make_env([(1, 1)], max_steps=5)
    set_snow(env, {(6, 6): 3})
    for t in range(5):
        _, _, terminated, truncated, info = env.step({0: WAIT})
        assert terminated[0] is False
    assert truncated[0] is True and info[0]["episode"]["steps"] == 5


def test_no_snow_ever_does_not_count_as_success(make_env):
    env = make_env([(1, 1)], max_steps=3)
    _, _, terminated, _, _ = env.step({0: RIGHT})
    assert terminated[0] is False


def test_reset_starts_a_fresh_episode(make_env):
    env = make_env([(1, 1)], max_steps=2)
    set_snow(env, {(6, 6): 3})
    env.step({0: WAIT}), env.step({0: WAIT})
    env.reset(seed=3)
    assert env.step_count == 0 and env.episode_metrics()["steps"] == 0
    env.step({0: WAIT})  # works again


def test_metrics_are_reported(make_env):
    env = make_env([(1, 1)])
    set_snow(env, {(2, 1): 3, (3, 1): 0, (6, 6): 3})
    env.step({0: RIGHT})  # onto the level-3 cell: busy for 8 steps
    for _ in range(7):
        env.step({0: RIGHT})  # ignored while busy
    env.step({0: RIGHT})  # free again: goes onto a clean cell
    m = env.episode_metrics()
    assert m["steps"] == 9 and m["snow_cleared"] == 3 and m["repeated_work"] == 1
    assert m["events/moved"] == 2 and m["events/plowing"] == 7 and m["plowing_steps"] == 8
    assert m["cells_cleared"] == 1 and m["cooperation_rate"] == 0.0 and 0 < m["fraction_clean"] < 1


# ------------------------------------------------------------------ robustness
@pytest.mark.parametrize("map_name,n_plows", [("small", 1), ("small", 3), ("network_1", 2)])
def test_many_random_episodes_do_not_crash(map_name, n_plows):
    env = SnowPlowEnv(MAPS[map_name], SnowplowConfig(n_plows=n_plows, max_steps=60))
    rng = np.random.default_rng(0)
    for episode in range(8):
        env.reset(seed=episode)
        for _ in range(60):
            actions = {i: int(rng.integers(len(ACTIONS))) for i in range(n_plows)}  # including illegal ones
            obs, rewards, terminated, truncated, info = env.step(actions)
            positions = [pos(env, i) for i in range(n_plows)]
            assert len(set(positions)) == n_plows  # plows never share a cell
            assert all(env.road_map.passable[y, x] for x, y in positions)
            assert env.snow_field.snow.max() <= 3 and env.snow_field.snow.min() >= 0
            if all(terminated.values()) or any(truncated.values()):
                break


def test_state_vector_and_rendering(make_env):
    env = make_env([(1, 1), (2, 2)])
    assert env.state().shape == ((len(GRID_CHANNELS) + 2) * env.height * env.width,)
    frame = env.get_frame(highlight=False, tile_size=16)
    assert frame.shape == (env.height * 16, env.width * 16, 3)
    set_snow(env, {(4, 4): 3})
    assert not (env.get_frame(False, 16) == frame).all()  # snow shows up (tile cache is not stale)
    env.snow_field.clouds.append(Cloud(cells=np.array([[0, 7]]), ttl=2))
    assert not (env.get_frame(False, 16) == frame).all()
