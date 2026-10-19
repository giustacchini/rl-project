"""Cooperative plowing: shared cells, time-based plowing per snow level, busy plows, fair rewards.

Default plow times (SnowConfig):
    one plow       level 1 / 2 / 3 -> 1 / 5 / 10 steps
    two+ plows     level 1 / 2 / 3 -> 1 / 2 / 3 steps
The step on which a plow arrives on a cell already counts as a working step.
"""

import json

import numpy as np
import pytest
from conftest import DRY, open_map, place, set_snow

from Enums import ACTIONS, MovementActions, SnowplowEvent
from snowplow.baselines import CooperativeGreedyPolicy, GreedyPolicy
from snowplow.config import SnowConfig, SnowplowConfig
from snowplow.encoding import encode_observation
from snowplow.env import GRID_CHANNELS, SnowPlowEnv
from snowplow.evaluation import run_policy, summarize
from snowplow.factory import config_from_dict, config_to_dict
from snowplow.road_network_configs import MAPS

UP, DOWN, RIGHT, LEFT, WAIT = (
    MovementActions.UP,
    MovementActions.DOWN,
    MovementActions.RIGHT,
    MovementActions.LEFT,
    MovementActions.DO_NOTHING,
)
KEEP_RUNNING = {(7, 7): 1}  # an extra snowy cell nobody touches


def pos(env, i=0):
    return tuple(int(v) for v in env.agents[i].state.pos)


def status(env, i=0):
    return env.last_movement_results[i]["status"]


def steps_until_clean(env, cell, actions, limit=20):
    """Step with `actions` until `cell` is clean; return how many steps it took."""
    x, y = cell
    for n in range(1, limit + 1):
        env.step(actions)
        if env.snow_field.snow[y, x] == 0:
            return n
    raise AssertionError("cell never became clean")


# ------------------------------------------------------------------ config
def test_plow_time_lookup():
    cfg = SnowConfig()
    assert [cfg.plow_time(level, 1) for level in (1, 2, 3)] == [1, 5, 10]
    assert [cfg.plow_time(level, 2) for level in (1, 2, 3)] == [1, 2, 3]
    assert cfg.plow_time(3, 5) == 3  # more than two plows use the team times
    assert cfg.plow_time(0, 1) == 0 and cfg.plow_time(2, 0) == 0


def test_bad_plow_times_are_rejected():
    with pytest.raises(ValueError):
        SnowConfig(plow_time_single=(0, 1, 4))  # needs max_snow + 1 entries
    with pytest.raises(ValueError):
        SnowConfig(plow_time_team=(0, 0, 2, 3))  # a snowy cell takes >= 1 step


def test_config_survives_a_json_roundtrip():
    cfg = SnowplowConfig(n_plows=2, snow=SnowConfig(plow_time_single=(0, 2, 5, 9)))
    back = config_from_dict(json.loads(json.dumps(config_to_dict(cfg))))
    assert back == cfg and back.snow.plow_time_single == (0, 2, 5, 9)


# ------------------------------------------------------------------ shared cells
def test_two_plows_can_move_onto_the_same_cell(make_env):
    env = make_env([(1, 1), (2, 2)])
    env.step({0: RIGHT, 1: UP})  # both want (2, 1)
    assert pos(env, 0) == pos(env, 1) == (2, 1)
    assert status(env, 0) == status(env, 1) == "moved"


def test_a_plow_can_join_a_waiting_plow_and_plows_can_pass_each_other(make_env):
    env = make_env([(1, 1), (2, 1)])
    env.step({0: RIGHT, 1: WAIT})
    assert pos(env, 0) == pos(env, 1) == (2, 1)
    env = make_env([(1, 1), (2, 1)])
    env.step({0: RIGHT, 1: LEFT})
    assert pos(env, 0) == (2, 1) and pos(env, 1) == (1, 1)


# ------------------------------------------------------------------ plowing time
@pytest.mark.parametrize("level,expected", [(1, 1), (2, 5), (3, 10)])
def test_single_plow_time_per_level(make_env, level, expected):
    env = make_env([(1, 1)])
    set_snow(env, {(2, 1): level, **KEEP_RUNNING})
    env.step({0: RIGHT})  # arrival counts as the first working step
    took = 1 if env.snow_field.snow[1, 2] == 0 else 1 + steps_until_clean(env, (2, 1), {0: WAIT})
    assert took == expected


@pytest.mark.parametrize("level,expected", [(1, 1), (2, 2), (3, 3)])
def test_two_plows_together_are_faster(make_env, level, expected):
    env = make_env([(1, 1), (3, 1)])
    set_snow(env, {(2, 1): level, **KEEP_RUNNING})
    env.step({0: RIGHT, 1: LEFT})  # both arrive on (2, 1) at the same time
    took = 1 if env.snow_field.snow[1, 2] == 0 else 1 + steps_until_clean(env, (2, 1), {0: WAIT, 1: WAIT})
    assert took == expected


def test_a_late_helper_speeds_up_the_rest_of_the_work(make_env):
    env = make_env([(1, 1), (4, 1)])
    set_snow(env, {(2, 1): 3, **KEEP_RUNNING})
    env.step({0: RIGHT, 1: LEFT})  # plow 0 starts alone (1/10), plow 1 is on its way
    env.step({0: WAIT, 1: LEFT})  # plow 0: 1/10 done before this step; plow 1 now at (2, 1) too -> team speed 1/3
    assert pos(env, 1) == (2, 1) and env.snow_field.snow[1, 2] == 3
    # Work so far: 1/10 (alone) + 1/3 (together) = 0.433; the remaining 0.567 needs 2 more team steps.
    took = steps_until_clean(env, (2, 1), {0: WAIT, 1: WAIT})
    assert took == 2  # 4 steps in total instead of 10 alone


def test_snow_falling_during_plowing_slows_the_rest_down(make_env):
    env = make_env([(1, 1)])
    set_snow(env, {(1, 1): 2, **KEEP_RUNNING})
    env.step({0: WAIT})
    env.step({0: WAIT})  # 40% of the level-2 work (2/5) is done
    env.snow_field.snow[1, 1] = 3  # more snow falls: the remaining 60% needs level-3 speed (1/10)
    assert steps_until_clean(env, (1, 1), {0: WAIT}) == 6


# ------------------------------------------------------------------ busy plows
def test_a_busy_plow_can_only_wait_and_ignores_moves(make_env):
    env = make_env([(1, 1)])
    set_snow(env, {(2, 1): 2, **KEEP_RUNNING})
    env.step({0: RIGHT})
    mask = env.get_action_mask(0)
    assert mask.tolist() == [a == WAIT for a in ACTIONS]
    obs, *_ = env.step({0: RIGHT})  # ignored: the plow keeps plowing
    assert pos(env) == (2, 1) and status(env) == "plowing"
    assert SnowplowEvent.PLOWING in [e.type for e in env.event_log.events()]
    assert obs[0]["grid"][GRID_CHANNELS.index("plow_progress"), 1, 2] == pytest.approx(0.4)
    for _ in range(3):
        env.step({0: WAIT})  # done after 5 steps in total
    assert env.get_action_mask(0)[ACTIONS.index(RIGHT)]  # free again
    env.step({0: RIGHT})
    assert pos(env) == (3, 1)


# ------------------------------------------------------------------ rewards
def test_team_reward_is_split_by_the_work_done(make_env):
    env = make_env([(1, 1), (3, 1)])
    set_snow(env, {(2, 1): 3, **KEEP_RUNNING})
    total = np.zeros(2)
    for actions in ({0: RIGHT, 1: LEFT}, {0: WAIT, 1: WAIT}, {0: WAIT, 1: WAIT}):
        _, rewards, *_ = env.step(actions)
        total += [rewards[0], rewards[1]]
    assert env.snow_field.snow[1, 2] == 0
    assert total.tolist() == [1.5, 1.5]  # half of the 3 layers each; movement has no reward
    m = env.episode_metrics()
    assert m["cells_cleared_together"] == 1 and m["cooperation_rate"] == 1.0


def test_the_team_earns_the_same_per_cell_alone_or_together(make_env):
    def clearing_reward(n_plows):
        env = make_env([(2, 1)] * n_plows)
        set_snow(env, {(2, 1): 3, **KEEP_RUNNING})
        events = []
        for _ in range(10):
            env.step({i: WAIT for i in range(n_plows)})
            events += [e for e in env.event_log.events() if e.type == SnowplowEvent.CELL_CLEARED]
        return sum(e.data["reward_scale"] for e in events)

    assert clearing_reward(1) == pytest.approx(3) and clearing_reward(2) == pytest.approx(3)


# ------------------------------------------------------------------ observations
def test_a_teammate_on_the_same_cell_is_visible_in_the_encoding(make_env):
    env = make_env([(2, 1), (2, 1)])
    obs = env.gen_obs()
    cells = env.height * env.width
    vec = encode_observation(obs[0])
    others = vec[(len(GRID_CHANNELS) + 1) * cells : (len(GRID_CHANNELS) + 2) * cells].reshape(env.height, env.width)
    assert others[1, 2] == 1.0


# ------------------------------------------------------------------ baselines + rendering
def test_cooperative_greedy_sends_a_helper(make_env):
    env = make_env([(1, 1), (5, 1)])
    set_snow(env, {(1, 1): 3})  # plow 0 is busy on a heavy cell, plow 1 is idle 4 cells away
    actions = CooperativeGreedyPolicy()(env.gen_obs())
    assert actions[0] == ACTIONS.index(WAIT) and actions[1] == ACTIONS.index(LEFT)
    actions = GreedyPolicy()(env.gen_obs())  # the solo greedy leaves the teammate alone
    assert actions[1] == ACTIONS.index(WAIT)


def test_cooperative_greedy_cooperates_on_a_real_map():
    config = SnowplowConfig(n_plows=2, max_steps=150)
    env = SnowPlowEnv(MAPS["network_1"], config)
    coop = summarize(run_policy(env, CooperativeGreedyPolicy.from_config(config.snow), episodes=3, seed=0))
    solo = summarize(run_policy(SnowPlowEnv(MAPS["network_1"], config), GreedyPolicy(), episodes=3, seed=0))
    # The solo greedy only shares a cell by accident (driving over a teammate's cell).
    assert coop["cells_cleared_together"] > solo["cells_cleared_together"]


def test_rendering_with_plowing_in_progress(make_env):
    env = SnowPlowEnv(open_map(), SnowplowConfig(n_plows=2, snow=DRY), render_mode="rgb_array")
    env.reset(seed=0)
    place(env, [(2, 1), (2, 1)])
    set_snow(env, {(2, 1): 3})
    env.step({0: WAIT, 1: WAIT})
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3


# ------------------------------------------------------------------ reward per snow level
@pytest.mark.parametrize("level", [1, 2, 3])
def test_cleaning_a_cell_pays_its_snow_level(make_env, level):
    env = make_env([(2, 1)])
    set_snow(env, {(2, 1): level, **KEEP_RUNNING})
    total = 0.0
    while env.snow_field.snow[1, 2] > 0:
        _, rewards, *_ = env.step({0: WAIT})
        total += rewards[0]
    assert total == level  # 1 / 2 / 3, no other reward while plowing
    assert SnowplowEvent.CELL_CLEARED in [e.type for e in env.event_log.events()]


# ------------------------------------------------------------------ snowfall scales with plows
@pytest.mark.parametrize("n_plows,expected", [(1, 0.02), (2, 0.04), (5, 0.10)])
def test_cloud_spawn_chance_scales_with_the_number_of_plows(n_plows, expected):
    env = SnowPlowEnv(open_map(), SnowplowConfig(n_plows=n_plows))
    assert env.snow_field.spawn_prob == pytest.approx(expected)


def test_cloud_spawn_chance_can_stay_fixed_and_is_capped():
    assert SnowConfig(scale_spawn_with_plows=False).spawn_prob(5) == pytest.approx(0.02)
    assert SnowConfig(cloud_spawn_prob=0.4).spawn_prob(5) == 1.0
