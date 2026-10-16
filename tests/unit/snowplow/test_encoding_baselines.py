"""Observation encoding for the learners, and the reference (non-learning) policies."""

from Enums import ACTIONS
from snowplow.baselines import GreedyPolicy, RandomPolicy
from snowplow.config import SnowplowConfig
from snowplow.encoding import encode_observation
from snowplow.env import GRID_CHANNELS, SnowPlowEnv
from snowplow.evaluation import run_policy, summarize
from snowplow.road_network_configs import MAPS


def test_full_observation_vector_layout(make_env):
    env = make_env([(1, 1), (3, 3)])
    obs, *_ = env.step({0: 0, 1: 0})
    vec = encode_observation(obs[0])
    cells = env.height * env.width
    assert vec.shape == ((len(GRID_CHANNELS) + 2) * cells + len(ACTIONS),)
    self_map = vec[len(GRID_CHANNELS) * cells : (len(GRID_CHANNELS) + 1) * cells].reshape(env.height, env.width)
    others_map = vec[(len(GRID_CHANNELS) + 1) * cells : (len(GRID_CHANNELS) + 2) * cells].reshape(env.height, env.width)
    assert self_map[1, 1] == 1 and self_map.sum() == 1
    assert others_map[3, 3] == 1 and others_map.sum() == 1


def test_local_observation_does_not_reveal_the_absolute_position(make_env):
    env_a = make_env([(2, 2)], observation="local", view_size=3)
    env_b = make_env([(5, 5)], observation="local", view_size=3)
    va = encode_observation(env_a.step({0: 0})[0][0])
    vb = encode_observation(env_b.step({0: 0})[0][0])
    assert va.shape == vb.shape == ((len(GRID_CHANNELS) + 2) * 9 + len(ACTIONS),)
    assert (va == vb).all()  # identical surroundings -> identical observation


def test_random_policy_only_picks_legal_actions():
    env = SnowPlowEnv(MAPS["network_1"], SnowplowConfig(n_plows=2))
    obs, _ = env.reset(seed=0)
    policy = RandomPolicy(seed=1)
    for _ in range(100):
        actions = policy(obs)
        for i, a in actions.items():
            assert obs[i]["action_mask"][a]
        obs, *_ = env.step(actions)


def test_greedy_clears_a_simple_map_quickly():
    env = SnowPlowEnv(MAPS["small"], SnowplowConfig(max_steps=200))
    summary = summarize(run_policy(env, GreedyPolicy(), episodes=10, seed=0))
    assert summary["success_rate"] >= 0.5
    assert summary["events/invalid_move"] == 0  # it only uses legal moves


def test_greedy_beats_random():
    config = SnowplowConfig(max_steps=150)
    greedy = summarize(run_policy(SnowPlowEnv(MAPS["small"], config), GreedyPolicy(), 10, seed=5))
    random = summarize(run_policy(SnowPlowEnv(MAPS["small"], config), RandomPolicy(0), 10, seed=5))
    assert greedy["total_reward"] > random["total_reward"]
