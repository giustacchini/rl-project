"""ParallelEnvAdapter against the real snowplow env (no PyTorch needed)."""

import numpy as np
import pytest

from algorithms.adapter import ParallelEnvAdapter
from snowplow.config import SnowplowConfig
from snowplow.factory import adapter_kwargs, make_env


@pytest.mark.parametrize("n_plows,observation", [(1, "full"), (2, "full"), (2, "local")])
def test_adapter_produces_fixed_shape_arrays_and_masks(n_plows, observation):
    config = SnowplowConfig(n_plows=n_plows, observation=observation, max_steps=20)
    adapter = ParallelEnvAdapter(make_env("small", config), **adapter_kwargs(config))
    obs, state, alive = adapter.reset(seed=0)
    assert obs.shape == (n_plows, adapter.obs_dim) and state.shape == (adapter.state_dim,)
    assert alive.tolist() == [1.0] * n_plows
    assert adapter.mask.shape == (n_plows, 5) and adapter.mask[:, 0].all()  # waiting is always legal

    for _ in range(20):
        actions = np.array([np.flatnonzero(adapter.mask[i])[0] for i in range(n_plows)])
        res = adapter.step(actions)
        assert res.obs.shape == obs.shape and res.reward.shape == (n_plows,)
        assert res.mask.shape == (n_plows, 5) and "positions" in res.info
    assert res.done and res.truncated and not res.terminated
    assert "episode" in res.info  # metrics are available to the learner at the end of the episode


def test_adapter_sums_to_the_team_reward():
    config = SnowplowConfig(n_plows=2, max_steps=5)
    adapter = ParallelEnvAdapter(make_env("small", config), **adapter_kwargs(config))
    adapter.reset(seed=0)
    total = 0.0
    for _ in range(5):
        res = adapter.step(np.array([0, 0]))  # both wait: -2 each
        total += res.reward.sum()
    assert total == -20.0
