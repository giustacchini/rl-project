"""Run a policy for several episodes and report the snowplow metrics.

The environment computes the metrics itself (env.episode_metrics()); here we only
run episodes and average them, so a random baseline, the greedy baseline and a
trained network are all measured the same way.
"""

from __future__ import annotations

import numpy as np

# Metrics that are averaged in the summary.
SUMMARY_KEYS = (
    "total_reward", "steps", "success", "fraction_clean", "snow_cleared", "snow_fallen",
    "snow_remaining", "repeated_work", "events/invalid_move", "events/agent_collision",
    "main_streets_cleared", "secondary_streets_cleared",
    # Cooperative plowing: how often the plows share the work (see SnowPlowEnv._plow_snow).
    "cells_cleared", "cells_cleared_together", "cooperation_rate", "plowing_steps", "shared_cell_steps",
)


def summarize(episodes: list[dict]) -> dict:
    """Mean (and std of the reward) over a list of per-episode metric dicts."""
    out = {key: float(np.mean([float(ep.get(key, 0.0)) for ep in episodes])) for key in SUMMARY_KEYS}
    out["std_total_reward"] = float(np.std([ep["total_reward"] for ep in episodes]))
    out["episodes"] = len(episodes)
    return out


def run_policy(env, policy, episodes: int, seed: int = 1000) -> list[dict]:
    """Evaluate `policy(obs_dict) -> {plow_id: action}` on a raw SnowPlowEnv. Episode i uses seed+i."""
    results = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        policy.reset()
        while True:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            if all(terminated.values()) or any(truncated.values()):
                results.append(info[0]["episode"])
                break
    return results


def run_trainer(trainer, episodes: int, seed: int = 1000, deterministic: bool = False) -> list[dict]:
    """Evaluate a trained PPO/MAPPO object (its policy runs through the adapter, with action masks)."""
    from algorithms.adapter import ParallelEnvAdapter  # local import: needs torch only for trainers

    adapter = ParallelEnvAdapter(trainer.env_fn(), **trainer.adapter_kwargs)
    act = trainer.make_policy(deterministic)
    results = []
    for i in range(episodes):
        obs, _, _ = adapter.reset(seed=seed + i)
        new_episode = True
        while True:
            res = adapter.step(act(obs, adapter.mask, new_episode))
            obs, new_episode = res.obs, False
            if res.done:
                results.append(res.info["episode"])
                break
    return results
