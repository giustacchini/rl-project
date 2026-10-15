"""Learning algorithms, independent of any particular environment.

PPO      single agent (the single-plow experiments)
MAPPO    centralised critic + shared decentralised actors (multi-plow experiments)
MADDPG   centralised critics + per-agent actors (reserved for the soccer / ethics project;
         it has no action masking yet, so it is not wired into the snowplow CLI)

Every algorithm talks to the environment through `ParallelEnvAdapter`, which only
needs a PettingZoo-*Parallel*-style env (reset / step with one dict entry per agent).

PyTorch is imported lazily: `algorithms.adapter` and `algorithms.rewards` work without it.
"""

from algorithms.adapter import ParallelEnvAdapter, StepResult, flatten_obs
from algorithms.rewards import LocalCooperationReward, plow_progress_lcr, spgg_lcr

_TORCH_NAMES = {
    "PPO": "algorithms.ppo",
    "MAPPO": "algorithms.ppo",
    "PPOConfig": "algorithms.ppo",
    "MADDPG": "algorithms.maddpg",
    "MADDPGConfig": "algorithms.maddpg",
}

__all__ = [
    "ParallelEnvAdapter", "StepResult", "flatten_obs",
    "LocalCooperationReward", "plow_progress_lcr", "spgg_lcr",
    *_TORCH_NAMES,
]


def __getattr__(name):  # imported on first use, so torch is only needed when a trainer is
    if name in _TORCH_NAMES:
        import importlib

        return getattr(importlib.import_module(_TORCH_NAMES[name]), name)
    raise AttributeError(f"module 'algorithms' has no attribute {name!r}")
