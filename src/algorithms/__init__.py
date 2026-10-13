"""Learning algorithms, independent of any particular environment.

PPO      single agent (also used for the single-plow experiments)
MAPPO    centralised critic + shared decentralised actors (Snowplow, multi-plow)
IPPO     MAPPO with per-agent critics (baseline)
MADDPG   centralised critics + per-agent actors (Soccer, mixed coop/competitive)

Every algorithm talks to the environment through `ParallelEnvAdapter`, which
only needs the PettingZoo *Parallel* API (reset / step with one dict entry
per agent).
"""

from algorithms.adapter import ParallelEnvAdapter, flatten_obs
from algorithms.maddpg import MADDPG, MADDPGConfig
from algorithms.ppo import IPPO, MAPPO, PPO, PPOConfig
from algorithms.rewards import LocalCooperationReward, plow_progress_lcr, spgg_lcr

__all__ = [
    "ParallelEnvAdapter",
    "flatten_obs",
    "PPO",
    "MAPPO",
    "IPPO",
    "PPOConfig",
    "MADDPG",
    "MADDPGConfig",
    "LocalCooperationReward",
    "spgg_lcr",
    "plow_progress_lcr",
]
