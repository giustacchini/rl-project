"""Builds environments and the pieces the learners need, from names and a config.

Every script (train / evaluate / play / baselines) goes through here, so they all
construct the environment in exactly the same way.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from Enums import ACTIONS
from snowplow.config import SnowConfig, SnowplowConfig
from snowplow.encoding import encode_observation
from snowplow.env import SnowPlowEnv
from snowplow.maps import MAPS


def make_env(map_name: str, config: SnowplowConfig, render_mode: str | None = None, render_fps: int = 20) -> SnowPlowEnv:
    if map_name not in MAPS:
        raise ValueError(f"unknown map {map_name!r}; choose one of {sorted(MAPS)}")
    return SnowPlowEnv(MAPS[map_name], config, render_mode=render_mode, render_fps=render_fps)


def make_env_fn(map_name: str, config: SnowplowConfig) -> Callable[[], SnowPlowEnv]:
    """Zero-argument function returning a fresh env (what the PPO trainers expect)."""
    return lambda: make_env(map_name, config)


def adapter_kwargs(config: SnowplowConfig) -> dict:
    """Arguments for `ParallelEnvAdapter` so learners can talk to the snowplow env."""
    return {
        "agent_ids": list(range(config.n_plows)),
        "n_actions": len(ACTIONS),
        "obs_encoder": encode_observation,
    }


def config_to_dict(config: SnowplowConfig) -> dict:
    return asdict(config)


def config_from_dict(data: dict) -> SnowplowConfig:
    data = dict(data)
    data["snow"] = SnowConfig(**data.get("snow", {}))
    return SnowplowConfig(**data)
