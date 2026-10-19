"""Experiment presets. CLI values override these, then dataclass defaults apply."""

from copy import deepcopy


def _preset(n_plows, observation, map_name, algo):
    return {
        "env_args": {
            "map": map_name,
            "n_plows": n_plows,
            "observation": observation,
            "view_size": 5,
            "max_steps": 1_000,
            "allow_shared_cells": n_plows > 1,
        },
        "snow_args": {},  # SnowConfig supplies the branch's current snow settings.
        "ppo_args": {
            "recurrent": observation == "local",
            "n_envs": 8,
            "rollout_len": 64,
            "lr": 3e-4,
            "hidden": 128,
            "seed": 0,
        },
        "training_args": {
            "algo": algo,
            "steps": 10_000_000,
            "checkpoint_every": 1_000_000,
        },
    }


EXPERIMENTS = {
    "exp1": _preset(1, "full", "network_1", "ppo"),
    "exp2": _preset(1, "local", "network_1", "ppo"),
    "exp3": _preset(5, "full", "network_3", "mappo"),
    "exp4": _preset(5, "local", "network_3", "mappo"),
}


def get_experiment(name):
    """Return an independent preset; no experiment means ordinary defaults."""
    if name is None:
        return {key: {} for key in ("env_args", "snow_args", "ppo_args", "training_args")}
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {name!r}")
    return deepcopy(EXPERIMENTS[name])
