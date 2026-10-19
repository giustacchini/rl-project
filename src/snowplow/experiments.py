"""Experiment presets. CLI values override these, then dataclass defaults apply."""

from copy import deepcopy

# Taken from the MultiGridEnv init constructor
ENV_GRID_DEFAULT_ARGS = {
    "mission_space": "maximize reward",
    "agents": 1,
    "grid_size": None,
    "width": None,
    "height": None,
    "max_steps": 100,
    "see_through_walls": False,
    "agent_view_size": 7,
    "allow_agent_overlap": True,
    "joint_reward": False,
    "success_termination_mode": "any",
    "failure_termination_mode": "all",
    "render_mode": None,  # 'Human' or 'rgb_array' renders the grid in a window
    "screen_size": 640,  # Around 500 could be good in our case.
    "highlight": False,
    "tile_size": 32,
    "agent_pov": False,
    "global_obs": False,  # Depends on the experiment
}

# Snow configuration settings.
# Read more in src/snowplow/config.py
SNOW_DEFAULT_ARGS = {
    # "fixed_spawn_prob": True, # do not scale the cloud spawn chance with the number of plows
    "max_snow": 3,  # int
    "clear_per_pass": 2,  # int
    "plow_time_single": (0, 1, 5, 10),  # tuple
    "plow_time_team": (0, 1, 2, 3),  # tuple
    "initial_clouds": 10,  # int
    "cloud_spawn_prob": 0.02,  # float
    "scale_spawn_with_plows": True,  # bool
    "cloud_min_size": 3,  # int
    "cloud_max_size": 9,  # int
    "cloud_min_lifetime": 1,  # int
    "cloud_max_lifetime": 1,  # int
    "snow_per_step": 1,  # int
    "storm_steps": None,  # int | None = None
}

# DEFAULT PPO ARGS
PPO_DEFAULT_ARGS = {
    "n_envs": 8,  # int: environments stepped per rollout (more -> less noisy)
    "rollout_len": 128,  # int: steps per env per update (-> BPTT length when recurrent)
    "gamma": 0.99,  # float:
    "gae_lambda": 0.95,  # float:
    "clip": 0.2,  # float:
    "epochs": 4,  # int:
    "n_minibatches": 4,  # int:  split over *streams* (env x agent), never over time
    "lr": 3e-4,  # float:
    "anneal_lr": True,  # bool:
    "ent_coef": 0.01,  # float:
    "vf_coef": 0.5,  # float:
    "max_grad_norm": 0.5,  # float:
    "hidden": 128,  # int:
    "layers": 2,  # int:
    "recurrent": False,  # bool:  GRU in actor and critic (partial observability)
    "agent_id": True,  # bool:   append one-hot agent id (needed for a shared actor to specialise)
    "reward_mode": "individual",  # str:  "individual" | "team_sum" | "team_mean"
    "activation": "tanh",  # str:  "tanh" (PPO default) or "relu" (MAPPO-LCR paper)
    "reward_scale": 1.0,  # float:    training rewards are multiplied by this (keeps value targets small when returns are in the 100s)
    "seed": 0,  # int:
    "device": "cpu",  # str: "cpu" or "cuda:0"
}


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
