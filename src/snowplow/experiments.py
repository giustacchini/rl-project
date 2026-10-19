"""The four experiments of the project as presets.

They differ only in the number of plows and in what a plow can see; the
environment and the algorithms are the same code.
"""

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


checkpoint_every = 1_000_000  # Save model every n steps
max_episode_steps = 1_000  # Episode length limit.
total_steps = 10_000_000  # Total environment steps.


# EXPERIMENTS = {
#     # 1 plow, sees the whole map: the first thing that must work.
#     "exp1": {
#         # Strictly for mosaic env setup
#         "env_args": {
#             "map": "network_1",
#             "n_plows": 1,  # number of snowplows (agents)
#             "observation": "full",  # what a plow sees: the whole map or a window
#             "view": 5,  # window size for --obs local
#             "max_steps": max_episode_steps,  # episode length limit
#             "exclusive_cells": True,  # forbid plows from sharing a cell (old blocking rules, no cooperative plowing)
#         },
#         # Algorithm settings
#         # Read more in src/algorithms/ppo.py/PPOConfig
#         "ppo_args": {
#             "algo": "ppo",
#             "recurrent": False,
#             "n_envs": 8,
#             "rollout-len": 64,
#             "lr": 0.0003,
#             "hidden": 128,
#             "seed": 0,
#             "steps": total_steps,
#             # "max_steps": max_episode_steps,
#             "checkpoint_every": checkpoint_every,
#         },
#     },
#     # 1 plow, sees only a 5x5 window: needs memory -> recurrent (GRU) policy.
#     "exp2": {
#         "n_plows": 1,
#         "observation": "local",
#         "view": 5,
#         "algo": "ppo",
#         "recurrent": True,
#         "map": "network_1",
#         "n_envs": 8,
#         "rollout-len": 64,
#         "lr": 0.0003,
#         "hidden": 128,
#         "seed": 0,
#         "steps": total_steps,
#         "max_steps": max_episode_steps,
#         "checkpoint_every": checkpoint_every,
#     },
#     # Multiple plows, everybody sees the whole map.
#     "exp3": {
#         "n_plows": 5,
#         "observation": "full",
#         "algo": "mappo",
#         "recurrent": False,
#         "map": "network_3",
#         "n_envs": 8,
#         "rollout-len": 64,
#         "lr": 0.0003,
#         "hidden": 128,
#         "seed": 0,
#         "steps": total_steps,
#         "max_steps": max_episode_steps,
#         "checkpoint_every": checkpoint_every,
#     },
#     # Multiple plows, local windows, recurrent.
#     "exp4": {
#         "n_plows": 5,
#         "observation": "local",
#         "view": 5,
#         "algo": "mappo",
#         "recurrent": True,
#         "map": "network_3",
#         "n_envs": 8,
#         "rollout-len": 64,
#         "lr": 0.0003,
#         "hidden": 128,
#         "seed": 0,
#         "steps": total_steps,
#         "max_steps": max_episode_steps,
#         "checkpoint_every": checkpoint_every,
#     },
# }


def get_experiment(exp):
    # EXPERIMENTS.get(exp)
    snow_config_args = SNOW_DEFAULT_ARGS
    env_grid_args = ENV_GRID_DEFAULT_ARGS
    ppo_args = PPO_DEFAULT_ARGS

    if exp == "exp1":
        # 1 plow, sees the whole map: the first thing that must work.
        # ENV_GRID_ARGS
        env_grid_args["map"] = "network_1"
        env_grid_args["n_plows"] = 1
        env_grid_args["observation"] = "full"
        env_grid_args["view"] = 5
        env_grid_args["max_episode_steps"] = max_episode_steps
        env_grid_args["exclusive_cells"] = True

        # SNOW_CONFIG_ARGS
        snow_config_args["clear_per_pass"] = 1
        snow_config_args["scale_spawn_with_plows"] = False

        # PPO_ARGS / train args
        ppo_args["algo"] = "ppo"
        ppo_args["recurrent"] = False
        ppo_args["rollout_len"] = 64
        ppo_args["steps"] = total_steps
        ppo_args["checkpoint_every"] = checkpoint_every

        # ====== Other potential useful algo parameters ======
        # ppo_args["gamma"]
        # ppo_args["gae_lambda"]
        # ppo_args["clip"]
        # ppo_args["epochs"]
        # ppo_args["n_minibatches"]
        # ppo_args["anneal_lr"]
        # ppo_args["ent_coef"]
        # ppo_args["vf_coef"]
        # ppo_args["max_grad_norm"]
        # ppo_args["layers"]
        # ppo_args["agent_id"]
        # ppo_args["reward_mode"]
        # ppo_args["activation"]
        # ppo_args["reward_scale"]
        # ppo_args["device"]

    elif exp == "exp2":
        # ENV_GRID_ARGS
        env_grid_args["map"] = "network_1"
        env_grid_args["n_plows"] = 1
        env_grid_args["observation"] = "local"
        env_grid_args["view"] = 5
        env_grid_args["max_steps"] = max_episode_steps
        env_grid_args["exclusive_cells"] = True

        # SNOW_CONFIG_ARGS
        snow_config_args["clear_per_pass"] = 1
        snow_config_args["scale_spawn_with_plows"] = False

        # PPO_ARGS / train args
        ppo_args["algo"] = "ppo"
        ppo_args["recurrent"] = True
        ppo_args["rollout_len"] = 64
        ppo_args["steps"] = total_steps
        ppo_args["checkpoint_every"] = checkpoint_every

    elif exp == "exp3":
        # ENV_GRID_ARGS
        env_grid_args["map"] = "network_3"
        env_grid_args["n_plows"] = 1
        env_grid_args["observation"] = "full"
        env_grid_args["view"] = 5
        env_grid_args["max_steps"] = max_episode_steps
        env_grid_args["exclusive_cells"] = True

        # SNOW_CONFIG_ARGS
        snow_config_args["clear_per_pass"] = 1
        snow_config_args["scale_spawn_with_plows"] = False

        # PPO_ARGS / train args
        ppo_args["algo"] = "mappo"
        ppo_args["recurrent"] = False
        ppo_args["rollout_len"] = 64
        ppo_args["steps"] = total_steps
        ppo_args["checkpoint_every"] = checkpoint_every

    elif exp == "exp4":
        # ENV_GRID_ARGS
        env_grid_args["map"] = "network_3"
        env_grid_args["n_plows"] = 1
        env_grid_args["observation"] = "local"
        env_grid_args["view"] = 5
        env_grid_args["max_steps"] = max_episode_steps
        env_grid_args["exclusive_cells"] = True

        # SNOW_CONFIG_ARGS
        snow_config_args["clear_per_pass"] = 1
        snow_config_args["scale_spawn_with_plows"] = False

        # PPO_ARGS / train args
        ppo_args["algo"] = "mappo"
        ppo_args["recurrent"] = True
        ppo_args["rollout_len"] = 64
        ppo_args["steps"] = total_steps
        ppo_args["checkpoint_every"] = checkpoint_every

    return {"snow_config_args": snow_config_args, "env_grid_args": env_grid_args, "ppo_args": ppo_args}
