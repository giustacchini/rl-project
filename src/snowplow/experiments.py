"""The four experiments of the project as presets.

They differ only in the number of plows and in what a plow can see; the
environment and the algorithms are the same code.
"""

EXPERIMENTS = {
    # 1 plow, sees the whole map: the first thing that must work.
    "exp1": {"n_plows": 1, "observation": "full", "algo": "ppo", "recurrent": False, "map": "network_3"},
    # 1 plow, sees only a 5x5 window: needs memory -> recurrent (GRU) policy.
    "exp2": {"n_plows": 1, "observation": "local", "algo": "ppo", "recurrent": True, "map": "network_3"},
    # several plows, everybody sees the whole map.
    "exp3": {"n_plows": 2, "observation": "full", "algo": "mappo", "recurrent": False, "map": "network_3"},
    # several plows, local windows, recurrent.
    "exp4": {"n_plows": 2, "observation": "local", "algo": "mappo", "recurrent": True, "map": "network_3"},
}
