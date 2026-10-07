# python
# scripts/random_episode_runner.py
"""
Run many episodes with random actions and print a short summary per
episode.
Usage:
    random_episode_runner.py --episodes 50
"""

import argparse

import numpy as np

from Enums import MovementActions
from shared.rendering import print_grid
from snow_plow_env import SnowPlowEnv

WEIRD_MOVEMENT_STATUSES = {
    "blocked_invalid_action",
    "blocked_same_destination",
    "blocked_swap",
    "blocked_occupied_cell",
}


def run(num_episodes: int, num_agents: int, width: int, height: int, max_steps: int, render_last: bool, seed: int):
    env = SnowPlowEnv(agents=num_agents, width=width, height=height, max_steps=max_steps)
    rng = np.random.default_rng(seed)

    status_counts = {}

    for episode in range(num_episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))

        for step in range(max_steps):
            actions = {agent.index: list(MovementActions)[rng.integers(0, len(MovementActions))] for agent in env.agents}
            print(actions)
            env.handle_actions(actions)

            weird_results = [record for record in env.last_movement_results.values() if record["status"] in WEIRD_MOVEMENT_STATUSES]
            if weird_results:
                statuses = sorted({record["status"] for record in weird_results})
                print(f"\n!!! WEIRD MOVEMENT STATUS AT EPISODE {episode}, " f"STEP {step}: {', '.join(statuses).upper()} !!!")
                print_grid(env)

            for record in env.last_movement_results.values():
                status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1

            # if render_last and episode == num_episodes - 1:
            # print(f"--- final state of episode {episode} ---")
            print_grid(env)

    print(f"\nRan {num_episodes} episodes x {max_steps} steps x {num_agents} agents " f"without crashing.")
    print("Movement status counts across all episodes:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:30s} {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--render-last", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(args.episodes, args.agents, args.width, args.height, args.max_steps, args.render_last, args.seed)
