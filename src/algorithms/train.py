"""Train an algorithm on an environment.

Examples (run from the repository root):
    python -m algorithms.train --algo ppo --env toy_coverage --agents 1 --steps 100000
    python -m algorithms.train --algo mappo --env toy_coverage --agents 3 --steps 300000 --reward-mode team_sum
    python -m algorithms.train --algo mappo --env snowplow --agents 2 --steps 100000
"""

import argparse
import json
from pathlib import Path

from Enums import MovementActions
from algorithms import IPPO, MADDPG, MAPPO, PPO, MADDPGConfig, PPOConfig
from algorithms.toy_envs import ToyChaseEnv, ToyCoverageEnv
from snow_plow_env import SnowPlowEnv
from Traffic_configs import road_network_1

ALGOS = {"ppo": PPO, "ippo": IPPO, "mappo": MAPPO, "maddpg": MADDPG}


def make_env_fn(name: str, agents: int, size: int, max_steps: int, partial: bool, env: SnowPlowEnv):
    if name == "toy_coverage":
        return (
            lambda: ToyCoverageEnv(
                n_agents=agents,
                size=size,
                max_steps=max_steps,
                partial=partial,
            ),
            {},
        )
    if name == "toy_chase":
        return lambda: ToyChaseEnv(size=size, max_steps=max_steps), {}
    if name == "snowplow":
        return (
            lambda: SnowPlowEnv(
                agents=agents,
                max_steps=max_steps,
                road_network=road_network_1,
            ),
            {
                "agent_ids": [agent.index for agent in env.agents],
                "n_actions": len(MovementActions),
                "action_decoder": lambda action: MovementActions(action),
            },
        )
    raise ValueError(f"unknown env {name!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=ALGOS, required=True)
    p.add_argument("--env", choices=["toy_coverage", "toy_chase", "snowplow"], default="snowplow")
    p.add_argument("--agents", type=int, default=1)
    p.add_argument("--size", type=int, default=4, help="grid size for toy environments")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--partial", action="store_true", help="partial observability (toy env: 3x3 window)")
    p.add_argument("--recurrent", action="store_true", help="GRU policy/critic (PPO family only)")
    p.add_argument("--steps", type=int, default=100_000, help="total environment steps")
    p.add_argument("--reward-mode", default="individual", choices=["individual", "team_sum", "team_mean"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--out", default="runs/latest", help="output directory (model + history)")
    args = p.parse_args()

    env_fn, adapter_kwargs = make_env_fn(
        args.env,
        args.agents,
        args.size,
        args.max_steps,
        args.partial,
    )
    if args.algo == "maddpg":
        algo = MADDPG(env_fn, MADDPGConfig(seed=args.seed), adapter_kwargs)
    else:
        cfg = PPOConfig(seed=args.seed, recurrent=args.recurrent, reward_mode=args.reward_mode)
        algo = ALGOS[args.algo](env_fn, cfg, adapter_kwargs)

    history = algo.learn(args.steps)
    result = algo.evaluate(args.eval_episodes)
    print("evaluation:", result)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    algo.save(str(out / "model.pt"))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "eval.json").write_text(json.dumps({"args": vars(args), "eval": result}, indent=1))
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
