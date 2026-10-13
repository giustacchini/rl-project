"""Train an algorithm on an environment.

Examples (run from the repo root):
    python src/train.py --algo ppo    --env toy_coverage --agents 1 --steps 100000
    python src/train.py --algo mappo  --env toy_coverage --agents 3 --steps 300000 --reward-mode team_sum
    python src/train.py --algo mappo  --env toy_coverage --agents 2 --partial --recurrent
    python src/train.py --algo maddpg --env toy_chase    --steps 100000

Once `SnowPlowEnv` has a Parallel-style `step()`, add it to ENVS below (see docs/ALGORITHMS.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from algorithms import IPPO, MADDPG, MAPPO, PPO, MADDPGConfig, PPOConfig
from algorithms.toy_envs import ToyChaseEnv, ToyCoverageEnv

ALGOS = {"ppo": PPO, "ippo": IPPO, "mappo": MAPPO, "maddpg": MADDPG}


def make_env_fn(name: str, agents: int, size: int, max_steps: int, partial: bool):
    if name == "toy_coverage":
        return lambda: ToyCoverageEnv(n_agents=agents, size=size, max_steps=max_steps, partial=partial), {}
    if name == "toy_chase":
        return lambda: ToyChaseEnv(size=size, max_steps=max_steps), {}
    if name == "snowplow":
        raise NotImplementedError(
            "SnowPlowEnv does not expose a Parallel-style step() yet. Register it here with the right "
            "ParallelEnvAdapter kwargs once it does (see docs/ALGORITHMS.md, section 'Plugging in the real environments')."
        )
    raise ValueError(f"unknown env {name!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=ALGOS, required=True)
    p.add_argument("--env", default="toy_coverage")
    p.add_argument("--agents", type=int, default=1)
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--partial", action="store_true", help="partial observability (toy env: 3x3 window)")
    p.add_argument("--recurrent", action="store_true", help="GRU policy/critic (PPO family only)")
    p.add_argument("--steps", type=int, default=100_000, help="total environment steps")
    p.add_argument("--reward-mode", default="individual", choices=["individual", "team_sum", "team_mean"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--out", default="runs/latest", help="output directory (model + history)")
    args = p.parse_args()

    env_fn, adapter_kwargs = make_env_fn(args.env, args.agents, args.size, args.max_steps, args.partial)
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
