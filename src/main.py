"""Command line entry point of the project.

    python src/main.py baselines --episodes 50                     # random + greedy reference scores
    python src/main.py play --policy greedy --render               # watch a policy in a window
    python src/main.py play --policy random --gif out.gif          # record an episode (no window needed)
    python src/main.py train --experiment exp1 --steps 500000      # train PPO / MAPPO
    python src/main.py evaluate --run runs/exp1_network_1_ppo_s0   # score a trained model
    python src/main.py play --policy model --run runs/exp1_network_1_ppo_s0 --render

See README.md for the full pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from snowplow.baselines import GreedyPolicy, RandomPolicy
from snowplow.config import SnowConfig, SnowplowConfig
from snowplow.evaluation import run_policy, run_trainer, summarize
from snowplow.experiments import EXPERIMENTS
from snowplow.factory import adapter_kwargs, config_from_dict, config_to_dict, make_env, make_env_fn
from snowplow.maps import MAPS


# ----------------------------------------------------------------------------- arguments
def add_env_args(p: argparse.ArgumentParser) -> None:
    """Options that define the environment. Any of them overrides the --experiment preset."""
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS), help="preset: plows / observation / algorithm")
    p.add_argument("--map", default="network_1", choices=sorted(MAPS), help="road network (small = quick debugging)")
    p.add_argument("--plows", type=int, help="number of snowplows (agents)")
    p.add_argument("--obs", choices=["full", "local"], help="what a plow sees: the whole map or a window")
    p.add_argument("--view", type=int, default=5, choices=[3, 5], help="window size for --obs local")
    p.add_argument("--max-steps", type=int, default=300, help="episode length limit")
    p.add_argument("--spawn-prob", type=float, default=None, help="chance per step of a new cloud (default 0.02)")
    p.add_argument("--initial-clouds", type=int, default=None, help="clouds that already snowed at the start (default 2)")


def build_config(args) -> SnowplowConfig:
    """Merge the experiment preset with explicit command line options."""
    preset = EXPERIMENTS.get(args.experiment, {})
    snow = {}
    if args.spawn_prob is not None:
        snow["cloud_spawn_prob"] = args.spawn_prob
    if args.initial_clouds is not None:
        snow["initial_clouds"] = args.initial_clouds
    return SnowplowConfig(
        n_plows=args.plows or preset.get("n_plows", 1),
        max_steps=args.max_steps,
        observation=args.obs or preset.get("observation", "full"),
        view_size=args.view,
        snow=SnowConfig(**snow),
    )


# ------------------------------------------------------------------------------ commands
def cmd_baselines(args) -> None:
    """Score the non-learning policies; this is the bar a trained agent has to beat."""
    config = build_config(args)
    rows = {}
    policies = [("random", RandomPolicy(seed=args.seed))]
    if config.observation == "full":  # greedy searches the whole map, so it needs the full view
        policies.append(("greedy", GreedyPolicy()))
    else:
        print("note: the greedy baseline needs --obs full and is skipped")
    for name, policy in policies:
        env = make_env(args.map, config)
        rows[name] = summarize(run_policy(env, policy, args.episodes, seed=args.seed))
    print_table(rows)
    if args.out:
        write_results(Path(args.out), rows, {"map": args.map, "snowplow": config_to_dict(config)})


def cmd_train(args) -> None:
    from algorithms import MAPPO, PPO, PPOConfig, plow_progress_lcr  # imported here: needs torch

    config = build_config(args)
    preset = EXPERIMENTS.get(args.experiment, {})
    algo = args.algo or preset.get("algo") or ("ppo" if config.n_plows == 1 else "mappo")
    recurrent = args.recurrent or preset.get("recurrent", False)
    reward_mode = args.reward_mode or ("individual" if algo == "ppo" else "team_mean")
    ppo_config = PPOConfig(
        n_envs=args.n_envs, rollout_len=args.rollout_len, lr=args.lr, hidden=args.hidden,
        recurrent=recurrent, reward_mode=reward_mode, reward_scale=args.reward_scale, seed=args.seed,
    )
    run_dir = Path(args.out or f"runs/{args.experiment or 'custom'}_{args.map}_{algo}_s{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    reward_transform = plow_progress_lcr(args.lcr_zeta) if args.lcr_zeta > 0 else None
    trainer = {"ppo": PPO, "mappo": MAPPO}[algo](
        make_env_fn(args.map, config), ppo_config, adapter_kwargs(config), reward_transform=reward_transform
    )
    print(f"training {algo} on map '{args.map}': {config.n_plows} plow(s), obs={config.observation}, "
          f"obs_dim={trainer.obs_dim}, recurrent={recurrent} -> {run_dir}")

    from dataclasses import asdict

    (run_dir / "config.json").write_text(json.dumps({
        "map": args.map, "algo": algo, "snowplow": config_to_dict(config), "ppo": asdict(ppo_config),
        "steps": args.steps, "lcr_zeta": args.lcr_zeta,
    }, indent=1))
    history = trainer.learn(args.steps, log_every=args.log_every)
    trainer.save(str(run_dir / "model.pt"))
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))

    results = summarize(run_trainer(trainer, args.eval_episodes, seed=args.seed + 10_000))
    write_results(run_dir, {"trained": results}, {"map": args.map, "algo": algo})
    print_table({"trained": results})
    print(f"saved model, history and metrics to {run_dir}/")


def load_trainer(run_dir: Path):
    """Rebuild a trainer from a run directory (config.json) and load its weights (model.pt)."""
    from algorithms import MAPPO, PPO, PPOConfig  # needs torch

    saved = json.loads((run_dir / "config.json").read_text())
    config = config_from_dict(saved["snowplow"])
    trainer = {"ppo": PPO, "mappo": MAPPO}[saved["algo"]](
        make_env_fn(saved["map"], config), PPOConfig(**saved["ppo"]), adapter_kwargs(config)
    )
    trainer.load(str(run_dir / "model.pt"))
    return trainer, saved["map"], config


def cmd_evaluate(args) -> None:
    trainer, _, _ = load_trainer(Path(args.run))
    results = summarize(run_trainer(trainer, args.episodes, seed=args.seed, deterministic=args.deterministic))
    print_table({"trained": results})
    write_results(Path(args.run), {"trained": results}, {"episodes": args.episodes}, name="evaluation")


def cmd_play(args) -> None:
    """Run ONE episode and show it (window and/or GIF)."""
    render_mode = "human" if args.render else None
    frames = []

    if args.policy == "model":
        from algorithms.adapter import ParallelEnvAdapter

        trainer, map_name, config = load_trainer(Path(args.run))
        env = make_env(map_name, config, render_mode)
        adapter = ParallelEnvAdapter(env, **adapter_kwargs(config))
        act = trainer.make_policy(deterministic=args.deterministic)
        obs, _, _ = adapter.reset(seed=args.seed)
        new_episode, done = True, False
        while not done:
            if args.gif:
                frames.append(env.get_frame(False, 24))
            res = adapter.step(act(obs, adapter.mask, new_episode))
            obs, new_episode, done = res.obs, False, res.done
        info = res.info.get("episode", {})
    else:
        config = build_config(args)
        if args.policy == "greedy" and config.observation != "full":
            raise SystemExit("the greedy policy needs the full view (--obs full)")
        env = make_env(args.map, config, render_mode)
        policy = RandomPolicy(args.seed) if args.policy == "random" else GreedyPolicy()
        obs, _ = env.reset(seed=args.seed)
        policy.reset()
        while True:
            if args.gif:
                frames.append(env.get_frame(False, 24))
            obs, _, terminated, truncated, infos = env.step(policy(obs))
            if all(terminated.values()) or any(truncated.values()):
                info = infos[0]["episode"]
                break

    if args.gif and frames:
        from PIL import Image

        frames.append(env.get_frame(False, 24))
        images = [Image.fromarray(f) for f in frames]
        images[0].save(args.gif, save_all=True, append_images=images[1:], duration=120, loop=0)
        print(f"saved {len(images)} frames to {args.gif}")
    env.close()
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}, indent=1))


# ------------------------------------------------------------------------------- output
def print_table(rows: dict[str, dict]) -> None:
    keys = ["success_rate", "total_reward", "steps", "fraction_clean", "snow_cleared", "repeated_work",
            "events/invalid_move", "events/agent_collision"]
    print(f"{'policy':10s}" + "".join(f"{k.split('/')[-1][:14]:>15s}" for k in keys))
    for name, row in rows.items():
        print(f"{name:10s}" + "".join(f"{row[k]:15.2f}" for k in keys))


def write_results(directory: Path, rows: dict, meta: dict, name: str = "metrics") -> None:
    """Write <name>.json and <name>.csv (one row per policy)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps({"meta": meta, "results": rows}, indent=1))
    with open(directory / f"{name}.csv", "w", newline="") as f:
        first = next(iter(rows.values()))
        writer = csv.writer(f)
        writer.writerow(["policy"] + list(first))
        for policy, row in rows.items():
            writer.writerow([policy] + [row[k] for k in first])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baselines", help="score the random and greedy reference policies")
    add_env_args(p)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", help="directory for metrics.json / metrics.csv")
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("play", help="run one episode and show it")
    add_env_args(p)
    p.add_argument("--policy", choices=["random", "greedy", "model"], default="greedy")
    p.add_argument("--run", help="run directory (for --policy model)")
    p.add_argument("--render", action="store_true", help="open a window")
    p.add_argument("--gif", help="save the episode as an animated GIF")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="model: take the most likely action")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("train", help="train PPO (1 plow) or MAPPO (several plows)")
    add_env_args(p)
    p.add_argument("--algo", choices=["ppo", "mappo"], help="default: from the experiment, else by number of plows")
    p.add_argument("--steps", type=int, default=500_000, help="total environment steps")
    p.add_argument("--recurrent", action="store_true", help="GRU policy (needed for --obs local)")
    p.add_argument("--reward-mode", choices=["individual", "team_sum", "team_mean"])
    p.add_argument("--lcr-zeta", type=float, default=0.0, help="MAPPO-LCR reward shaping strength (0 = off)")
    p.add_argument("--reward-scale", type=float, default=0.1, help="multiply training rewards by this (reporting stays raw)")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10, help="print every N PPO updates")
    p.add_argument("--out", help="run directory (default runs/<experiment>_<map>_<algo>_s<seed>)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="score a trained model")
    p.add_argument("--run", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--deterministic", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
