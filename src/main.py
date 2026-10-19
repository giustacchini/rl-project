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
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from snowplow.baselines import CooperativeGreedyPolicy, GreedyPolicy, RandomPolicy
from snowplow.config import SnowConfig, SnowplowConfig
from snowplow.evaluation import run_policy, run_trainer, summarize
from snowplow.experiments import EXPERIMENTS
from snowplow.factory import (
    adapter_kwargs,
    config_from_dict,
    config_to_dict,
    make_env,
    make_env_fn,
)
from snowplow.road_network_configs import MAPS


# ----------------------------------------------------------------------------- arguments
def add_env_args(p: argparse.ArgumentParser) -> None:
    """Options that define the environment.

    Explicit command-line values override an experiment preset. Values omitted
    on the command line come from the preset, then SnowplowConfig defaults.
    """
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS), help="preset: plows / observation / algorithm")
    p.add_argument("--map", choices=sorted(MAPS), help="road network (small = quick debugging)")
    p.add_argument("--plows", type=int, help="number of snowplows (agents)")
    p.add_argument("--obs", choices=["full", "local"], help="what a plow sees: the whole map or a window")
    p.add_argument("--view", type=int, choices=[3, 5], help="window size for --obs local")
    p.add_argument("--max-steps", type=int, help="episode length limit")
    p.add_argument(
        "--spawn-prob",
        type=float,
        default=None,
        help="chance per step of a new cloud PER PLOW: the real chance is this x number of plows (default 0.02)",
    )
    p.add_argument("--fixed-spawn-prob", action="store_true", help="do not scale the cloud spawn chance with the number of plows")
    p.add_argument("--initial-clouds", type=int, default=None, help="clouds that already snowed at the start (default 2)")
    # Cooperative plowing (see SnowConfig.plow_time_single / plow_time_team).
    p.add_argument(
        "--plow-time-single",
        type=int,
        nargs="+",
        default=None,
        metavar="T",
        help="steps ONE plow needs per snow level 1..max_snow (default: 1 4 8)",
    )
    p.add_argument(
        "--plow-time-team",
        type=int,
        nargs="+",
        default=None,
        metavar="T",
        help="steps 2+ plows on the same cell need per snow level 1..max_snow (default: 1 2 3)",
    )
    p.add_argument(
        "--exclusive-cells", action="store_true", help="forbid plows from sharing a cell (old blocking rules, no cooperative plowing)"
    )


def resolve_environment(args: argparse.Namespace) -> tuple[str, SnowplowConfig]:
    """Resolve the map and environment config used by a command."""
    experiment = EXPERIMENTS.get(args.experiment, {})
    env_args = experiment["env_args"]
    algo_args = experiment["algo_args"]
    snow_args = experiment["snow_args"]
    snow = {}
    if args.spawn_prob is not None:
        snow["cloud_spawn_prob"] = args.cloud_spawn_prob
    if args.initial_clouds is not None:
        snow["initial_clouds"] = args.initial_clouds
    if getattr(args, "fixed_spawn_prob", False):
        snow["scale_spawn_with_plows"] = False
    # Plow times are given for levels 1..max_snow on the command line; level 0 needs no work.
    if getattr(args, "plow_time_single", None):
        snow["plow_time_single"] = (0, *args.plow_time_single)
    if getattr(args, "plow_time_team", None):
        snow["plow_time_team"] = (0, *args.plow_time_team)

    config_defaults = SnowplowConfig()
    config = SnowplowConfig(
        n_plows=args.plows if args.plows is not None else preset.get("n_plows", config_defaults.n_plows),
        max_steps=args.max_steps if args.max_steps is not None else preset.get("max_steps", config_defaults.max_steps),
        observation=args.obs if args.obs is not None else preset.get("observation", config_defaults.observation),
        view_size=args.view if args.view is not None else preset.get("view_size", config_defaults.view_size),
        allow_shared_cells=not getattr(args, "exclusive_cells", False),
        snow=SnowConfig(**snow),
    )
    map_name = args.map if args.map is not None else preset.get("map", "network_3")
    return map_name, config


def build_config(args: argparse.Namespace) -> SnowplowConfig:
    """Build the resolved environment config for callers that only need it."""
    return resolve_environment(args)[1]


def create_run_dir(args: argparse.Namespace, map_name: str, algo: str) -> Path:
    """Create a run directory without reusing a default directory."""
    if args.out:
        run_dir = Path(args.out)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path("runs") / f"{args.experiment or 'custom'}_{map_name}_{algo}_seed{args.seed}_{timestamp}"
    run_dir = base
    suffix = 1
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            run_dir = Path(f"{base}_{suffix}")
            suffix += 1


# ------------------------------------------------------------------------------ commands
def cmd_baselines(args) -> None:
    """Score the non-learning policies; this is the bar a trained agent has to beat."""
    map_name, config = resolve_environment(args)
    rows = {}
    policies = [("random", RandomPolicy(seed=args.seed))]
    if config.observation == "full":  # greedy searches the whole map, so it needs the full view
        policies.append(("greedy", GreedyPolicy()))
        if config.n_plows > 1:  # helping a teammate only makes sense with 2+ plows
            policies.append(("greedy_coop", CooperativeGreedyPolicy.from_config(config.snow)))
    else:
        print("note: the greedy baseline needs --obs full and is skipped")
    for name, policy in policies:
        env = make_env(map_name, config)
        rows[name] = summarize(run_policy(env, policy, args.episodes, seed=args.seed))
    print_table(rows)
    if args.out:
        write_results(Path(args.out), rows, {"map": map_name, "snowplow": config_to_dict(config)})


def cmd_train(args) -> None:
    from algorithms import (  # imported here: needs torch
        MAPPO,
        PPO,
        PPOConfig,
        plow_progress_lcr,
    )

    # Read experiment setup configuration
    # TODO: FIX CORRECT ENV MAPPINGS
    map_name, config = resolve_environment(args)
    preset = EXPERIMENTS.get(args.experiment, {})

    algo = args.algo or preset.get("algo") or ("ppo" if config.n_plows == 1 else "mappo")
    recurrent = args.recurrent or preset.get("recurrent", False)
    reward_mode = args.reward_mode or ("individual" if algo == "ppo" else "team_mean")
    checkpoint_every = args.checkpoint_every or preset.get("checkpoint_every")

    ppo_config = PPOConfig(
        n_envs=args.n_envs,
        rollout_len=args.rollout_len,
        lr=args.lr,
        hidden=args.hidden,
        recurrent=recurrent,
        reward_mode=reward_mode,
        reward_scale=args.reward_scale,
        seed=args.seed,
    )
    run_dir = create_run_dir(args, map_name, algo)
    print("run_dir:", run_dir)

    reward_transform = plow_progress_lcr(args.lcr_zeta) if args.lcr_zeta > 0 else None
    trainer = {"ppo": PPO, "mappo": MAPPO}[algo](
        make_env_fn(map_name, config), ppo_config, adapter_kwargs(config), reward_transform=reward_transform
    )
    print(
        f"training {algo} on map '{map_name}': {config.n_plows} plow(s), obs={config.observation}, "
        f"obs_dim={trainer.obs_dim}, recurrent={recurrent}"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "map": map_name,
                "algo": algo,
                "snowplow": config_to_dict(config),
                "ppo": asdict(ppo_config),
                "steps": args.steps,
                "lcr_zeta": args.lcr_zeta,
            },
            indent=1,
        )
    )

    # Here is the training start point
    history = trainer.learn(args.steps, log_every=args.log_every, callback=make_checkpoint_callback(trainer, run_dir, checkpoint_every))
    trainer.save(str(run_dir / "model.pt"))
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))

    results = summarize(run_trainer(trainer, args.eval_episodes, seed=args.seed + 10_000))
    write_results(run_dir, {"trained": results}, {"map": map_name, "algo": algo})
    print_table({"trained": results})
    print(f"saved model, history and metrics to {run_dir}/")


def make_checkpoint_callback(trainer, run_dir: Path, every: int | None):
    """Callback for trainer.learn() that saves the model every `every` environment steps.

    Checkpoints go to <run_dir>/checkpoints/model_step_<env steps>.pt. The trainer reports its
    step count once per update (rollout_len x n_envs steps), so a checkpoint is written on the
    first update that reaches or passes each multiple of `every`. Returns None when disabled.
    """
    print("every:", every)
    if not every or every <= 0:
        return None
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    next_save = {"at": every}  # mutable so the closure can advance it

    def callback(stats: dict) -> None:
        steps = int(stats["env_steps"])
        if steps < next_save["at"]:
            return
        path = checkpoint_dir / f"model_step_{steps}.pt"
        trainer.save(str(path))
        print(f"checkpoint saved: {path}")
        while next_save["at"] <= steps:  # skip multiples already passed by a long update
            next_save["at"] += every

    return callback


def load_trainer(run_dir: Path, checkpoint: str | None = None):
    """Rebuild a trainer from a run directory (config.json) and load its weights.

    By default the final model.pt is loaded; `checkpoint` selects another weights file, either a
    path or a file name inside <run_dir>/checkpoints (e.g. model_step_100000.pt).
    """
    from algorithms import MAPPO, PPO, PPOConfig  # needs torch

    saved = json.loads((run_dir / "config.json").read_text())
    config = config_from_dict(saved["snowplow"])
    trainer = {"ppo": PPO, "mappo": MAPPO}[saved["algo"]](
        make_env_fn(saved["map"], config), PPOConfig(**saved["ppo"]), adapter_kwargs(config)
    )
    weights = run_dir / "model.pt"
    if checkpoint:
        weights = Path(checkpoint)
        if not weights.exists():
            weights = run_dir / "checkpoints" / checkpoint
        if not weights.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint}")
    trainer.load(str(weights))
    return trainer, saved["map"], config


def cmd_evaluate(args) -> None:
    trainer, _, _ = load_trainer(Path(args.run), args.checkpoint)
    results = summarize(run_trainer(trainer, args.episodes, seed=args.seed, deterministic=args.deterministic))
    print_table({"trained": results})
    write_results(Path(args.run), {"trained": results}, {"episodes": args.episodes}, name="evaluation")


def cmd_play(args) -> None:
    """Run ONE episode and show it (window and/or GIF)."""
    render_mode = "human" if args.render else None
    frames = []

    if args.policy == "model":
        from algorithms.adapter import ParallelEnvAdapter

        trainer, map_name, config = load_trainer(Path(args.run), args.checkpoint)
        env = make_env(map_name, config, render_mode, render_fps=args.render_fps)
        adapter = ParallelEnvAdapter(env, **adapter_kwargs(config))
        act = trainer.make_policy(deterministic=args.deterministic)
        obs, _, _ = adapter.reset(seed=args.seed)
        new_episode, done = True, False
        while not done:
            if args.gif:
                frames.append(env.get_frame(False, 24))
            res = adapter.step(act(obs, adapter.mask, new_episode))
            # check when snow clouds appear in a certain area of the grid (for debugging)
            # print(env.step_count, [(r, c, int(env.snow_field.snow[r, c]), bool(env.snow_field.cloud_mask()[r, c])) for r, c in [(2, 0), (3, 0), (4, 0), (4, 1)]])
            obs, new_episode, done = res.obs, False, res.done
        info = res.info.get("episode", {})
    else:
        map_name, config = resolve_environment(args)
        if args.policy in ("greedy", "greedy_coop") and config.observation != "full":
            raise SystemExit("the greedy policy needs the full view (--obs full)")
        env = make_env(map_name, config, render_mode, render_fps=args.render_fps)
        if args.policy == "random":
            policy = RandomPolicy(args.seed)
        elif args.policy == "greedy_coop":
            policy = CooperativeGreedyPolicy.from_config(config.snow)
        else:
            policy = GreedyPolicy()
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
    keys = [
        "total_reward",
        "steps",
        "fraction_clean",
        "snow_cleared",
        "repeated_work",
        "events/invalid_move",
        "events/agent_collision",
        "cells_cleared_together",  # cooperative plowing: cells cleaned by 2+ plows together
        "cooperation_rate",  # ... as a fraction of all cleaned cells
    ]
    print(f"{'policy':12s}" + "".join(f"{k.split('/')[-1][:14]:>15s}" for k in keys))
    for name, row in rows.items():
        print(f"{name:12s}" + "".join(f"{row.get(k, 0.0):15.2f}" for k in keys))


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
    p.add_argument("--policy", choices=["random", "greedy", "greedy_coop", "model"], default="greedy")
    p.add_argument("--run", help="run directory (for --policy model)")
    p.add_argument("--checkpoint", help="load this checkpoint instead of model.pt (file in <run>/checkpoints or a path)")
    p.add_argument("--render", action="store_true", help="open a window")
    p.add_argument("--gif", help="save the episode as an animated GIF")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="model: take the most likely action")
    p.add_argument("--render-fps", type=int, default=20, dest="render_fps")
    p.add_argument("--martin-likes-snow", action="store_true", help="Martin likes snow")
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
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        metavar="STEPS",
        help="also save the model every STEPS environment steps to <run>/checkpoints/",
    )
    p.add_argument("--log-every", type=int, default=10, help="print every N PPO updates")
    p.add_argument("--out", help="run directory (default includes the current date and time)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="score a trained model")
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", help="load this checkpoint instead of model.pt (file in <run>/checkpoints or a path)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--deterministic", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
