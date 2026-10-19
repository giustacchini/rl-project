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
from snowplow.experiments import get_experiment
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
    p.add_argument("--experiment", help="preset: plows / observation / algorithm")
    p.add_argument("--map", choices=sorted(MAPS), help="road network (small = quick debugging)")
    p.add_argument("--plows", type=int, help="number of snowplows (agents)")
    p.add_argument("--obs", choices=["full", "local"], help="what a plow sees: the whole map or a window")
    p.add_argument("--view", type=int, choices=[3, 5], help="window size for --obs local")
    p.add_argument("--max-steps", type=int, help="episode length limit")
    p.add_argument(
        "--cloud-spawn-prob",
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
        "--exclusive-cells",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="forbid plows from sharing a cell (use --no-exclusive-cells to allow sharing)",
    )


def resolve_environment(args: argparse.Namespace) -> tuple[str, SnowplowConfig]:
    """Resolve the map and environment config used by a command."""
    experiment = get_experiment(args.experiment)
    env_grid_args: dict = experiment["env_grid_args"]
    snow_config_args: dict = experiment["snow_config_args"]
    snow = {}
    if args.cloud_spawn_prob is not None:
        snow["cloud_spawn_prob"] = args.cloud_spawn_prob
    if args.initial_clouds is not None:
        snow["initial_clouds"] = args.initial_clouds
    if getattr(args, "fixed_spawn_prob", False):
        snow["scale_spawn_with_plows"] = False

    snow["clear_per_pass"] = snow_config_args.get("clear_per_pass", 2)
    # Plow times are given for levels 1..max_snow on the command line; level 0 needs no work.
    if getattr(args, "plow_time_single", None):
        snow["plow_time_single"] = (0, *args.plow_time_single)
    if getattr(args, "plow_time_team", None):
        snow["plow_time_team"] = (0, *args.plow_time_team)

    config_defaults = SnowplowConfig()
    config = SnowplowConfig(
        n_plows=args.plows if args.plows is not None else env_grid_args.get("n_plows", config_defaults.n_plows),
        max_steps=args.max_steps if args.max_steps is not None else env_grid_args.get("max_steps", config_defaults.max_steps),
        observation=args.obs if args.obs is not None else env_grid_args.get("observation", config_defaults.observation),
        view_size=args.view if args.view is not None else env_grid_args.get("view_size", config_defaults.view_size),
        allow_shared_cells=not getattr(args, "exclusive_cells", False),
        snow=SnowConfig(**snow),
    )
    map_name = args.map if args.map is not None else env_grid_args.get("map", "network_3")
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


def write_json(path: Path, value) -> None:
    """Replace JSON atomically so readers never see a partially written history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False))
    temporary.replace(path)


def resolve_training(args, config):
    """Resolve training and PPO options with CLI > experiment > defaults."""
    from algorithms import PPOConfig

    preset = get_experiment(args.experiment)
    training = {"steps": 500_000, "checkpoint_every": None, **preset["training_args"]}
    ppo = {"reward_scale": 0.1, **preset["ppo_args"]}
    for key in ("n_envs", "rollout_len", "lr", "hidden", "recurrent", "reward_mode", "reward_scale", "seed"):
        value = getattr(args, key, None)
        if value is not None:
            ppo[key] = value
    for key in ("algo", "steps", "checkpoint_every"):
        value = getattr(args, key, None)
        if value is not None:
            training[key] = value
    training.setdefault("algo", "ppo" if config.n_plows == 1 else "mappo")
    ppo.setdefault("reward_mode", "individual" if training["algo"] == "ppo" else "team_mean")
    ppo.setdefault("recurrent", config.observation == "local")
    if training["steps"] <= 0:
        raise ValueError("--steps must be positive")
    if training["checkpoint_every"] is not None and training["checkpoint_every"] < 0:
        raise ValueError("--checkpoint-every must be >= 0 (0 disables checkpoints)")
    return training, PPOConfig(**ppo)


def cmd_train(args) -> None:
    from algorithms import MAPPO, PPO, plow_progress_lcr

    if args.resume:
        # Resume uses the checkpoint's environment and optimizer configuration.
        override_names = (
            "experiment",
            "map",
            "plows",
            "obs",
            "view",
            "max_steps",
            "spawn_prob",
            "initial_clouds",
            "plow_time_single",
            "plow_time_team",
            "exclusive_cells",
            "algo",
            "n_envs",
            "rollout_len",
            "lr",
            "hidden",
            "recurrent",
            "reward_mode",
            "reward_scale",
            "seed",
        )
        if any(getattr(args, name, None) is not None for name in override_names) or args.fixed_spawn_prob or args.lcr_zeta:
            raise ValueError("--resume restores saved settings; only --steps, --checkpoint-every, logging/evaluation and --out may change")
        trainer, map_name, config = load_trainer(Path(args.resume), resume=True)
        metadata = dict(trainer.run_config)
        training = {
            "steps": args.steps if args.steps is not None else metadata["steps"],
            "checkpoint_every": args.checkpoint_every if args.checkpoint_every is not None else metadata.get("checkpoint_every"),
        }
        if training["steps"] <= trainer.env_steps:
            raise ValueError("--steps must exceed the checkpoint's completed environment steps")
        if training["checkpoint_every"] is not None and training["checkpoint_every"] < 0:
            raise ValueError("--checkpoint-every must be >= 0")
        algo = metadata["algo"]
        source = Path(args.resume)
        run_dir = Path(args.out) if args.out else (source if source.is_dir() else source.parent)
        if run_dir.name == "checkpoints" and not args.out:
            run_dir = run_dir.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata["resumed_from"] = str(source.resolve())
        metadata["resume_mode"] = "optimizer and progress restored; episodes restart"
        metadata.update(training)
    else:
        map_name, config = resolve_environment(args)
        training, ppo_config = resolve_training(args, config)
        algo = training["algo"]
        if algo == "ppo" and config.n_plows != 1:
            raise ValueError("PPO requires one plow; select --algo mappo for multiple plows")
        reward_transform = plow_progress_lcr(args.lcr_zeta) if args.lcr_zeta > 0 else None
        trainer = {"ppo": PPO, "mappo": MAPPO}[algo](
            make_env_fn(map_name, config), ppo_config, adapter_kwargs(config), reward_transform=reward_transform
        )
        args.seed = ppo_config.seed
        run_dir = create_run_dir(args, map_name, algo)
        if (run_dir / "config.json").exists() or (run_dir / "model.pt").exists():
            raise ValueError("Run directory already contains training artifacts; use --resume or a new --out")
        metadata = {
            "map": map_name,
            "algo": algo,
            "snowplow": config_to_dict(config),
            "ppo": asdict(ppo_config),
            "steps": training["steps"],
            "checkpoint_every": training["checkpoint_every"],
            "lcr_zeta": args.lcr_zeta,
            "history_metrics": "rolling means over the last 20 completed training episodes; null before completion",
        }

    trainer.run_config = metadata
    write_json(run_dir / "config.json", metadata)
    write_json(run_dir / "history.json", trainer.history)
    checkpoint = make_checkpoint_callback(trainer, run_dir, training["checkpoint_every"])

    def on_update(stats):
        write_json(run_dir / "history.json", trainer.history)
        if checkpoint:
            checkpoint(stats)

    print(f"training {algo} on '{map_name}': {config.n_plows} plow(s), obs={config.observation} -> {run_dir}")
    trainer.learn(training["steps"], log_every=args.log_every, callback=on_update)
    trainer.save(str(run_dir / "model.pt"))
    results = summarize(run_trainer(trainer, args.eval_episodes, seed=trainer.cfg.seed + 10_000))
    write_results(run_dir, {"trained": results}, {"map": map_name, "algo": algo})
    print_table({"trained": results})
    print(f"saved model, history and metrics to {run_dir}/")


def make_checkpoint_callback(trainer, run_dir: Path, every: int | None):
    """Save at the first completed update crossing each environment-step interval."""
    if not every:
        return None
    if every < 0:
        raise ValueError("checkpoint interval must be positive or 0 to disable")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    next_save = (getattr(trainer, "env_steps", 0) // every + 1) * every

    def callback(stats: dict) -> None:
        nonlocal next_save
        steps = int(stats["env_steps"])
        if steps >= next_save:
            path = checkpoint_dir / f"model_step_{steps}.pt"
            trainer.save(str(path))
            print(f"checkpoint saved: {path}")
            next_save = (steps // every + 1) * every

    return callback


def checkpoint_path(run: Path, checkpoint: str | None = None) -> Path:
    if checkpoint:
        selected = Path(checkpoint)
        candidates = (
            (selected,)
            if selected.is_absolute()
            else (
                run / checkpoint,
                run / "checkpoints" / checkpoint,
                selected,
            )
        )
    else:
        candidates = (run / "model.pt",) if run.is_dir() else (run,)
    for path in candidates:
        if path.is_file():
            return path
    if not checkpoint and run.is_dir():
        checkpoints = [path for path in (run / "checkpoints").glob("model_step_*.pt") if path.stem.removeprefix("model_step_").isdigit()]
        if checkpoints:
            return max(checkpoints, key=lambda path: int(path.stem.removeprefix("model_step_")))
    raise ValueError(f"checkpoint not found: {checkpoint or run}")


def load_trainer(run_dir: Path, checkpoint: str | None = None, resume: bool = False):
    """Load a run, checkpoint name/path, or standalone self-contained checkpoint.

    Legacy weights use the accompanying config.json and remain inference-only.
    """
    import torch

    from algorithms import MAPPO, PPO, PPOConfig, plow_progress_lcr

    weights = checkpoint_path(run_dir, checkpoint)
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    saved = payload.get("run_config")
    if not saved:
        configs = (run_dir / "config.json", weights.parent / "config.json", weights.parent.parent / "config.json")
        config_path = next((path for path in configs if path.is_file()), None)
        if config_path is None:
            raise ValueError("Legacy weights require the run's config.json")
        saved = json.loads(config_path.read_text())
    config = config_from_dict(saved["snowplow"])
    ppo_config = PPOConfig(**saved["ppo"])
    if ppo_config.device.startswith("cuda") and not torch.cuda.is_available():
        ppo_config.device = "cpu"
    transform = plow_progress_lcr(saved.get("lcr_zeta", 0)) if resume and saved.get("lcr_zeta", 0) > 0 else None
    trainer = {"ppo": PPO, "mappo": MAPPO}[saved["algo"]](
        make_env_fn(saved["map"], config), ppo_config, adapter_kwargs(config), reward_transform=transform
    )
    trainer.load(str(weights), resume=resume)
    trainer.run_config = saved
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


def main(argv=None) -> None:
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
    p.add_argument("--steps", type=int, default=None, help="total environment steps")
    p.add_argument("--recurrent", action=argparse.BooleanOptionalAction, default=None, help="GRU policy (needed for --obs local)")
    p.add_argument("--reward-mode", choices=["individual", "team_sum", "team_mean"])
    p.add_argument("--lcr-zeta", type=float, default=0.0, help="MAPPO-LCR reward shaping strength (0 = off)")
    p.add_argument("--reward-scale", type=float, default=None, help="multiply training rewards by this (reporting stays raw)")
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--rollout-len", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        metavar="STEPS",
        help="also save the model every STEPS environment steps to <run>/checkpoints/",
    )
    p.add_argument("--resume", help="resume a run directory or checkpoint; --steps is the total target budget")
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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
