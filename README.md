# Multi-agent snowplow environment

A seeded gridworld where snowplows clear roads, obey lane directions, and act
simultaneously. The shared movement, observations, events, rewards, and metrics
can also support other gridworld environments.

## Setup

Use Python 3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements include CPU PyTorch for training. Environment simulation,
random/greedy baselines, and rendering do not require PyTorch; training tests
are skipped when it is absent.

## Run

All commands below run from the repository root using the activated environment.

```bash
# Watch Experiment 1: one plow with full observability.
python src/main.py play --experiment exp1 --policy greedy --render

# Watch two plows that help each other on heavy snow.
python src/main.py play --experiment exp3 --policy greedy_coop --render

# Run random and greedy baselines and save JSON/CSV metrics.
python src/main.py baselines --experiment exp1 --episodes 50 --seed 1000 --out results/exp1

# Quick headless check on a map.
python src/main.py baselines --map network_1 --episodes 2 --max-steps 30

# Export an episode as a GIF without opening a window.
python src/main.py play --map network_1 --policy random --max-steps 30 --gif results/random.gif

# Train and evaluate a model (requires PyTorch).
python src/main.py train --experiment exp1 --steps 500000 --seed 0 --out results/exp1_model --checkpoint-every 100000
python src/main.py evaluate --run results/exp1_model --episodes 100 --seed 1000
python src/main.py play --policy model --run results/exp1_model --render
```

Create the GIF's parent directory before exporting. Use `python src/main.py --help`
or a subcommand's `--help` for all options. Rendering a window requires a desktop;
GIF export and baseline evaluation can run without one.

| Preset | Plows | Observations | Trainer |
|---|---:|---|---|
| `exp1` | 1 | Full map | PPO |
| `exp2` | 1 | Local window | Recurrent PPO |
| `exp3` | 2 | Full map | MAPPO |
| `exp4` | 2 | Local window | Recurrent MAPPO |

`--map`, `--plows`, `--obs`, `--view`, and `--max-steps` override the environment
preset. Omitted values come from the preset and then `SnowplowConfig` defaults.
The resolved map and environment configuration are saved in a training run's
`config.json`; evaluation and model playback always load those saved values.
Maps are selected with `--map network_1` (10 rows, 19 columns), `--map network_2`,
or `--map network_3`.
An episode ends on successful cleaning after snow has fallen, or at `--max-steps`.

## Code structure

```text
src/
  main.py                  CLI: play, baselines, train, evaluate
  Enums.py                 Action ordering, terrain categories, event names
  shared/                  Movement, grid checks, observations, events, rewards, metrics
  snowplow/
    env.py                 The single SnowPlowEnv implementation and step lifecycle
    config.py              SnowplowConfig and SnowConfig
    maps.py                Map rectangles and map registry
    road_map.py            Terrain, street membership, and traffic rules
    snow.py                Snow levels, clearing, and cloud lifecycle
    cell.py                Mosaic rendering objects
    encoding.py            Observation/state vectors for algorithms
    factory.py             Environment construction shared by the CLI and trainers
    baselines.py            Random and greedy reference policies
    evaluation.py          Episode execution and metric aggregation
    experiments.py         Experiment presets
    rewards.py             Snowplow event reward values
  algorithms/              Environment adapter, PPO/MAPPO, reward shaping, MADDPG
```

Use `snowplow.env.SnowPlowEnv` and configure it with `SnowplowConfig`; for named
maps use `snowplow.factory.make_env`. The former root-level environment, traffic,
and cell modules have been retired. See [migration notes](docs/REFACTOR_NOTES.md).

## Conventions and behavior

Map rectangles use `(row, column)` positions and `(row count, column count)` sizes.
Simulation arrays use `[row, column]`; agent positions use `(x, y) = (column, row)`.

Actions are indexed in the order `DO_NOTHING, UP, DOWN, RIGHT, LEFT`. Policies
receive legal-action masks. Full observations include snow, terrain, lane rules,
and plow positions; cloud locations are hidden. Local observations use 3x3 or
5x5 windows. The API is Parallel-style and uses the project's algorithm adapter;
it is not a claim of strict PettingZoo API compatibility.

Default snow levels are 0–3. Movement costs -1; waiting and plowing cost 0.
Cleaning a cell pays its snow level: +1 / +2 / +3 for a level 1 / 2 / 3 cell
(`SNOW_LEVEL_REWARDS`). Street completion is still counted in the metrics but no
longer rewarded. Values and sharing rules are defined in `snowplow/rewards.py`
and `shared/rewards.py`.

New clouds appear with probability `cloud_spawn_prob` (default 0.02) **per plow**
each step: 0.02 with one plow, 0.10 with five. Use `--spawn-prob` to change the
per-plow value and `--fixed-spawn-prob` to stop the scaling.

Training can save intermediate models with `--checkpoint-every STEPS`; they go to
`<run>/checkpoints/model_step_<steps>.pt`. Load one with
`evaluate --run <run> --checkpoint model_step_100000.pt` (also works for
`play --policy model`).

### Cooperative plowing

Plows may share a cell, and plowing takes time. A plow standing on a snowy cell is
**busy**: its only legal action is `DO_NOTHING` until the cell is completely clean.
The time needed depends on the snow level and on how many plows work on the cell:

| Snow level | One plow | Two or more plows |
|---:|---:|---:|
| 1 | 1 step | 1 step |
| 2 | 4 steps | 2 steps |
| 3 | 8 steps | 3 steps |

The arrival step counts as the first working step. Work is stored as a fraction per
cell (`plow_progress`, also an observation channel), so a helper that arrives late
still speeds up the rest of the job, and snow falling during plowing slows it down.
The clearing reward of a cell (its layers) is split between the plows in proportion
to the work each did, so the team earns the same per cell alone or together:
cooperation only pays off by being faster.

Settings: `SnowConfig.plow_time_single`, `SnowConfig.plow_time_team` and
`SnowplowConfig.allow_shared_cells` (CLI: `--plow-time-single 1 4 8`,
`--plow-time-team 1 2 3`, `--exclusive-cells`). The `greedy_coop` baseline sends an
idle plow to help a busy teammate when it can arrive before the teammate finishes
alone. Episode metrics include `cells_cleared_together` and `cooperation_rate`
(fraction of cleaned cells that were cleaned by 2+ plows) to track whether trained
agents learn to cooperate. Changing the observation (new `plow_progress` channel)
means models trained before this change cannot be loaded.

## Tests and documentation

```bash
python -m pytest -q
```

The tests cover traffic layout, lane restrictions, simultaneous collisions,
seeded spawning and snowfall, observations, rewards, termination, metrics,
rendering, and algorithm adapters. Training smoke tests also run when PyTorch
is installed. See [testing guide](docs/TESTING.md),
[course assignment](docs/ASSIGNMENT.md), and [contribution process](docs/PROCESS.md).
