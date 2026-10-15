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

# Run random and greedy baselines and save JSON/CSV metrics.
python src/main.py baselines --experiment exp1 --episodes 50 --seed 1000 --out results/exp1

# Quick headless check on the smaller map.
python src/main.py baselines --map small --episodes 2 --max-steps 30

# Export an episode as a GIF without opening a window.
python src/main.py play --map small --policy random --max-steps 30 --gif results/random.gif

# Train and evaluate a model (requires PyTorch).
python src/main.py train --experiment exp1 --steps 500000 --seed 0 --out results/exp1_model
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

`--plows`, `--obs`, and `--view` override the environment preset. Maps are selected
with `--map network_1` (10 rows, 19 columns) or `--map small` (8 rows, 9 columns).
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

Default snow levels are 0–3 and plows clear at most two layers per pass. Movement
costs -1, waiting -2, and invalid movement -5. Clearing a snowy cell gives +1;
completing a secondary/main street gives +2/+4. Collision events cost -10 and
completion gives +100 as shared team rewards. A street is one configured
rectangle, so splitting rectangles changes street-completion rewards. Values
and sharing rules are defined in `snowplow/rewards.py` and `shared/rewards.py`.

## Tests and documentation

```bash
python -m pytest -q
```

The tests cover traffic layout, lane restrictions, simultaneous collisions,
seeded spawning and snowfall, observations, rewards, termination, metrics,
rendering, and algorithm adapters. Training smoke tests also run when PyTorch
is installed. See [testing guide](docs/TESTING.md),
[course assignment](docs/ASSIGNMENT.md), and [contribution process](docs/PROCESS.md).
