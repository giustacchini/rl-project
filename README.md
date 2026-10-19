# Multi-agent snowplow reinforcement learning

A gridworld where PPO controls one plow and MAPPO controls multiple plows.
Players act simultaneously, follow road directions, and clear snow through
configurable time-based work. Full-map and local-window observations support
four experiment presets. Shared movement, observation, event, reward, and metric
utilities are separate from snowplow rules.

## 1. Install and check the project

The verified environment is **Linux, Python 3.12, CPU PyTorch**. Run commands
from the repository root. Direct dependencies are pinned in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check

# A consistent CPU thread count also makes short runs faster on many-core machines.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m pytest -q
```

PyTorch is needed for training and saved-model playback/evaluation. Pillow
supports GIF/PNG export; Matplotlib supports plots. The requirements include
pytest and coverage so the complete suite, including training smoke tests, runs
after installation. Tests may otherwise skip training when PyTorch is missing.
A window requires a graphical desktop; evaluation, training, plots, and GIF
export work without one. On a headless machine do not pass `--render`.

## 2. Run the bundled trained models

No training is required to use these self-contained checkpoints:

| File | Algorithm | Plows | Observation | Map | Training seed | Saved steps |
|---|---|---:|---|---|---:|---:|
| `models/exp1/model.pt` | PPO | 1 | Full map | `network_1` | 0 | 500,224 |
| `models/exp2/model.pt` | Recurrent PPO | 1 | Local 5×5 | `network_1` | 0 | 500,224 |

Both checkpoints contain model/optimizer state, environment and PPO configuration,
update counters, and training history. Both have 977 updates, a 1,000-step episode
horizon, and a requested training budget of 500,000 steps. Their rollout size is
512 transitions, explaining the final 500,224 steps. These files have not been
retrained or rewritten by the code review.

```bash
# Watch either trained policy. --policy model is essential; the default is greedy.
python src/main.py play --policy model --run models/exp1 --render --seed 31
python src/main.py play --policy model --run models/exp2 --render --seed 31

# Evaluate headlessly, keeping generated results outside the models folder.
python src/main.py evaluate --run models/exp1 --episodes 50 --seed 1000 --out results/exp1_eval
python src/main.py evaluate --run models/exp2/model.pt --episodes 50 --seed 1000 --out results/exp2_eval

# Export without opening a window; parent directories are created automatically.
python src/main.py play --policy model --run models/exp1 --seed 31 --gif results/exp1_seed31.gif
```

Evaluation writes `evaluation.json` and `evaluation.csv`. Model mode restores the
checkpoint's map, plow count, observation mode, and horizon; options such as
`--plows` do not override that saved configuration. `--deterministic` selects
argmax actions instead of sampling; report which mode you use because the policy
can behave differently. The seed controls environment randomness and policy
sampling. It is not a claim that repeated evaluation seeds are independent
training runs.

Checkpoint configuration does not embed the Python rule implementation or its
Git commit. The bundled models' snow configuration has scaling enabled and
shared cells enabled, whereas current exp1/exp2 presets disable both; with one
plow those two differences do not affect behavior. To reproduce an original
reported result, retain its source commit, checkpoint, dependency environment,
seeds, and evaluation settings. Loading a checkpoint into modified rules does
not reconstruct those original rules.

Optional identity check on Linux:

```bash
sha256sum models/exp1/model.pt models/exp2/model.pt
```

Expected SHA-256 values:

```text
71e783efc37c862987d9898e71f17078981d52e6944636bc5cf8ee219dfdba30  models/exp1/model.pt
6b0e0729bec9b33d50773a493920e5cdabd808d83fc3b7ce286acc589f4e8a36  models/exp2/model.pt
```

## 3. Run baselines and train experiments

| Preset | Plows | Observation | Trainer | Map |
|---|---:|---|---|---|
| `exp1` | 1 | Full map | PPO | `network_1` |
| `exp2` | 1 | Local 5×5 | Recurrent PPO | `network_1` |
| `exp3` | 5 | Full map | MAPPO | `network_3` |
| `exp4` | 5 | Local 5×5 | Recurrent MAPPO | `network_3` |

All presets currently request 500,000 training transitions, 1,000 transitions
per episode, and a checkpoint every 100,000 transitions. They use a fixed 0.02
cloud-spawn probability per transition. Explicit CLI options override presets;
remaining values come from the configuration dataclasses. Without a preset,
the default is one plow, a 500-step horizon, and `network_3`.

```bash
# Quick random/greedy baseline run, with JSON/CSV output.
python src/main.py baselines --experiment exp1 --episodes 2 --max-steps 30 --seed 1000 --out results/baselines

# Non-learning policy demonstrations.
python src/main.py play --experiment exp1 --policy greedy --render --seed 31
python src/main.py play --experiment exp3 --policy greedy_coop --render --seed 31
python src/main.py play --experiment exp2 --policy random --render --seed 31

# Small training smoke check before a full run.
python src/main.py train --experiment exp1 --steps 128 --max-steps 8 --n-envs 2 --rollout-len 8 --hidden 32 --checkpoint-every 64 --eval-episodes 2 --seed 0 --out runs/smoke_seed0

# Full runs. Use fresh output directories when repeating.
python src/main.py train --experiment exp1 --seed 0 --out runs/exp1_seed0
python src/main.py train --experiment exp2 --seed 0 --out runs/exp2_seed0
python src/main.py train --experiment exp3 --seed 0 --out runs/exp3_seed0
python src/main.py train --experiment exp4 --seed 0 --out runs/exp4_seed0
```

Greedy policies need the full map; local-observation baselines use random actions.
Change `--map`, `--plows`, `--obs`, `--view`, or `--max-steps` explicitly when
needed. PPO requires one plow; use MAPPO for multiple plows. Available maps are
`network_1`, `network_2`, and `network_3`; there is no `small` map alias.
Use `python src/main.py <command> --help` for all options.

## 4. Outputs, checkpoints, and reward comparisons

Each training run saves:

| Artifact | Contents |
|---|---|
| `config.json` | Resolved map, environment/PPO settings, seed, step budget, checkpoint interval |
| `history.json` | Algorithm losses and rolling environment metrics, saved after every update |
| `checkpoints/model_step_<steps>.pt` | Periodic weights, optimizer, counters, configuration, history, trainer RNG |
| `model.pt` | Final checkpoint |
| `metrics.json` / `metrics.csv` | Held-out end-of-training evaluation |

History has one row per update, not per episode. `episode_metrics` includes raw
reward, snow cleared/fallen/remaining, clean fraction, repeated work, street
completion, plowing steps, and cooperation. Means use the latest 20 completed
training episodes. Before the first completion, measurements are `null` rather
than zero. Training reward scaling/shaping does not change reported raw rewards.

```bash
# Evaluate or display an intermediate checkpoint; use its actual filename.
python src/main.py evaluate --run runs/exp1_seed0 --checkpoint model_step_100352.pt --episodes 50 --seed 1000 --out results/exp1_checkpoint
python src/main.py play --policy model --run runs/exp1_seed0 --checkpoint model_step_100352.pt --render

# Resume to a TOTAL budget, starting fresh episodes with restored optimizer/history.
python src/main.py train --resume runs/exp1_seed0 --steps 1000000 --eval-episodes 10

# Branch from a bundled model without modifying its directory.
python src/main.py train --resume models/exp1/model.pt --steps 1000000 --out runs/exp1_continued --eval-episodes 10

# Plot training reward, or overlay histories from different runs.
python src/evaluate.py runs/exp1_seed0 --out results/exp1_reward.png
python src/evaluate.py runs/exp1_seed0 runs/exp2_seed0 --labels "Full observation" "Local observation" --x-axis env_steps --out results/reward_comparison.svg
python src/evaluate.py runs/exp1_seed0 runs/exp3_seed0 --metric snow_cleared --x-axis env_steps --out results/snow_comparison.png
```

Periodic saves occur after complete updates; filenames use actual steps and may
exceed an interval. One update is `n_envs × rollout_len` environment transitions,
not multiplied by the number of plows. `--steps` rounds up to a full rollout.
`--checkpoint-every 0` disables intermediate saves. An interrupted run without
`model.pt` loads its latest numeric periodic checkpoint.

Resume restores optimization and progress, but does not restore mid-episode
simulator snapshots or recurrent rollout states. It starts fresh episodes;
bit-for-bit continuation is not guaranteed. Resuming an older checkpoint into
the same directory replaces later history; use `--out` to preserve that run.

The model files contain embedded histories but no separate history JSON files.
To plot one without modifying `models/`, export its history first:

```bash
mkdir -p results
python -c "import json, torch; from pathlib import Path; c=torch.load('models/exp1/model.pt', map_location='cpu', weights_only=True); Path('results/exp1_history.json').write_text(json.dumps(c['history'], indent=1, allow_nan=False))"
python src/evaluate.py results/exp1_history.json --out results/bundled_exp1_reward.png
```

`src/evaluate.py` plots recorded training metrics; `src/main.py evaluate` runs
new episodes. Curves are not confidence intervals. Compare equal environment
step budgets and consistent maps, snowfall, episode horizons and evaluation
seeds; exp3/exp4 change both map size and agent count by default. Compare several
independent training seeds before drawing performance conclusions.

## 5. Current mechanics and source organization

- Actions: `DO_NOTHING, UP, DOWN, RIGHT, LEFT`; legal-action masks reject terrain,
  lane violations, and movement while a plow is busy.
- Snow levels: 0–3. Single-plow cleaning takes 1/5/10 transitions for levels 1/2/3;
  a team sharing a cell takes 1/2/3. The arrival transition counts as work.
- Only cleaning a cell earns reward: +1/+2/+3 by snow level, apportioned according
  to each contributor's work. Movement, waiting, plowing itself, street completion,
  collisions, and clean-map status have no additional reward.
- Episodes end at the configured time limit, even if the map becomes clean.
- Exp3/exp4 allow shared cells for cooperative plowing; exp1/exp2 disable sharing.
  Override with `--exclusive-cells` or `--no-exclusive-cells`.
- Presets keep snowfall probability fixed. `--cloud-spawn-prob` (alias
  `--spawn-prob`) sets its value; `--no-fixed-spawn-prob` enables scaling by plow
  count, capped at 1. Without a preset, `SnowConfig` scales by default.
- Full observations expose terrain, snow, permitted directions, plow positions,
  and work progress; local views are 3×3 or 5×5. Cloud locations are hidden from
  policies; their dark renderer overlay does not mean snow disappeared.
- Map rectangles use `(row, column)` and `(row count, column count)`. Arrays use
  `[row, column]`; agent/grid coordinates use `(x, y) = (column, row)`. Map gaps
  become obstacles; later terrain layers replace earlier cells and street
  metrics use the final layout. Connectivity checking is an explicit diagnostic.

The existing structure is:

```text
src/main.py                       Train/evaluate/play/baseline CLI
src/evaluate.py                   History plots and experiment comparisons
src/Enums.py                      Action ordering and event/terrain categories
src/shared/                       Movement, grid checks, events, rewards, observations, metrics
src/snowplow/env.py                Environment lifecycle and domain interactions
src/snowplow/config.py             Snow and environment dataclasses
src/snowplow/experiments.py        Four experiment presets
src/snowplow/road_network_configs.py Map definitions and registry
src/snowplow/road_map.py           Terrain expansion, lane rules, street membership
src/snowplow/snow.py               Snowfall and cooperative work progress
src/snowplow/cell.py               Tile rendering
src/snowplow/encoding.py           Learner observation/global-state vectors
src/snowplow/rewards.py            Current cleaning rewards
src/snowplow/factory.py            Consistent environment construction
src/snowplow/baselines.py          Random, greedy, and cooperative greedy policies
src/snowplow/evaluation.py         Seeded episode evaluation and aggregation
src/algorithms/                   Adapter, PPO/MAPPO, reward shaping, MADDPG
models/exp1/, models/exp2/         Bundled self-contained checkpoints
```

The environment uses a Parallel-style API through the project adapter; strict
PettingZoo compatibility is not claimed. MADDPG is present but is not connected
to the snowplow CLI. Training smoke tests verify operation, not convergence or
that a trained agent outperforms a baseline.

## 6. Reproducibility and Git workflow

Keep the same Python/dependency versions, source commit, explicit seed, map,
resolved config, checkpoint and evaluation mode. CPU runs with identical settings
are tested for reproducibility; results across hardware/library changes need
not be bit-identical. A resumed run is distinct from uninterrupted training.

Record versions and the current source revision with each reported experiment:

```bash
mkdir -p results
python --version
python -m pip freeze > results/environment.txt
git rev-parse HEAD > results/source_commit.txt
git status --short
```

A commit ID describes committed code only. Commit reviewed changes before final
experiments, and record any local differences if running a dirty checkout.
`runs/` and `results/` are ignored generated-output directories. The supplied
models are tracked artifacts; keep their metadata with any replacement models.

```bash
# Follow the team's feature-branch and review process.
git switch -c yourinitials/describe-change
git diff --check
python -m pytest -q
git diff
git add <changed-files>
git commit -m "fix(training): describe the verified change"
```

Open a pull request to `dev` and request teammate review as described in
[the contribution process](docs/PROCESS.md). See also [testing](docs/TESTING.md),
[training details](docs/TRAINING.md), and [the course assignment](docs/ASSIGNMENT.md).
