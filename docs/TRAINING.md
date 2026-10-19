# Training, checkpoints, and reward plots

Run these commands from the repository root in the project virtual environment.
Install `requirements.txt`, including PyTorch for training and Matplotlib for plots.

## Configuration resolution

The four presets in `src/snowplow/experiments.py` use the same schema:

- `env_args`: map, plow count, observation mode, view size, episode length, shared cells.
- `snow_args`: `SnowConfig` overrides; omitted values use this branch's defaults.
- `ppo_args`: `PPOConfig` overrides.
- `training_args`: algorithm, total step target, checkpoint interval.

An explicit CLI value overrides the preset, then dataclass defaults apply.
Zero-valued snowfall options and `--checkpoint-every 0` are actual overrides.
Boolean settings can be reversed with `--no-recurrent` or `--no-exclusive-cells`.
Without a preset, the default map is `network_3`, with one plow and PPO.

The presets use fixed cloud-spawn probability by default. `--no-fixed-spawn-prob`
enables scaling; `--fixed-spawn-prob` disables it. `--cloud-spawn-prob` and its
legacy alias `--spawn-prob` set the probability.

The presets retain one-plow `network_1` runs for exp1/exp2 and five-plow
`network_3` runs for exp3/exp4. Exp2/exp4 are recurrent with local observations.
Their default training target is 500,000 environment steps and checkpoint
interval is 100,000 steps. Override these for quick checks:

```bash
python src/main.py train --experiment exp1 --steps 4096 --max-steps 16 --n-envs 2 --rollout-len 16 --hidden 32 --checkpoint-every 1024 --eval-episodes 2 --out runs/training_check
```

An explicit `--out` directory containing config/model artifacts is not reused
for a new training run. Use a fresh directory or `--resume`.

## History is saved during training

`history.json` is a JSON array with one row per **optimizer update**, not one row
per episode. It is atomically replaced after each completed update, before the
final evaluation, so it can be read while training is running.

Each row includes the algorithm losses plus:

| Field | Meaning |
|---|---|
| `update` | Cumulative update number, preserved on resume |
| `env_steps` | Total transitions summed over parallel environments, not agents |
| `episodes_completed` | Total completed training episodes |
| `metrics_episodes` | Number of episodes in the rolling metric window, up to 20 |
| `mean_return` | Rolling mean raw team return, before reward shaping/scaling |
| `mean_ep_len` | Rolling mean episode length |
| `total_reward` | Rolling mean environment episode reward |
| `episode_metrics` | Rolling means of the numeric fields emitted by the environment |

`episode_metrics` includes `snow_cleared`, `snow_fallen`, `snow_remaining`,
`fraction_clean`, `repeated_work`, movement/collision counts, street completion,
`cells_cleared`, `cells_cleared_together`, `cooperation_rate`, `plowing_steps`,
and `shared_cell_steps`. Metrics follow the current environment summary rather
than inserting removed fields such as success into fixed-length episodes.

Before an episode completes, reward/length measurements are `null` and
`episode_metrics` is empty. If no new episode completes during an update, the
rolling window retains its prior measurements. These are training statistics;
the end-of-run `metrics.json`/CSV remains a separate held-out evaluation.

## Checkpoints and resume

Periodic checkpoints are saved as `checkpoints/model_step_<actual_steps>.pt`.
Intervals count environment transitions across parallel environments. A save
happens at the first completed update that reaches or passes an interval, so
the filename may exceed the requested multiple. If one update crosses several
intervals, it saves one model for that update, not duplicate copies. `0` disables
periodic checkpointing; the final `model.pt` is still saved.

New checkpoints contain weights, optimizer state, configuration, counters,
training history, recent episode summaries, and trainer RNG states. CLI-generated
checkpoints also contain map/environment configuration, so they can be loaded
without an accompanying `config.json`.

```bash
# Evaluate or watch an intermediate model.
python src/main.py evaluate --run runs/training_check --checkpoint model_step_1024.pt --episodes 5
python src/main.py play --run runs/training_check --policy model --checkpoint model_step_1024.pt --render

# Continue optimization to a TOTAL target of 8192 steps (not 8192 extra steps).
python src/main.py train --resume runs/training_check --steps 8192 --eval-episodes 2

# Branch from an older checkpoint into a separate run directory.
python src/main.py train --resume runs/training_check/checkpoints/model_step_1024.pt --steps 8192 --out runs/checkpoint_branch --eval-episodes 2
```

`--run` also accepts a standalone checkpoint file. A run directory loads its
final `model.pt` if present; otherwise it selects the largest numeric periodic
checkpoint, which allows recovery after interrupted training.

Resume restores optimization, configuration, counters, and checkpoint history,
but **starts fresh episodes**. Environment snapshots and recurrent rollout
states are not restored; it is not a bit-for-bit continuation midway through
an episode. The total budget rounds up to a complete rollout. Resuming an older
checkpoint into the same directory replaces history/model artifacts from later
updates; use `--out` to preserve that later run. Environment/network overrides
are rejected on resume to prevent architecture mismatches.

Legacy weights-only checkpoints still support evaluation/play with their
original `config.json`, provided the architecture matches. They cannot resume
optimization because they contain no optimizer/progress state.

## Plot updates against total reward

```bash
python src/evaluate.py runs/training_check
python src/evaluate.py runs/training_check/history.json --out results/reward.svg
python src/evaluate.py runs/training_check --metric snow_cleared --out results/snow_cleared.png
```

The default output is `<run>/total_reward.png`: training update on the x-axis
and mean raw episode reward on the y-axis. It reads the history saved during
training, supports older `mean_return` histories, and leaves missing observations
as gaps. At least one completed episode is needed. `--show` additionally opens
the figure; otherwise plotting works headlessly. Output directories are created.
`src/evaluate.py` plots history; `src/main.py evaluate` actually runs episodes.

### Compare experiments

Pass multiple run directories or history files to overlay their curves:

```bash
python src/evaluate.py runs/exp1/history.json runs/exp3/history.json --labels "1 plow" "5 plows" --out results/comparison.png
python src/evaluate.py runs/exp1 runs/exp2 runs/exp3 runs/exp4 --out results/all_experiments.svg
python src/evaluate.py runs/exp1 runs/exp3 --metric snow_cleared --x-axis env_steps --out results/snow_comparison.png
```

Legend labels default to run-directory names (or filename stems for other JSON
filenames). `--labels` supplies one label per input, in the same order. Each curve
keeps its own length and x positions; missing rewards remain gaps. Multiple inputs
default to `comparison_total_reward.png` in the working directory; single-input
output defaults are unchanged.

Updates can represent different amounts of experience when rollout lengths or
parallel-environment counts differ. Use `--x-axis env_steps` to compare by training
budget; every input must contain `env_steps`. Curves show rolling raw episode
metrics rather than repeated-seed confidence intervals. Team reward magnitudes
and snowfall settings may also differ across experiments; account for those
settings when interpreting performance.
