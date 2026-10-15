# Refactor notes: where did the old code go?

The snowplow implementation lives in `src/snowplow/`. The old root-level
environment, traffic, and cell modules have been removed; callers must use the
package API described below. `src/main.py` is the only command-line entry point.

| Old | New | What changed |
|---|---|---|
| `src/snow_plow_env.py` | `src/snowplow/env.py` | real `reset()` / `step()` (12-stage lifecycle), rewards returned, metrics, action mask, seeded RNG |
| `src/traffic.py` (rules) + the same rules copied inside `SnowPlowEnv` | `src/snowplow/road_map.py` | ONE function `compute_allowed_moves`; result is a numpy table `allowed[y, x, action]` |
| `highway_/intersection_/road_traffic_rules` | same function | fixes: highway cell with no wall beside it returned `[None]`; intersection at (x=11, y=4) allowed `UP` into a wall |
| `src/traffic.py` (demo window) / `src/main.py` | `src/main.py play --render` | one CLI |
| `src/Traffic_configs.py` | `src/snowplow/maps.py` | + `small` debug map, registry `MAPS` |
| `src/grid_cell.py` | `src/snowplow/cell.py` | drawing only; snow now lives in `snowplow/snow.py` (numpy) |
| `let_it_snow_let_it_snow_let_it_snow` (ma) | `SnowField` in `snow.py` | seeded RNG (was global `np.random`), 3x3..9x9 irregular clouds, lifetime, snow only on roads |
| `get_observable_grid()` returned `grid.state` (no snow, no lane info) | `SnowPlowEnv.get_observable_grid()` | float `[9, H, W]`: snow, terrain one-hot, allowed moves |
| `handle_actions` returned only individual rewards (shared +100 / -10 were lost) | `RewardResult.per_agent()` | shared reward split equally between the team |
| `algorithms/train.py` (crashed: wrong arguments) | `src/main.py train` | |
| `algorithms/toy_envs.py`, `random_episode_runner.py`, `IPPO`, `shared/entities.py`, `tests/conftext.py` | deleted | unused |
| `algorithms/ppo.py` | same file | + action masks, `make_policy()`, `reward_scale`; IPPO removed |
| `algorithms/adapter.py` | same file | + action masks, `positions` info for LCR |
| tests for old API | `tests/unit/...` | rewritten; golden table of the traffic rules in `tests/golden/` |

## Migrating callers and tests

- Import `SnowPlowEnv` from `snowplow.env` and maps from `snowplow.maps`.
- Pass `SnowplowConfig(n_plows=..., max_steps=...)` as `config` instead of passing
  `agents`, `width`, `height`, or `max_steps` directly to the environment.
- Map dimensions come from the map's `size`. Invalid rectangles are rejected by
  `RoadMap.from_config`; the environment no longer takes independent dimensions.
- Call `reset()` and `step()`; direct calls to the old `handle_actions()` skipped
  the snow, cloud, reward, termination, observation, and metric lifecycle.
- Read simulation terrain from `env.road_map.codes` and snow from
  `env.snow_field.snow`. Render cells use `terrain`, not the old `cell_type` field.
- Traffic-grid regression tests now live in `tests/unit/snowplow/test_traffic_grid.py`.
  They check the original CSV layout, walkability, rendering, seeded distinct
  spawning, terrain preservation, and rejection of a map whose rectangles do
  not fit its dimensions.
