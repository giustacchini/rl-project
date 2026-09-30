# GitHub Issues – RL Environment Implementation

These issue descriptions are based on the agreed implementation plan.

## Team note

There are **four people** in the RL group, and **three of the four also work on the Ethics project**.

Because of that, it is useful to assign the three shared-infrastructure issues to people who also work on Ethics, so the same architecture can later be reused for the soccer environment.

A practical split could be:

| Role | Main issue(s) | Reusable for Ethics? |
|---|---|---|
| Person A | #1, #2 – architecture + base environment | Yes |
| Person B | #3, #5 – grid/movement + rendering/testing | Yes |
| Person C | #4 – observations/rewards/events | Yes |
| Person D | #6, #7 – snowplow roads + snow/cloud mechanics | Mostly RL-specific |

After the first phase, everyone can help with #8–#10.

## Suggested order

1. #1 Define shared interfaces
2. #2–#5 Shared backbone in parallel
3. #6–#8 Snowplow Experiment 1 implementation in parallel
4. #9 Validation
5. #10 Integration and feature freeze

## Issue files

- `01_define-shared-interfaces.md` — Define shared environment interfaces and repository structure
- `02_base-multi-agent-env.md` — Implement the base multi-agent environment lifecycle
- `03_grid-entities-movement.md` — Implement grid, entities, and simultaneous movement resolution
- `04_observations-rewards-events.md` — Implement shared observation, reward, and event interfaces
- `05_rendering-testing-backbone.md` — Add rendering and tests for the shared backbone
- `06_snowplow-road-network.md` — Implement snowplow road network and traffic rules
- `07_snow-cloud-mechanics.md` — Implement snow levels, plowing, and cloud events
- `08_experiment-1-rewards-termination-metrics.md` — Implement Experiment 1 rewards, termination, and metrics
- `09_validate-experiment-1.md` — Validate Experiment 1 with manual and random actions
- `10_experiment-1-integration.md` — Integrate and freeze Snowplow Experiment 1

## Suggested GitHub labels

- `architecture`
- `shared`
- `environment`
- `grid`
- `movement`
- `observations`
- `rewards`
- `testing`
- `rendering`
- `snowplow`
- `snow`
- `weather`
- `metrics`
- `integration`
- `experiment-1`
- `priority-high`

## Suggested milestones

### Shared backbone

Issues #1–#5.

### Experiment 1

Issues #6–#10.

## Important

The exact reward values and some environment parameters can still be tuned later. The first goal is to make the environment correct, reproducible, testable, and easy to extend.