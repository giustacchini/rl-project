"""Generic per-episode metrics: counts what happened, independent of the domain.

The environment calls `record_step` once per step with the rewards and the events
of that step; domain code (snowplow, soccer) can add its own numbers with `add`.
`summary()` returns a flat dict that can be logged to JSON/CSV.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from shared.events import Event


class EpisodeMetrics:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.reward_per_agent: dict[Any, float] = defaultdict(float)
        self.event_counts: dict[str, int] = defaultdict(int)
        self.extra: dict[str, float] = defaultdict(float)

    def record_step(self, rewards: dict, events: list[Event]) -> None:
        """Accumulate one step: rewards {agent_id: float} and the events it produced."""
        self.steps += 1
        for agent_id, reward in rewards.items():
            self.reward_per_agent[agent_id] += float(reward)
        for event in events:
            self.event_counts[event.type.name.lower()] += 1

    def add(self, name: str, amount: float = 1.0) -> None:
        """Domain-specific counter, e.g. add('snow_cleared', 2)."""
        self.extra[name] += amount

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "steps": self.steps,
            "total_reward": float(sum(self.reward_per_agent.values())),
        }
        out.update({f"events/{name}": count for name, count in sorted(self.event_counts.items())})
        out.update(dict(self.extra))
        return out
