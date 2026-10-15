"""Observation providers for all environments.

A provider only READS the environment and never changes it. It is separate from
the environment's step logic, so the view (full or local) can be changed without
rewriting the environment.

An environment must offer:
    env.get_observable_grid()      -> float array [C, H, W] with everything agents may see
                                      (the environment decides what is visible, e.g. the
                                      snowplow env hides clouds)
    env.get_action_mask(agent_id)  -> bool array [n_actions]  (optional; True = legal action)

Every observation is a dict:
    grid            [C, H, W] (full) or [C, k, k] window centred on the agent (local;
                    cells outside the map are all zeros)
    agent_positions {agent_id: (x, y)} of the agents the observer can see
    self_position   (x, y) of the observed agent
    view_origin     (x, y) map coordinates of the grid's top-left cell: (0, 0) for the full
                    view, (x - r, y - r) for a local window. Lets encoders place positions.
    action_mask     bool [n_actions]                       (only if the env provides it)
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from mosaic_multigrid.base import AgentID

Observation = dict[str, Any]


def _active_agents(env) -> list:
    """Terminated agents get no observation."""
    return [agent for agent in env.agents if not agent.state.terminated]


def _action_mask(env, agent_id: AgentID):
    """The env's action mask for this agent, or None if the env has no such concept."""
    getter = getattr(env, "get_action_mask", None)
    return None if getter is None else np.asarray(getter(agent_id), dtype=bool)


class ObservationProvider(ABC):
    @abstractmethod
    def get_observations(self, env) -> dict[AgentID, Observation]:
        """Generate observations for all active agents."""


class FullObservationProvider(ObservationProvider):
    """Every active agent sees the whole observable grid and all agent positions."""

    def get_observations(self, env) -> dict[AgentID, Observation]:
        active_agents = _active_agents(env)
        agent_positions = {agent.index: tuple(int(v) for v in agent.state.pos) for agent in active_agents}
        grid = env.get_observable_grid()

        observations = {}
        for agent in active_agents:
            obs = {
                "grid": grid.copy(),  # one copy per agent: callers may modify it safely
                "agent_positions": agent_positions.copy(),
                "self_position": agent_positions[agent.index],
                "view_origin": (0, 0),
            }
            mask = _action_mask(env, agent.index)
            if mask is not None:
                obs["action_mask"] = mask
            observations[agent.index] = obs
        return observations


class LocalObservationProvider(ObservationProvider):
    """Partial observability: each agent only sees a `view_size` x `view_size` window around itself.

    Outside the map the window is filled with zeros (all channels), which is how the
    agent can tell "off the map" apart from every real cell type. Only the other agents
    that stand inside the window are reported.
    """

    def __init__(self, view_size: int):
        if view_size not in (3, 5):
            raise ValueError("view_size must be 3 or 5")
        self.view_size = view_size

    def _window(self, grid: np.ndarray, x: int, y: int) -> np.ndarray:
        """Crop [C, k, k] around (x, y), zero-padded where the window leaves the map."""
        k, r = self.view_size, self.view_size // 2
        channels, height, width = grid.shape
        window = np.zeros((channels, k, k), dtype=grid.dtype)
        # Part of the window that overlaps the map, in map coordinates ...
        y0, y1 = max(y - r, 0), min(y + r + 1, height)
        x0, x1 = max(x - r, 0), min(x + r + 1, width)
        # ... and the same part in window coordinates.
        window[:, y0 - (y - r) : y1 - (y - r), x0 - (x - r) : x1 - (x - r)] = grid[:, y0:y1, x0:x1]
        return window

    def get_observations(self, env) -> dict[AgentID, Observation]:
        active_agents = _active_agents(env)
        positions = {agent.index: tuple(int(v) for v in agent.state.pos) for agent in active_agents}
        grid = env.get_observable_grid()
        r = self.view_size // 2

        observations = {}
        for agent in active_agents:
            x, y = positions[agent.index]
            visible = {
                other_id: pos
                for other_id, pos in positions.items()
                if abs(pos[0] - x) <= r and abs(pos[1] - y) <= r
            }
            obs = {
                "grid": self._window(grid, x, y),
                "agent_positions": visible,
                "self_position": (x, y),
                "view_origin": (x - r, y - r),
            }
            mask = _action_mask(env, agent.index)
            if mask is not None:
                obs["action_mask"] = mask
            observations[agent.index] = obs
        return observations
