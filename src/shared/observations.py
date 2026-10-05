from abc import ABC, abstractmethod
from typing import Any

from mosaic_multigrid.base import AgentID
Observation = dict[str, Any]


class ObservationProvider(ABC):
    @abstractmethod
    def get_observations(self, env) -> dict[AgentID, Observation]:
        """Generate observations for all active agents."""
        pass

class FullObservationProvider(ObservationProvider):
    def get_observations(self, env) -> dict[AgentID, Observation]:
        observations = {}

        active_agents = [
            agent
            for agent in env.agents
            if not agent.state.terminated
        ]

        agent_positions = {
            agent.index: tuple(agent.state.pos)
            for agent in active_agents
        }

        for agent in active_agents:
            observations[agent.index] = {
                "grid": env.grid.state.copy(),
                "agent_positions": agent_positions.copy(),
                "self_position": tuple(agent.state.pos),
            }

        return observations

class LocalObservationProvider(ObservationProvider):
    def __init__(self, view_size: int):
        if view_size not in (3, 5):
            raise ValueError("view_size must be 3 or 5")

        self.view_size = view_size

    def get_observations(self, env) -> dict[AgentID, Observation]:
        raise NotImplementedError(
            "Local observations will be implemented later."
        )

