from abc import ABC, abstractmethod
from mosaic_multigrid.base import AgentID
from shared.events import Event
from enum import Enum


class RewardStrategy(ABC):
    @abstractmethod
    def calculate_rewards(
        self,
        events: list[Event],
        agent_teams: dict[AgentID, int | None],
    ) -> dict[AgentID, float]:
        """Calculate rewards for the agents in agent_teams."""
        pass

class EventRewardCalculator(RewardStrategy):
    def __init__(
        self,
        reward_mapping: dict[Enum, float],
        team_shared_events: set[Enum] | None = None,
    ):
        self.reward_mapping = reward_mapping.copy()
        self.team_shared_events = (
            set(team_shared_events)
            if team_shared_events is not None
            else set()
        )

    def calculate_rewards(
        self,
        events: list[Event],
        agent_teams: dict[AgentID, int | None],
    ) -> dict[AgentID, float]:
        rewards = {
            agent_id: 0.0
            for agent_id in agent_teams
        }

        for event in events:
            reward = self.reward_mapping.get(event.type, 0.0)

            if event.type in self.team_shared_events:
                if event.team_id is None:
                    raise ValueError(
                        "Team-shared events must have a team_id"
                    )
                
                for agent_id, team_id in agent_teams.items():
                    if team_id == event.team_id:
                        rewards[agent_id] += reward

            elif event.agent_id in rewards:
                rewards[event.agent_id] += reward

        return rewards
