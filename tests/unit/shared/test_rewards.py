from shared.events import Event, EventType
from shared.rewards import EventRewardCalculator
from enum import Enum
import pytest

def test_collision_penalty_applies_only_to_given_agent():
    calculator = EventRewardCalculator(
        reward_mapping={
            EventType.AGENT_COLLISION: -10.0,
        },
    )

    events = [
        Event(
            type=EventType.AGENT_COLLISION,
            agent_id=0,
            team_id=1,
        ),
    ]

    agent_teams = {
        0: 1,
        1: 1,
    }

    rewards = calculator.calculate_rewards(events, agent_teams)

    assert rewards == {
        0: -10.0,
        1: 0.0,
    }

def test_shared_reward_goes_only_to_matching_team():
    class DomainEventType(Enum):
        GOAL = "goal"

    calculator = EventRewardCalculator(
        reward_mapping={DomainEventType.GOAL: 10.0},
        team_shared_events={DomainEventType.GOAL},
    )

    events = [
        Event(
            type=DomainEventType.GOAL,
            agent_id=0,
            team_id=1,
        ),
    ]

    agent_teams = {
        0: 1,
        1: 1,
        2: 2,
    }

    rewards = calculator.calculate_rewards(events, agent_teams)

    assert rewards == {
        0: 10.0,
        1: 10.0,
        2: 0.0,
    }

def test_team_shared_event_requires_team_id():
    class DomainEventType(Enum):
        GOAL = "goal"

    calculator = EventRewardCalculator(
        reward_mapping={DomainEventType.GOAL: 10.0},
        team_shared_events={DomainEventType.GOAL},
    )

    events = [
        Event(type=DomainEventType.GOAL, agent_id=0),
    ]

    with pytest.raises(ValueError):
        calculator.calculate_rewards(events, {0: 1})


def test_rewards_from_multiple_events_are_added():
    calculator = EventRewardCalculator(
        reward_mapping={
            EventType.MOVED: -1.0,
            EventType.AGENT_COLLISION: -10.0,
        },
    )

    events = [
        Event(type=EventType.MOVED, agent_id=0),
        Event(type=EventType.AGENT_COLLISION, agent_id=0),
    ]

    rewards = calculator.calculate_rewards(events, {0: None})

    assert rewards == {0: -11.0}

def test_no_events_returns_zero_for_all_agents():
    calculator = EventRewardCalculator(
        reward_mapping={
            EventType.AGENT_COLLISION: -10.0,
        },
    )

    rewards = calculator.calculate_rewards(
        events=[],
        agent_teams={0: None, 1: None},
    )

    assert rewards == {
        0: 0.0,
        1: 0.0,
    }

def test_event_without_reward_mapping_gives_zero():
    calculator = EventRewardCalculator(
        reward_mapping={
            EventType.AGENT_COLLISION: -10.0,
        },
    )

    events = [
        Event(type=EventType.MOVED, agent_id=0),
    ]

    rewards = calculator.calculate_rewards(events, {0: None})

    assert rewards == {0: 0.0}