from Enums import EventType, SnowplowEvent
from shared.events import Event
from snowplow.rewards import make_snowplow_reward_calculator


def test_collision_and_movement_have_no_reward():
    calculator = make_snowplow_reward_calculator()
    events = [
        Event(type=EventType.MOVED, agent_id=0),
        Event(type=EventType.AGENT_COLLISION, agent_id=0),
    ]

    result = calculator.calculate_rewards(events, {0: None, 1: None})

    assert result.individual == {0: 0.0, 1: 0.0}
    assert result.shared_by_team == {}
    assert result.team_total(None) == 0.0


def test_cell_cleaning_is_individual_and_all_clean_has_no_bonus():
    calculator = make_snowplow_reward_calculator()
    events = [
        Event(type=SnowplowEvent.CELL_CLEARED, agent_id=1),
        Event(type=SnowplowEvent.ALL_CLEAN),
    ]

    result = calculator.calculate_rewards(events, {0: None, 1: None})

    assert result.individual == {0: 0.0, 1: 1.0}
    assert result.shared_by_team == {}
    assert result.team_total(None) == 1.0