from shared.events import Event, EventType


def test_event_stores_given_values():
    event = Event(
        type=EventType.MOVED,
        agent_id=0,
    )

    assert event.type == EventType.MOVED
    assert event.agent_id == 0

def test_event_uses_default_values():
    event = Event(type=EventType.MOVED)

    assert event.agent_id is None
    assert event.team_id is None
    assert event.data == {}

def test_event_stores_team_id():
    event = Event(
        type=EventType.MOVED,
        agent_id=0,
        team_id=1,
    )

    assert event.team_id == 1

def test_event_stores_extra_data():
    event = Event(
        type=EventType.AGENT_COLLISION,
        agent_id=0,
        data={"other_agent_id": 2},
    )

    assert event.data == {"other_agent_id": 2}

def test_each_event_has_its_own_data():
    first_event = Event(type=EventType.MOVED, agent_id=0)
    second_event = Event(type=EventType.MOVED, agent_id=1)

    first_event.data["position"] = (2, 3)

    assert second_event.data == {}
