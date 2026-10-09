from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from mosaic_multigrid.base import AgentID


class EventType(Enum):
    MOVED = auto()
    INVALID_MOVE = auto()
    AGENT_COLLISION = auto()


@dataclass(frozen=True)
class Event:
    type: Enum
    agent_id: AgentID | None = None
    team_id: int | None = None
    step: int = 0
    data: dict[str, Any] = field(default_factory=dict)


class EventLog:
    def __init__(self) -> None:
        self._events: list[Event] = []

    def emit(
        self,
        type: Enum,
        agent_id: AgentID | None = None,
        team_id: int | None = None,
        step: int = 0,
        **data: Any,
    ) -> None:
        self._events.append(
            Event(type=type, agent_id=agent_id, team_id=team_id, step=step, data=data)
        )

    def events(self) -> list[Event]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()
