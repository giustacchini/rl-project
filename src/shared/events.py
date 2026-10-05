from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class EventType(Enum):
    MOVED = "moved"
    INVALID_MOVE = "invalid_move"
    AGENT_COLLISION = "agent_collision"


@dataclass
class Event:
    type: Enum
    agent_id: int | None = None
    team_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)