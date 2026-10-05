from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from mosaic_multigrid.base import AgentID

class EventType(Enum):
    MOVED = "moved"
    INVALID_MOVE = "invalid_move"
    AGENT_COLLISION = "agent_collision"


@dataclass
class Event:
    type: Enum
    agent_id: AgentID | None = None
    team_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)