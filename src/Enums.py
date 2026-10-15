"""Enumerations shared by the whole project (snowplow now, soccer later)."""

import enum
from enum import Enum, auto


class MovementActions(enum.Enum):
    """The five things an agent can do in one step.

    The value is the (dx, dy) displacement on the grid. Note the grid
    convention used everywhere: x = column, y = row, and y grows DOWNWARDS,
    so UP is (0, -1).
    """

    DO_NOTHING = (0, 0)
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)


# The fixed ordering of actions. A policy outputs an integer `i` and the
# environment executes ACTIONS[i]; action masks use the same order, so
# mask[i] == True means "ACTIONS[i] is legal in the current cell".
# (Order: DO_NOTHING, UP, DOWN, RIGHT, LEFT.)
ACTIONS = tuple(MovementActions)


class GridCellTypes(enum.Enum):
    """Kinds of terrain. The value is the key used in the road-network dicts."""

    OBSTACLE = "obstacles"
    ROAD = "roads"  # secondary street (two-way)
    INTERSECTION = "intersections"  # where a secondary street crosses the highway
    HIGHWAY = "highways"  # main street (two opposite one-way lanes)


class EventType(Enum):
    """Generic events any environment can emit (movement bookkeeping)."""

    MOVED = auto()
    INVALID_MOVE = auto()
    AGENT_COLLISION = auto()
    DO_NOTHING = auto()


class SnowplowEvent(Enum):
    """Events specific to the snowplow task (used for rewards)."""

    PARTIALLY_CLEARED = auto()  # a plow removed snow from one cell
    SECONDARY_STREET_CLEARED = auto()  # a secondary street became snow-free
    MAIN_STREET_CLEARED = auto()  # a main street (highway) became snow-free
    ALL_CLEAN = auto()  # every road is snow-free: episode success
