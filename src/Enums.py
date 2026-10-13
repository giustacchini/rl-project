import enum

import numpy as np


class MovementActions(enum.Enum):
    DO_NOTHING = (0, 0)
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)


class GridCellTypes(enum.Enum):
    OBSTACLE = "obstacles"
    ROAD = "roads"
    INTERSECTION = "intersections"
    HIGHWAY = "highways"
