import enum


class MovementActions(enum.Enum):
    DO_NOTHING = (0, 0)
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)


class GridCellTypes(enum.Enum):
    OBSTACLE = 0
    ROAD = 1
    INTERSECTION = 2
    HIGHWAY = 3
