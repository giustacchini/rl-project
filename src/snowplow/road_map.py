"""The static part of the snowplow world: terrain, traffic rules and streets.

Everything in here is computed ONCE from a map dict (see snowplow/maps.py) and
never changes during an episode. It is stored as plain numpy arrays indexed
[row, column] = [y, x]:

    codes[y, x]       terrain type (OBSTACLE / ROAD / HIGHWAY / INTERSECTION)
    allowed[y, x, a]  True if action ACTIONS[a] is legal when standing on (x, y)
    street_id[y, x]   which street the cell belongs to (-1 for obstacles)

`allowed` is the single source of truth for the traffic rules. It is used by
  * the environment, to reject illegal moves (wrong-way lane, wall, map edge);
  * the observation, so the agent can *see* the lane directions;
  * the action mask, so the policy can never even *choose* an illegal move.

TRAFFIC RULES (one rule per terrain type)
  ROAD          two-way: can move into any neighbouring cell that is not an obstacle.
  HIGHWAY       one-way lane. The highway is horizontal and two lanes wide: the
                northern lane flows LEFT (west), the southern lane flows RIGHT (east).
  INTERSECTION  can move in every direction except "against the lane" it belongs
                to. It is the only place where a plow can change lane or turn
                onto a secondary street.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from Enums import ACTIONS, GridCellTypes, MovementActions

# Integer codes stored in the `codes` array.
OBSTACLE, ROAD, HIGHWAY, INTERSECTION = 0, 1, 2, 3
_CODE_OF = {
    GridCellTypes.OBSTACLE: OBSTACLE,
    GridCellTypes.ROAD: ROAD,
    GridCellTypes.HIGHWAY: HIGHWAY,
    GridCellTypes.INTERSECTION: INTERSECTION,
}
# The key of each category inside a map dict ("roads", "highways", ...).
_CATEGORIES = {cell_type.value: cell_type for cell_type in GridCellTypes}

# The four real moves, with their (dx, dy) displacement.
_MOVES = (MovementActions.UP, MovementActions.DOWN, MovementActions.RIGHT, MovementActions.LEFT)
_STEP = {move: move.value for move in _MOVES}
_NO_OP = ACTIONS.index(MovementActions.DO_NOTHING)


def _code_at(codes: np.ndarray, x: int, y: int) -> int | None:
    """Terrain code at (x, y), or None when (x, y) is outside the map."""
    height, width = codes.shape
    return int(codes[y, x]) if 0 <= x < width and 0 <= y < height else None


def compute_allowed_moves(codes: np.ndarray) -> np.ndarray:
    """Build the traffic-rule table `allowed[y, x, action_index]` (bool).

    DO_NOTHING is legal on every drivable cell; the four real moves follow the
    rules described in the module docstring. A move is only legal if it also
    stays inside the map and does not enter an obstacle.
    """
    height, width = codes.shape
    allowed = np.zeros((height, width, len(ACTIONS)), dtype=bool)

    for y in range(height):
        for x in range(width):
            code = int(codes[y, x])
            if code == OBSTACLE:
                continue
            allowed[y, x, _NO_OP] = True

            # Terrain code of the neighbour reached by each move (None = off the map).
            neighbour = {move: _code_at(codes, x + dx, y + dy) for move, (dx, dy) in _STEP.items()}
            # Moves that do not run into the map edge or an obstacle.
            open_moves = {m for m, c in neighbour.items() if c not in (None, OBSTACLE)}

            if code == ROAD:
                moves = open_moves

            elif code == HIGHWAY:
                # A lane is the *edge* of the highway on the side where the neighbour
                # is not highway terrain: edge above -> northern lane -> flows LEFT.
                edge_above = neighbour[MovementActions.UP] not in (HIGHWAY, INTERSECTION)
                edge_below = neighbour[MovementActions.DOWN] not in (HIGHWAY, INTERSECTION)
                if edge_above == edge_below:
                    raise ValueError(
                        f"Cannot infer the lane direction of the highway cell at (x={x}, y={y}): main "
                        "streets must be horizontal and exactly two cells wide."
                    )
                flow = MovementActions.LEFT if edge_above else MovementActions.RIGHT
                moves = {flow} & open_moves

            else:  # INTERSECTION
                # The intersection column is two cells tall (one per lane). The lower cell
                # belongs to the eastbound lane (cannot go LEFT), the upper cell to the
                # westbound lane (cannot go RIGHT). Moving UP/DOWN switches lane.
                moves = set(open_moves)
                if neighbour[MovementActions.UP] == INTERSECTION:
                    moves.discard(MovementActions.LEFT)
                if neighbour[MovementActions.DOWN] == INTERSECTION:
                    moves.discard(MovementActions.RIGHT)

            for move in moves:
                allowed[y, x, ACTIONS.index(move)] = True

    return allowed


@dataclass(frozen=True)
class RoadMap:
    """Immutable description of the static world. Build it with `RoadMap.from_config`."""

    codes: np.ndarray  # [H, W] int8 terrain code
    allowed: np.ndarray  # [H, W, len(ACTIONS)] bool traffic-rule table
    street_id: np.ndarray  # [H, W] int16, -1 on obstacles
    street_is_main: np.ndarray  # [n_streets] bool (True = main street / highway)
    street_cells: tuple  # per street: (ys, xs) index arrays of its cells

    @property
    def height(self) -> int:
        return self.codes.shape[0]

    @property
    def width(self) -> int:
        return self.codes.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.codes.shape

    @property
    def passable(self) -> np.ndarray:
        """[H, W] bool: True on every drivable cell (everything except obstacles)."""
        return self.codes != OBSTACLE

    @property
    def n_streets(self) -> int:
        return len(self.street_cells)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_config(cls, road_network: dict) -> "RoadMap":
        """Expand a map dict (rectangles) into arrays and validate it."""
        height, width = road_network["size"]
        if height < 3 or width < 3:  # limit of mosaic_multigrid's Grid
            raise ValueError(f"maps must be at least 3x3 cells, got {height}x{width}")
        codes = np.full((height, width), -1, dtype=np.int8)  # -1 = not assigned yet
        street_id = np.full((height, width), -1, dtype=np.int16)
        street_cells: list[tuple[np.ndarray, np.ndarray]] = []
        street_is_main: list[bool] = []

        for category, cell_type in _CATEGORIES.items():
            for rect in road_network.get(category, []):
                row, column = rect["position"]
                n_rows, n_columns = rect["size"]
                if not (0 <= row and 0 <= column and row + n_rows <= height and column + n_columns <= width):
                    raise ValueError(f"{category} rectangle {rect} does not fit in a {height}x{width} map")
                if (codes[row : row + n_rows, column : column + n_columns] != -1).any():
                    raise ValueError(f"{category} rectangle {rect} overlaps another rectangle")
                codes[row : row + n_rows, column : column + n_columns] = _CODE_OF[cell_type]

                if cell_type is not GridCellTypes.OBSTACLE:
                    # One rectangle = one street (see snowplow/maps.py).
                    ys, xs = np.mgrid[row : row + n_rows, column : column + n_columns]
                    street_id[ys, xs] = len(street_cells)
                    street_cells.append((ys.ravel(), xs.ravel()))
                    street_is_main.append(cell_type in (GridCellTypes.HIGHWAY, GridCellTypes.INTERSECTION))

        if (codes == -1).any():
            ys, xs = np.nonzero(codes == -1)
            missing = [(int(y), int(x)) for y, x in zip(ys[:5], xs[:5])]
            raise ValueError(f"Map has cells that belong to no category (row, column): {missing} ...")

        road_map = cls(
            codes=codes,
            allowed=compute_allowed_moves(codes),
            street_id=street_id,
            street_is_main=np.array(street_is_main, dtype=bool),
            street_cells=tuple(street_cells),
        )
        road_map._validate()
        return road_map

    # --------------------------------------------------------------- validate
    def _validate(self) -> None:
        """Reject maps where a plow could get trapped or could not reach some road."""
        real_moves = self.allowed[..., 1:].any(axis=-1)  # has at least one non-wait move
        trapped = self.passable & ~real_moves
        if trapped.any():
            ys, xs = np.nonzero(trapped)
            cells = [(int(x), int(y)) for y, x in zip(ys[:5], xs[:5])]
            raise ValueError(f"Dead-end cells with no legal move (x, y): {cells}")

        # Every drivable cell must be reachable from every other one (strong connectivity),
        # otherwise a plow could be unable to ever finish the job.
        if not (self._reach(forward=True) and self._reach(forward=False)):
            raise ValueError("The road network is not strongly connected: some roads cannot be reached")

    def _reach(self, forward: bool) -> bool:
        """BFS from one drivable cell over the move graph (or over its reverse)."""
        start = tuple(int(v) for v in np.argwhere(self.passable)[0][::-1])  # (x, y)

        def neighbours(x, y):
            out = []
            for move, (dx, dy) in _STEP.items():
                if forward:  # cells that (x, y) can step onto
                    tx, ty, sx, sy = x + dx, y + dy, x, y
                else:  # cells that can step onto (x, y)
                    tx, ty, sx, sy = x - dx, y - dy, x - dx, y - dy
                if 0 <= tx < self.width and 0 <= ty < self.height and 0 <= sx < self.width and 0 <= sy < self.height:
                    if self.allowed[sy, sx, ACTIONS.index(move)]:
                        out.append((tx, ty))
            return out

        seen = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for nxt in neighbours(x, y):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return len(seen) == int(self.passable.sum())
