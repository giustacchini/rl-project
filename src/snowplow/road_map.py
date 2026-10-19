"""The static part of the snowplow world: terrain, traffic rules and streets.

Everything in here is computed ONCE from a map dict (see road_network_configs.py) and
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
  ROAD          two-way: can move into any neighboring cell that is not an obstacle.
  HIGHWAY       one-way lane; nearby obstacles determine its horizontal/vertical flow.
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

            # Terrain code of the neighbor reached by each move (None = off the map).
            neighbor = {move: _code_at(codes, x + dx, y + dy) for move, (dx, dy) in _STEP.items()}
            # Moves that do not run into the map edge or an obstacle.
            open_moves = {m for m, c in neighbor.items() if c not in (None, OBSTACLE)}

            if code == ROAD:
                moves = open_moves

            elif code == HIGHWAY:
                # Looking for the closest cell that is obstacle to determine the traffic flow direction.
                edge_above = neighbor[MovementActions.UP] == OBSTACLE
                edge_below = neighbor[MovementActions.DOWN] == OBSTACLE
                edge_left = neighbor[MovementActions.LEFT] == OBSTACLE

                if edge_above:
                    flow = MovementActions.LEFT
                elif edge_below:
                    flow = MovementActions.RIGHT
                elif edge_left:
                    flow = MovementActions.DOWN
                else:  # Right edge is obstacle
                    flow = MovementActions.UP

                moves = {flow} & open_moves

            else:  # INTERSECTION
                # The intersection column is two cells tall (one per lane). The lower cell
                # belongs to the eastbound lane (cannot go LEFT), the upper cell to the
                # westbound lane (cannot go RIGHT). Moving UP/DOWN switches lane.

                # "IF INTERSECTIONS ARE PLACED (FROM CURRENT CELL) AT DIRECTION: IF#1 + IF#2"

                moves = set(open_moves)
                # Handle both single and overlapping highways
                if neighbor[MovementActions.UP] == INTERSECTION:
                    moves.discard(MovementActions.LEFT)

                if neighbor[MovementActions.LEFT] == INTERSECTION:
                    moves.discard(MovementActions.DOWN)

                if neighbor[MovementActions.DOWN] == INTERSECTION:
                    moves.discard(MovementActions.RIGHT)

                if neighbor[MovementActions.RIGHT] == INTERSECTION:
                    moves.discard(MovementActions.UP)

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
        """Expand rectangles, with later terrain layers replacing earlier cells.

        Category priority follows GridCellTypes: obstacles, roads, intersections,
        highways. Unassigned cells become obstacles. Connectivity is an optional
        diagnostic through _validate(), not required for fixed-horizon episodes.
        """
        height, width = road_network["size"]
        if height < 3 or width < 3:  # limit of mosaic_multigrid's Grid
            raise ValueError(f"maps must be at least 3x3 cells, got {height}x{width}")
        codes = np.full((height, width), -1, dtype=np.int8)  # -1 = not assigned yet
        street_id = np.full((height, width), -1, dtype=np.int16)
        street_is_main: list[bool] = []

        for category, cell_type in _CATEGORIES.items():
            for rect in road_network.get(category, []):
                row, column = rect["position"]
                n_rows, n_columns = rect["size"]
                if not (n_rows > 0 and n_columns > 0 and 0 <= row and 0 <= column and row + n_rows <= height and column + n_columns <= width):
                    raise ValueError(f"{category} rectangle {rect} does not fit in a {height}x{width} map")
                codes[row : row + n_rows, column : column + n_columns] = _CODE_OF[cell_type]

                if cell_type is not GridCellTypes.OBSTACLE:
                    # One rectangle = one candidate street; overlaps are normalized below.
                    ys, xs = np.mgrid[row : row + n_rows, column : column + n_columns]
                    street_id[ys, xs] = len(street_is_main)
                    street_is_main.append(cell_type in (GridCellTypes.HIGHWAY, GridCellTypes.INTERSECTION))

        # Treat every unassigned cell as an obstacle in the completed map.
        codes[codes == -1] = OBSTACLE
        if not (codes != OBSTACLE).any():
            raise ValueError("Map must contain at least one drivable cell")

        # Rebuild street membership from the final layout. An overlaid cell belongs
        # only to its final street, not to every rectangle originally covering it.
        street_id[codes == OBSTACLE] = -1
        final_ids = np.full_like(street_id, -1)
        final_cells, final_main = [], []
        for old_id, is_main in enumerate(street_is_main):
            ys, xs = np.nonzero(street_id == old_id)
            if len(ys):
                final_ids[ys, xs] = len(final_cells)
                final_cells.append((ys, xs))
                final_main.append(is_main)

        road_map = cls(
            codes=codes,
            allowed=compute_allowed_moves(codes),
            street_id=final_ids,
            street_is_main=np.array(final_main, dtype=bool),
            street_cells=tuple(final_cells),
        )
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

        def neighbors(x, y):
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
            for nxt in neighbors(x, y):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return len(seen) == int(self.passable.sum())
