"""Non-learning reference policies. Any trained agent should beat these.

Both policies only use what a plow *observes* (the obs dict), never the env internals:

    RandomPolicy            picks a uniformly random LEGAL action (uses the action mask).
    GreedyPolicy            drives along the shortest legal path to the nearest snowy cell
                            (breadth-first search over the lane-direction channels). Every plow
                            works ALONE: it never targets a cell another plow is working on.
    CooperativeGreedyPolicy like GreedyPolicy, but an idle plow first checks whether it can reach
                            a teammate's heavy-snow cell before the teammate finishes alone; if
                            so it goes there to plow together (faster with the team plow times).

A busy plow (standing on a snowy cell) can only DO_NOTHING; both greedy policies read that
from the action mask and simply wait until the cell is clean.

A policy is a callable:  policy(obs: dict[plow_id, obs_dict]) -> dict[plow_id, action_index]
and has `reset()` to be called at the start of each episode.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from Enums import ACTIONS, MovementActions

# Grid channels (see snowplow/env.py GRID_CHANNELS): 0 = snow, 5..8 = can_up/down/right/left,
# 9 = plowing progress.
_MOVE_CHANNELS = {
    MovementActions.UP: 5,
    MovementActions.DOWN: 6,
    MovementActions.RIGHT: 7,
    MovementActions.LEFT: 8,
}


class RandomPolicy:
    def __init__(self, seed: int = 0, legal_only: bool = True):
        self.seed, self.legal_only = seed, legal_only
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, obs: dict) -> dict:
        actions = {}
        for agent_id, o in obs.items():
            choices = np.flatnonzero(o["action_mask"]) if self.legal_only else np.arange(len(ACTIONS))
            actions[agent_id] = int(self.rng.choice(choices))
        return actions


_DO_NOTHING = ACTIONS.index(MovementActions.DO_NOTHING)
_PROGRESS_CHANNEL = 9


def _is_busy(o: dict) -> bool:
    """True if the action mask only allows DO_NOTHING because the plow is plowing its cell."""
    mask = np.asarray(o["action_mask"], dtype=bool)
    return bool(mask[_DO_NOTHING] and mask.sum() == 1 and o["grid"][0][o["self_position"][1], o["self_position"][0]] > 0)


def _shortest_paths(grid: np.ndarray, start: tuple[int, int]) -> dict:
    """Breadth-first search over the legal moves from `start`.

    Returns {cell: (distance, first action index used to leave `start`)} for every reachable
    cell (`start` itself has distance 0 and first action None).
    """
    _, height, width = grid.shape
    paths = {start: (0, None)}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        dist, first = paths[(x, y)]
        for move, channel in _MOVE_CHANNELS.items():
            if grid[channel, y, x] < 0.5:  # this move is not allowed from (x, y)
                continue
            dx, dy = move.value
            nxt = (x + dx, y + dy)
            if nxt in paths or not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            paths[nxt] = (dist + 1, first if first is not None else ACTIONS.index(move))
            queue.append(nxt)
    return paths


def _nearest_snow(grid: np.ndarray, paths: dict, taken: set):
    """(first action, target) of the closest reachable snowy cell that is not in `taken`, or (None, None).

    The start cell is never a target: a plow on a snowy cell is busy and does not plan.
    """
    snow = grid[0] > 0
    best = None
    for cell, (dist, first) in paths.items():
        if first is None or cell in taken or not snow[cell[1], cell[0]]:
            continue
        if best is None or dist < best[0]:
            best = (dist, first, cell)
    return (None, None) if best is None else (best[1], best[2])


class GreedyPolicy:
    """Nearest-snow-first, every plow works alone.

    With several plows each plow claims a different target, and cells where a teammate is
    already plowing are left to that teammate (no cooperation: this is the "solo" reference).
    """

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict) -> dict:
        actions = {}
        # Cells already being plowed by somebody belong to that plow.
        taken = {tuple(o["self_position"]) for o in obs.values() if _is_busy(o)}
        for agent_id, o in obs.items():
            if _is_busy(o):
                actions[agent_id] = _DO_NOTHING  # keep plowing until the cell is clean
                continue
            paths = _shortest_paths(o["grid"], tuple(o["self_position"]))
            action, target = _nearest_snow(o["grid"], paths, taken)
            if target is None:  # nothing reachable with snow: wait
                action = _DO_NOTHING
            else:
                taken.add(target)
            actions[agent_id] = action
        return actions


class CooperativeGreedyPolicy:
    """Greedy, but idle plows join a busy teammate when working together saves time.

    For every cell where a teammate is plowing alone, an idle plow estimates:
        remaining_alone = (1 - progress) * plow_time_single[level]   steps left for the teammate
        distance        = BFS steps needed to get there
    It goes to help if it can arrive before the teammate finishes alone AND the team plow time of
    that level is shorter than the solo one (no point helping on level 1). Otherwise it behaves
    like GreedyPolicy. At most one helper is sent to each cell. This is a hand-written picture of
    the behaviour we hope the learners discover by themselves.
    """

    def __init__(self, plow_time_single=(0, 1, 4, 8), plow_time_team=(0, 1, 2, 3), max_snow: int = 3):
        self.single, self.team, self.max_snow = tuple(plow_time_single), tuple(plow_time_team), max_snow

    @classmethod
    def from_config(cls, snow_config) -> "CooperativeGreedyPolicy":
        """Build the policy with the plow times of a SnowConfig."""
        return cls(snow_config.plow_time_single, snow_config.plow_time_team, snow_config.max_snow)

    def reset(self) -> None:
        pass

    def _level(self, grid: np.ndarray, cell: tuple[int, int]) -> int:
        """Snow level of a cell from the normalised snow channel."""
        return int(round(float(grid[0, cell[1], cell[0]]) * self.max_snow))

    def __call__(self, obs: dict) -> dict:
        actions = {}
        busy_cells = {}  # cell -> number of plows already working / heading there
        for o in obs.values():
            if _is_busy(o):
                cell = tuple(o["self_position"])
                busy_cells[cell] = busy_cells.get(cell, 0) + 1
        taken = set(busy_cells)

        for agent_id, o in obs.items():
            if _is_busy(o):
                actions[agent_id] = _DO_NOTHING
                continue
            grid = o["grid"]
            paths = _shortest_paths(grid, tuple(o["self_position"]))

            # 1. Is there a lonely teammate I can help in time?
            best_help = None
            for cell, n_workers in busy_cells.items():
                if n_workers != 1 or cell not in paths or paths[cell][1] is None:
                    continue
                level = self._level(grid, cell)
                if level <= 0 or self.team[level] >= self.single[level]:
                    continue  # team plowing is not faster for this level
                progress = float(grid[_PROGRESS_CHANNEL, cell[1], cell[0]])
                remaining_alone = (1.0 - progress) * self.single[level]
                dist, first = paths[cell]
                if dist < remaining_alone and (best_help is None or dist < best_help[0]):
                    best_help = (dist, first, cell)
            if best_help is not None:
                actions[agent_id] = best_help[1]
                busy_cells[best_help[2]] += 1  # one helper per cell is enough
                continue

            # 2. Otherwise: nearest free snowy cell, like GreedyPolicy.
            action, target = _nearest_snow(grid, paths, taken)
            if target is None:
                action = _DO_NOTHING
            else:
                taken.add(target)
            actions[agent_id] = action
        return actions
