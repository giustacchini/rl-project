"""Non-learning reference policies. Any trained agent should beat these.

Both policies only use what a plow *observes* (the obs dict), never the env internals:

    RandomPolicy  picks a uniformly random LEGAL action (uses the action mask).
    GreedyPolicy  drives along the shortest legal path to the nearest snowy cell
                  (breadth-first search over the lane-direction channels).

A policy is a callable:  policy(obs: dict[plow_id, obs_dict]) -> dict[plow_id, action_index]
and has `reset()` to be called at the start of each episode.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from Enums import ACTIONS, MovementActions

# Grid channels (see snowplow/env.py GRID_CHANNELS): 0 = snow, 5..8 = can_up/down/right/left.
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


class GreedyPolicy:
    """Nearest-snow-first. With several plows, each plow claims a different target."""

    def reset(self) -> None:
        pass

    @staticmethod
    def _first_move_to_snow(grid: np.ndarray, start: tuple[int, int], taken: set):
        """BFS from `start`; return (first action index, target cell) of the closest snowy cell.

        The start cell itself is not a target: a plow only clears cells it drives ONTO.
        """
        _, height, width = grid.shape
        snow = grid[0] > 0
        queue = deque([(start, None)])  # (cell, first action used to leave `start`)
        seen = {start}
        while queue:
            (x, y), first = queue.popleft()
            if first is not None and snow[y, x] and (x, y) not in taken:
                return first, (x, y)
            for move, channel in _MOVE_CHANNELS.items():
                if grid[channel, y, x] < 0.5:  # this move is not allowed from (x, y)
                    continue
                dx, dy = move.value
                nxt = (x + dx, y + dy)
                if nxt in seen or not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                    continue
                seen.add(nxt)
                queue.append((nxt, first if first is not None else ACTIONS.index(move)))
        return None, None

    def __call__(self, obs: dict) -> dict:
        actions, taken = {}, set()
        for agent_id, o in obs.items():
            action, target = self._first_move_to_snow(o["grid"], tuple(o["self_position"]), taken)
            if target is None:  # nothing reachable with snow: wait
                action = ACTIONS.index(MovementActions.DO_NOTHING)
            else:
                taken.add(target)
            actions[agent_id] = action
        return actions
