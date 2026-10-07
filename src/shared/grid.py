# python
# shared/grid.py
"""
Both snowplow and soccer can reuse these functions unchanged, neither
function below knows anything about snow, roads, or balls.
"""

from __future__ import annotations

import numpy as np


def in_bounds(width: int, height: int, pos) -> bool:
    """True if `pos` = (x, y) lies inside a `width` x `height` grid."""
    x, y = pos
    return 0 <= x < width and 0 <= y < height


def is_statically_blocked(grid, pos) -> bool:
    """
    True if the grid's static layer (walls, forbidden cells, ...) blocks
    this cell.
    """
    cell = grid.get(*pos)
    return cell is not None and not cell.can_overlap()


def agent_at(agent_states, pos, ignore_agent_id: int | None = None) -> int | None:
    """
    Return the index of the agent currently standing at `pos`, or None.

    `agent_states.pos` is an (N, 2) array (mosaic_multigrid's
    `AgentState`)
    """
    pos = np.asarray(pos)
    matches = np.all(agent_states.pos == pos, axis=1)
    for agent_id, is_match in enumerate(matches):
        if is_match and agent_id != ignore_agent_id:
            return agent_id
    return None


def is_cell_free(grid, agent_states, pos, width: int, height: int,
                  ignore_agent_id: int | None = None) -> bool:
    """
    True if `pos` is in bounds, not statically blocked, and not occupied
    by another agent. This is "is this move even possible"
    check that we can use for both projects, it can be call before applying
    their own extra rules like lane direction, goalkeeper area
    """
    if not in_bounds(width, height, pos):
        return False
    if is_statically_blocked(grid, pos):
        return False
    if agent_at(agent_states, pos, ignore_agent_id=ignore_agent_id) is not None:
        return False
    return True
