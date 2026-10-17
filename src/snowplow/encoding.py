"""Turn the environment's dict observations into flat float vectors for the learners.

The learners (PPO/MAPPO) work with plain vectors. These two functions are the ONLY
place that knows how a snowplow observation is laid out, so changing what the
agent sees means changing this file, not the algorithms.

Actor input (per plow), concatenated:
    grid           [C, h, w]  snow/3, terrain one-hot (4), allowed moves (4) -> lane directions,
                              plowing progress (1)
    self map       [h, w]     1 on the cell where this plow is
    others map     [h, w]     1 on the cells of the other plows it can see
    action mask    [n_actions]  1 for every legal action
For the full view h x w is the whole map; for the local view it is the window.
With the local view the plow does not get its absolute position (partial observability).

Critic state (MAPPO's centralised critic): the full grid plus one position map per plow.
"""

from __future__ import annotations

import numpy as np


def _position_map(positions, shape: tuple[int, int], origin: tuple[int, int]) -> np.ndarray:
    """[h, w] map with 1 at every (x, y) in `positions` that falls inside the view."""
    height, width = shape
    out = np.zeros(shape, dtype=np.float32)
    for x, y in positions:
        col, row = x - origin[0], y - origin[1]
        if 0 <= row < height and 0 <= col < width:
            out[row, col] = 1.0
    return out


def encode_observation(obs: dict) -> np.ndarray:
    """One agent's observation dict -> 1-D float32 vector (see module docstring)."""
    grid = np.asarray(obs["grid"], dtype=np.float32)
    shape = grid.shape[1:]
    origin = obs.get("view_origin", (0, 0))
    # Other plows are recognised by their id, not their position: plows can share a cell, and a
    # teammate on the same cell (= working together) must still show up in the "others" map.
    self_id = obs.get("self_id")
    if self_id is None:  # observations without an id: fall back to comparing positions
        others = [pos for pos in obs["agent_positions"].values() if pos != tuple(obs["self_position"])]
    else:
        others = [pos for agent_id, pos in obs["agent_positions"].items() if agent_id != self_id]
    parts = [
        grid.ravel(),
        _position_map([obs["self_position"]], shape, origin).ravel(),
        _position_map(others, shape, origin).ravel(),
    ]
    if "action_mask" in obs:
        parts.append(np.asarray(obs["action_mask"], dtype=np.float32))
    return np.concatenate(parts)


def encode_state(env) -> np.ndarray:
    """Global state for a centralised critic: full grid + one position map per plow (in id order)."""
    grid = env.get_observable_grid().astype(np.float32)
    shape = grid.shape[1:]
    maps = [_position_map([tuple(agent.state.pos)], shape, (0, 0)) for agent in env.agents]
    return np.concatenate([grid.ravel()] + [m.ravel() for m in maps])
