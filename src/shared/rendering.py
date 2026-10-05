# python
# shared/rendering.py
"""
Debug rendering for issue #5.

mosaic_multigrid's `MultiGridEnv.render()` / `get_frame()` are already
generic: they call `self.grid.render(tile_size, agents=self.agents, ...)`
with no sport-specific code in the base class (sport renderers only
override `get_full_render()` in soccer_game.py / basketball_game.py /
american_football_game.py for court markings). So for SnowPlow -- which
doesn't override `get_full_render` -- `env.render()` with
`render_mode='rgb_array'` already gives a correct top-down image with
every agent drawn, no custom renderer needed.

This module adds two things on top of that:
1. `save_frame()` - a one-liner to dump a PNG for a README/report figure.
2. `ascii_grid()` - a fast, dependency-free text render for use inside
   pytest / CI, where spinning up pygame for every test is unnecessary
   overhead and can be flaky in headless environments.
"""

from __future__ import annotations


def save_frame(env, path: str, tile_size: int = 32) -> None:
    """Render the current state to a PNG using Mosaic's own renderer."""
    import numpy as np
    from PIL import Image

    frame = env.get_frame(highlight=True, tile_size=tile_size)
    Image.fromarray(np.asarray(frame)).save(path)


def ascii_grid(env) -> str:
    """
    Fast text render for debugging/tests: one character per cell.
    '#' = static obstacle (can't overlap), digit = agent index,
    '.' = empty/walkable cell.
    """
    width, height = env.width, env.height
    rows = []
    for y in range(height):
        row_chars = []
        for x in range(width):
            cell = env.grid.get(x, y)
            char = "#" if (cell is not None and not cell.can_overlap()) else "."
            row_chars.append(char)
        rows.append(row_chars)

    for agent in env.agents:
        x, y = agent.state.pos
        rows[y][x] = str(agent.index)

    return "\n".join("".join(row) for row in rows)


def print_grid(env) -> None:
    print(ascii_grid(env))
    print(f"step={getattr(env, 'step_count', '?')}")
