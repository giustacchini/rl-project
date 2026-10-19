# python
# shared/rendering.py
"""
Shared PNG export and dependency-free text rendering.

PNG export uses the environment's renderer, including domain-specific visuals.
Text rendering shows blocking terrain and agent positions without opening a window.
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
