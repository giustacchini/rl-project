from __future__ import annotations

import numpy as np
from mosaic_multigrid.core import Color
from mosaic_multigrid.core.world_object import WorldObj
from mosaic_multigrid.utils.rendering import fill_coords, point_in_rect

from Enums import GridCellTypes, MovementActions


class GridCell(WorldObj):
    """Traffic terrain with walkability and a distinct obstacle color."""

    def __new__(
        cls,
        row: int,
        column: int,
        cell_type: GridCellTypes = GridCellTypes.ROAD,
        color: Color = Color.blue,
    ):
        new_color = color
        if cell_type == GridCellTypes.OBSTACLE:
            new_color = Color.green
        cell = super().__new__(cls, color=new_color)
        cell.row = row
        cell.column = column
        cell.snow_level = 0
        cell.cell_type = cell_type
        cell.pos = np.array((row, column))
        return cell

    def can_overlap(self) -> bool:
        return self.cell_type != GridCellTypes.OBSTACLE

    def render(self, img):
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), self.color.rgb())

    def set_snow_level(self, snow_level: int):
        if not isinstance(snow_level, int) or snow_level < 0:
            raise ValueError("snow_level must be a non-negative integer")
        elif snow_level > 3:
            raise ValueError("snow_level must be between 0 and 3")
        else:
            self.snow_level = snow_level

        # Snow attributes
        if snow_level > 0 and self.cell_type is not GridCellTypes.OBSTACLE:
            if snow_level == 3:
                self.color = Color.snow_level3  # White for snow-covered cells (level 3)
            if snow_level == 2:
                self.color = Color.snow_level2
            if snow_level == 1:
                self.color = Color.snow_level1  # More towards light blue for lower snow levels

    def set_movement_action(self, movement_actions: list[MovementActions] | None = None):
        self.movement_actions = movement_actions
