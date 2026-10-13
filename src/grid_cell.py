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
        movement_actions: list[MovementActions] = [MovementActions.DO_NOTHING],
        color: Color = Color.blue,
    ):

        # Specifically for GridCell, we want to store the cell_type and snow_level as attributes, so we override __init__.
        # def __init__(
        #     self,
        #     row: int,
        #     column: int,
        #     cell_type: GridCellTypes = GridCellTypes.ROAD,
        #     movement_actions: list[MovementActions] = [MovementActions.DO_NOTHING],
        #     snow_level: int = 1,
        # ):
        cls.row = row
        cls.column = column
        cls.movement_actions = movement_actions
        # super().__init__((row, column))
        cls.cell_type = cell_type
        cls.pos = np.array((row, column))

        new_color = color
        if cls.cell_type.name == GridCellTypes.ROAD.name:
            new_color = Color.green
        # elif cls.cell_type == GridCellTypes.INTERSECTION:
        #     cls.color = Color.darkgrey  # Dark gray for intersections
        # else:
        # cls.color = Color.red

        # print(cls.row, cls.column, cls.cell_type)

        cls.color = new_color
        cell = super().__new__(cls, color=new_color)
        return cell

    def can_overlap(self) -> bool:
        return self.cell_type != GridCellTypes.OBSTACLE

    def render(self, img):
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), self.color.rgb())

    def set_snow_level(self, snow_level: int):
        # Snow attributes
        if snow_level > 0 and self.cell_type is not GridCellTypes.OBSTACLE:
            self.color = Color.white  # White for snow-covered cells

        if not isinstance(snow_level, int) or snow_level < 0:
            raise ValueError("snow_level must be a non-negative integer")
        elif snow_level > 3:
            raise ValueError("snow_level must be between 0 and 3")
        else:
            self.snow_level = snow_level

        # self.render()
