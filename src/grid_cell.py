from mosaic_multigrid.core import Color
from mosaic_multigrid.core.world_object import WorldObj
from mosaic_multigrid.utils.rendering import fill_coords, point_in_rect

from Enums import GridCellTypes


class GridCell(WorldObj):
    """Traffic terrain with walkability and a distinct obstacle color."""

    def __new__(
        cls,
        color: Color = Color.blue,
        cell_type: GridCellTypes = GridCellTypes.ROAD,
        snow_level: int = 1,
    ):
        cell = super().__new__(cls, color=color)
        cell.cell_type = cell_type
        if cell.cell_type == GridCellTypes.OBSTACLE:
            cell.color = Color.green
        # elif cell.cell_type == GridCellTypes.INTERSECTION:
        #     cell.color = (100, 100, 100)  # Darker gray than regular road to specify an intersection
        else:
            cell.color = Color.grey

        if snow_level > 0 and cell_type is not GridCellTypes.OBSTACLE:
            cell.color = (255, 255, 255)  # White for snow-covered cells

        return cell

    def __init__(self, pos: tuple[int, int], cell_type: GridCellTypes, snow_level: int = 1):
        super().__init__(pos)
        self.cell_type = cell_type

        if not isinstance(snow_level, int) or snow_level < 0:
            raise ValueError("snow_level must be a non-negative integer")
        elif snow_level > 3:
            raise ValueError("snow_level must be between 0 and 3")
        else:
            self.snow_level = snow_level

    def can_overlap(self) -> bool:
        return self.cell_type != GridCellTypes.OBSTACLE

    def render(self, img):
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), self.color.rgb())
