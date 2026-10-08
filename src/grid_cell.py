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
        else:
            cell.color = Color.grey
        return cell

    def can_overlap(self) -> bool:
        return self.cell_type != GridCellTypes.OBSTACLE

    def render(self, img):
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), self.color.rgb())
