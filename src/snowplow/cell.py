"""A map cell as a mosaic WorldObj. It exists ONLY for drawing and for blocking.

The simulation state (snow, traffic rules) lives in numpy arrays (see
snowplow/road_map.py and snowplow/snow.py). The environment copies the snow level
into these cells right before drawing, so a cell never decides anything.

How snow gets onto the picture: a WorldObj is stored as an array
(type, color, state). mosaic caches rendered tiles keyed by that array, so the
snow level and the cloud flag are packed into the `state` slot -- when they
change, the key changes and the tile is redrawn (a plain Python attribute would
leave a stale cached tile, which is why the old rendering looked broken).
"""

from __future__ import annotations

from mosaic_multigrid.core import Color
from mosaic_multigrid.core.world_object import WorldObj
from mosaic_multigrid.utils.rendering import fill_coords, point_in_rect

from snowplow import road_map as rm

# One colour per terrain type (RGB). Registered once in mosaic's colour table.
_TERRAIN_COLOURS = {
    rm.OBSTACLE: ("building", (70, 120, 70)),
    rm.ROAD: ("secondary_road", (120, 120, 130)),
    rm.HIGHWAY: ("main_road", (70, 70, 80)),
    rm.INTERSECTION: ("crossing", (95, 95, 110)),
}
for _name, _rgb in _TERRAIN_COLOURS.values():
    if _name not in set(Color):
        Color.add_color(_name, _rgb)

# How strongly white snow is blended over the road colour, per snow level 0..3.
_SNOW_ALPHA = (0.0, 0.45, 0.75, 1.0)
_CLOUD_TINT = (40, 50, 90)  # dark blue overlay for cells under a (hidden) cloud
_CLOUD_ALPHA = 0.35


class GridCell(WorldObj):
    """One cell of the map: terrain colour + optional snow + optional cloud tint."""

    def __new__(cls, terrain: int = rm.ROAD):
        cell = super().__new__(cls, color=_TERRAIN_COLOURS[terrain][0])
        cell.terrain = terrain
        return cell

    def can_overlap(self) -> bool:
        """Agents can drive on everything except obstacles."""
        return self.terrain != rm.OBSTACLE

    def set_appearance(self, snow_level: int, under_cloud: bool) -> None:
        """Pack (snow level 0..3, cloud flag) into the state slot used as render-cache key."""
        self[WorldObj.STATE] = int(snow_level) + 4 * int(under_cloud)

    def render(self, img) -> None:
        state = int(self[WorldObj.STATE])
        snow, under_cloud = state % 4, state // 4
        rgb = [float(v) for v in self.color.rgb()]
        alpha = _SNOW_ALPHA[snow]
        rgb = [c * (1 - alpha) + 255 * alpha for c in rgb]  # blend towards white
        # if under_cloud:
        #     rgb = [c * (1 - _CLOUD_ALPHA) + t * _CLOUD_ALPHA for c, t in zip(rgb, _CLOUD_TINT)]
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), tuple(int(v) for v in rgb))
