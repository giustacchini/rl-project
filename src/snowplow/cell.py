"""A map cell as a mosaic WorldObj. It exists ONLY for drawing and for blocking.

The simulation state (snow, traffic rules) lives in numpy arrays (see
snowplow/road_map.py and snowplow/snow.py). The environment copies the snow level
into these cells right before drawing, so a cell never decides anything.

How snow gets onto the picture: a WorldObj is stored as an array
(type, color, state). mosaic caches rendered tiles keyed by that array, so the
snow level, the cloud flag and the plowing progress are packed into the `state` slot -- when they
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

# How strongly white snow is blended over the road colour, per snow level 0..3
# (more snow = whiter / less transparent, so the three levels are easy to tell apart).
_SNOW_ALPHA = (0.0, 0.45, 0.75, 1.0)
# Plowing progress bar drawn at the bottom of a cell that is being plowed.
_PROGRESS_BUCKETS = 4  # progress is shown in quarters (keeps the render cache small)
_PROGRESS_COLOUR = (255, 140, 0)  # orange: one plow working alone
_TEAM_PROGRESS_COLOUR = (0, 200, 255)  # cyan: 2+ plows on the cell (only one plow triangle is visible then)
_CLOUD_TINT = (219, 184, 12)  # dark blue overlay for cells under a (hidden) cloud
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

    def set_appearance(self, snow_level: int, under_cloud: bool, progress: float = 0.0, team: bool = False) -> None:
        """Pack (snow level 0..3, cloud flag, plowing progress, team flag) into the state slot (render-cache key).

        Layout: snow + 4 * cloud + 8 * progress_bucket + 32 * team, where progress_bucket is the plowing
        progress (0..1) rounded down to quarters (0..3). A cell with work started shows at least
        one quarter so it is visible that somebody is plowing it.
        """
        bucket = 0
        if progress > 0:
            bucket = max(1, min(_PROGRESS_BUCKETS - 1, int(progress * _PROGRESS_BUCKETS)))
        self[WorldObj.STATE] = min(int(snow_level), 3) + 4 * int(under_cloud) + 8 * bucket + 32 * int(team)

    def render(self, img) -> None:
        """Draw terrain colour, blend in white snow (per level), cloud tint and the plowing progress bar."""
        state = int(self[WorldObj.STATE])
        snow, under_cloud, bucket, team = state % 4, (state // 4) % 2, (state // 8) % 4, state // 32
        rgb = [float(v) for v in self.color.rgb()]
        alpha = _SNOW_ALPHA[snow]
        rgb = [c * (1 - alpha) + 255 * alpha for c in rgb]  # blend towards white
        if under_cloud:
            rgb = [c * (1 - _CLOUD_ALPHA) + t * _CLOUD_ALPHA for c, t in zip(rgb, _CLOUD_TINT)]
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), tuple(int(v) for v in rgb))
        if bucket:  # progress bar: grows from left to right while the plows work on the cell
            fill_coords(img, point_in_rect(0.08, 0.08 + 0.84 * bucket / _PROGRESS_BUCKETS, 0.84, 0.94),
                        _TEAM_PROGRESS_COLOUR if team else _PROGRESS_COLOUR)
