"""Snow on the roads: the snow layer, clouds, and (time-based, cooperative) plow clearing.

Pure numpy, no rendering and no mosaic objects. All randomness comes from the
`rng` (a numpy Generator) that the environment passes in, so a given seed always
produces the same snowfall. Coordinates in the public methods are (x, y) =
(column, row); arrays are indexed [y, x].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snowplow.config import SnowConfig
from snowplow.road_map import RoadMap


@dataclass
class Cloud:
    """A cloud: a fixed set of cells that snows for `ttl` more steps, then vanishes."""

    cells: np.ndarray  # (k, 2) int array of (x, y) cells under the cloud (all inside the map)
    ttl: int  # remaining lifetime in steps


class SnowField:
    """Snow level (0..max_snow) of every cell plus the clouds that produce it."""

    def __init__(self, road_map: RoadMap, config: SnowConfig):
        self.road_map = road_map
        self.cfg = config
        self._passable = road_map.passable
        self.snow = np.zeros(road_map.shape, dtype=np.int8)  # [H, W]
        # Fraction (0..1) of the plowing work already done on every cell. A cell is
        # cleaned completely when this reaches 1 (see plow_step).
        self.plow_progress = np.zeros(road_map.shape, dtype=np.float64)  # [H, W]
        self.clouds: list[Cloud] = []
        self.total_fallen = 0  # layers of snow that fell during this episode

    # -------------------------------------------------------------- lifecycle
    def reset(self, rng: np.random.Generator) -> None:
        """Start a new episode: clean map, then let `initial_clouds` clouds snow."""
        self.snow[:] = 0
        self.plow_progress[:] = 0.0
        self.clouds = []
        self.total_fallen = 0
        for _ in range(self.cfg.initial_clouds):
            # Anchor initial clouds on a road so that every one of them really drops snow.
            cloud = self._new_cloud(rng, on_road=True)
            for _ in range(cloud.ttl):  # it has already snowed for its whole lifetime
                self._snowfall(cloud)

    def age_clouds(self) -> None:
        """Step 7 of the env lifecycle: clouds get one step older; expired ones vanish."""
        for cloud in self.clouds:
            cloud.ttl -= 1
        self.clouds = [cloud for cloud in self.clouds if cloud.ttl > 0]

    def snowfall_step(self, rng: np.random.Generator, step: int) -> int:
        """Step 8 of the env lifecycle: maybe spawn a cloud, then every cloud snows.

        Returns the number of snow layers that fell during this step.
        """
        in_storm = self.cfg.storm_steps is None or step <= self.cfg.storm_steps
        if in_storm and rng.random() < self.cfg.cloud_spawn_prob:
            self.clouds.append(self._new_cloud(rng, on_road=False))
        return sum(self._snowfall(cloud) for cloud in self.clouds)

    # ------------------------------------------------------------------ plows
    def plow_step(self, workers: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
        """One step of time-based (cooperative) plowing.

        `workers` maps a cell (x, y) to the number of plows standing on it this step.
        Every snowy cell with at least one plow gets `1 / plow_time(level, n_plows)` of work:
        one plow on a level-3 cell needs 8 steps (1/8 per step), two plows need 3 steps
        (1/3 per step). Because the work is stored as a fraction, mixed situations also work:
        if a second plow joins half-way, the rest of the work simply goes faster. If snow falls
        on a cell while it is being plowed, the remaining work is done at the new level's speed.

        When the work of a cell reaches 1, ALL its snow is removed at once and its progress
        resets. Cells without any plow keep their progress (in practice a plow cannot leave a
        snowy cell, so unfinished work is never abandoned).

        Returns {(x, y): layers removed} for every cell that became clean during this step.
        """
        cleared = {}
        for (x, y), n_plows in workers.items():
            level = int(self.snow[y, x])
            if level == 0 or n_plows <= 0:
                self.plow_progress[y, x] = 0.0  # nothing to do on a clean cell
                continue
            self.plow_progress[y, x] += 1.0 / self.cfg.plow_time(level, n_plows)
            if self.plow_progress[y, x] >= 1.0 - 1e-9:  # tolerance: 1/3 + 1/3 + 1/3 must count as done
                cleared[(x, y)] = level
                self.snow[y, x] = 0
                self.plow_progress[y, x] = 0.0
        return cleared

    def clear(self, x: int, y: int) -> int:
        """Instantly remove ALL snow from (x, y) (helper for tests/debugging). Returns layers removed."""
        removed = int(self.snow[y, x])
        self.snow[y, x] = 0
        self.plow_progress[y, x] = 0.0
        return removed

    # ---------------------------------------------------------------- queries
    @property
    def is_clean(self) -> bool:
        """True when no road cell has snow."""
        return not self.snow.any()

    def street_snow(self, street: int) -> int:
        """Total layers of snow still on a street."""
        ys, xs = self.road_map.street_cells[street]
        return int(self.snow[ys, xs].sum())

    @property
    def fraction_clean(self) -> float:
        """Fraction of drivable cells that are snow-free."""
        return float(1.0 - (self.snow[self._passable] > 0).mean())

    def cloud_mask(self) -> np.ndarray:
        """[H, W] bool, True under a cloud. Only used for rendering/debugging, never observed."""
        mask = np.zeros(self.snow.shape, dtype=bool)
        for cloud in self.clouds:
            mask[cloud.cells[:, 1], cloud.cells[:, 0]] = True
        return mask

    # -------------------------------------------------------------- internals
    def _new_cloud(self, rng: np.random.Generator, on_road: bool) -> Cloud:
        """Create a cloud with a random size, an irregular (asymmetric) shape and a random lifetime."""
        height, width = self.snow.shape
        h = int(rng.integers(self.cfg.cloud_min_size, self.cfg.cloud_max_size + 1))
        w = int(rng.integers(self.cfg.cloud_min_size, self.cfg.cloud_max_size + 1))

        # Pick the "anchor" cell (the cloud's centre of mass).
        if on_road:
            roads = np.argwhere(self._passable)  # (y, x) rows
            cy, cx = (int(v) for v in roads[rng.integers(len(roads))])
        else:
            cy, cx = int(rng.integers(height)), int(rng.integers(width))

        # Irregular blob: an ellipse whose centre is shifted and whose border is noisy.
        ys, xs = np.mgrid[0:h, 0:w]
        centre_y = (h - 1) / 2 + rng.uniform(-h / 6, h / 6)
        centre_x = (w - 1) / 2 + rng.uniform(-w / 6, w / 6)
        radius = ((ys - centre_y) / (h / 2)) ** 2 + ((xs - centre_x) / (w / 2)) ** 2
        blob = (radius + rng.uniform(-0.35, 0.35, size=(h, w))) <= 1.0
        blob[h // 2, w // 2] = True  # never empty: the anchor cell is always covered

        by, bx = np.nonzero(blob)
        cells = np.stack([bx + cx - w // 2, by + cy - h // 2], axis=1)  # (x, y)
        inside = (cells[:, 0] >= 0) & (cells[:, 0] < width) & (cells[:, 1] >= 0) & (cells[:, 1] < height)
        ttl = int(rng.integers(self.cfg.cloud_min_lifetime, self.cfg.cloud_max_lifetime + 1))
        return Cloud(cells=cells[inside], ttl=ttl)

    def _snowfall(self, cloud: Cloud) -> int:
        """Add `snow_per_step` layers (capped at max_snow) to every ROAD cell under the cloud."""
        xs, ys = cloud.cells[:, 0], cloud.cells[:, 1]
        on_road = self._passable[ys, xs]  # clouds never put snow on buildings
        xs, ys = xs[on_road], ys[on_road]
        before = self.snow[ys, xs].astype(int)
        after = np.minimum(self.cfg.max_snow, before + self.cfg.snow_per_step)
        self.snow[ys, xs] = after
        added = int((after - before).sum())
        self.total_fallen += added
        return added