"""All tunable settings of the snowplow environment in one place.

Nothing else in the code base hard-codes these numbers, so an experiment is
just a different `SnowplowConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SnowConfig:
    """How snow falls and how plows remove it."""

    max_snow: int = 3  # snow levels are 0 (clean) .. max_snow (heavy)
    # LEGACY: the old "drive over a cell and remove up to N layers" rule. Plowing is now
    # time-based (see plow_time_single / plow_time_team below); this field is kept only so
    # that config.json files of older training runs still load. It is not used anymore.
    clear_per_pass: int = 2

    # COOPERATIVE PLOWING. A plow standing on a snowy cell is busy: it cannot leave until the
    # cell is completely clean. How many steps that takes depends on the snow level of the cell
    # and on how many plows work on it together. Index = snow level (index 0 is unused, a clean
    # cell needs no work). The step on which a plow arrives already counts as a working step.
    plow_time_single: tuple[int, ...] = (0, 1, 5, 10)  # one plow alone on the cell
    plow_time_team: tuple[int, ...] = (0, 1, 2, 3)  # two or more plows on the same cell

    # Clouds: hidden from the agents, they only drop snow on the roads below them.
    initial_clouds: int = 10 # clouds that have already snowed when the episode starts
    cloud_spawn_prob: float = 0.05  # chance per step that a new cloud appears
    cloud_min_size: int = 3  # bounding box of a cloud: between 3x3 ...
    cloud_max_size: int = 9  # ... and 9x9 cells (the blob shape is irregular)
    cloud_min_lifetime: int = 1  # steps a cloud keeps snowing
    cloud_max_lifetime: int = 1
    snow_per_step: int = 1  # layers added to each road cell under a cloud, per step
    storm_steps: int | None = None  # clouds only spawn during the first N steps (None = always)

    def __post_init__(self):
        if self.cloud_min_size > self.cloud_max_size or self.cloud_min_size < 1:
            raise ValueError("need 1 <= cloud_min_size <= cloud_max_size")
        if self.cloud_min_lifetime > self.cloud_max_lifetime or self.cloud_min_lifetime < 1:
            raise ValueError("need 1 <= cloud_min_lifetime <= cloud_max_lifetime")
        if not 0.0 <= self.cloud_spawn_prob <= 1.0:
            raise ValueError("cloud_spawn_prob must be in [0, 1]")
        if self.max_snow < 1 or self.clear_per_pass < 1 or self.snow_per_step < 1:
            raise ValueError("max_snow, clear_per_pass and snow_per_step must be >= 1")
        # config.json stores tuples as lists: turn them back into tuples so the config stays hashable.
        object.__setattr__(self, "plow_time_single", tuple(int(t) for t in self.plow_time_single))
        object.__setattr__(self, "plow_time_team", tuple(int(t) for t in self.plow_time_team))
        for name in ("plow_time_single", "plow_time_team"):
            times = getattr(self, name)
            if len(times) != self.max_snow + 1:
                raise ValueError(f"{name} needs one entry per snow level 0..max_snow ({self.max_snow + 1} values)")
            if any(t < 1 for t in times[1:]):
                raise ValueError(f"{name}: plowing a snowy cell takes at least 1 step")

    def plow_time(self, snow_level: int, n_plows: int) -> int:
        """Steps needed to fully clean a cell of `snow_level` when `n_plows` plows work on it together.

        One plow uses `plow_time_single`; two or more plows use `plow_time_team`.
        A clean cell (level 0) or a cell with nobody on it needs no work and returns 0.
        """
        if snow_level <= 0 or n_plows <= 0:
            return 0
        table = self.plow_time_single if n_plows == 1 else self.plow_time_team
        return table[min(int(snow_level), self.max_snow)]


@dataclass(frozen=True)
class SnowplowConfig:
    """Everything that defines one experiment of the snowplow environment."""

    n_plows: int = 1  # number of agents (the env is N-agent from the start)
    max_steps: int = 500  # episode is truncated after this many steps
    observation: str = "full"  # "full" (sees the whole map) or "local" (sees a window)
    view_size: int = 5  # window side for observation="local" (3 or 5)
    # True: several plows may stand on the same cell (needed for cooperative plowing), so they
    # never block each other. False: the old rules (same target / occupied cell / swap = blocked).
    allow_shared_cells: bool = True
    snow: SnowConfig = field(default_factory=SnowConfig)

    def __post_init__(self):
        if self.n_plows < 1:
            raise ValueError("n_plows must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.observation not in ("full", "local"):
            raise ValueError("observation must be 'full' or 'local'")
        if self.view_size not in (3, 5):
            raise ValueError("view_size must be 3 or 5")
