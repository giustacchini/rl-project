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
    clear_per_pass: int = 2  # layers a plow removes from a cell it drives onto

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


@dataclass(frozen=True)
class SnowplowConfig:
    """Everything that defines one experiment of the snowplow environment."""

    n_plows: int = 1  # number of agents (the env is N-agent from the start)
    max_steps: int = 300  # episode is truncated after this many steps
    observation: str = "full"  # "full" (sees the whole map) or "local" (sees a window)
    view_size: int = 5  # window side for observation="local" (3 or 5)
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
