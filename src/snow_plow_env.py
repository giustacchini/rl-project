"""Snow-plow environment with a configured traffic grid and shared movement."""

import math
from typing import Callable

import numpy as np
from mosaic_multigrid.base import AgentID, MultiGridEnv
from mosaic_multigrid.core.grid import Grid
from mosaic_multigrid.core.world_object import WorldObj

from Enums import GridCellTypes, MovementActions
from grid_cell import GridCell
from shared.grid import is_statically_blocked
from shared.movement import handle_simultaneous_movement


class SnowPlowEnv(MultiGridEnv):
    """Build traffic terrain and resolve simultaneous agent movement."""

    def __init__(
        self, agents, width, height, max_steps, render_mode=None, road_network=None
    ):
        if road_network is not None and road_network["size"] != (height, width):
            raise ValueError(
                "Road network size (rows, columns) must match (height, width)."
            )
        self.road_network = road_network
        super().__init__(
            agents=agents,
            width=width,
            height=height,
            max_steps=max_steps,
            render_mode=render_mode,
        )
        # Keep a usable empty grid before reset; agent placement happens on reset.
        if road_network is None:
            self.grid.wall_rect(0, 0, width, height)

    def _gen_grid(self, width: int, height: int):
        """Expand configured rectangles and place agents on walkable terrain."""
        self.grid = Grid(width, height)
        if self.road_network is None:
            self.grid.wall_rect(0, 0, width, height)
            for agent in self.agents:
                self.place_agent(agent)
            return

        cell_types = {
            "roads": GridCellTypes.ROAD,
            "highways": GridCellTypes.HIGHWAY,
            "intersections": GridCellTypes.INTERSECTION,
            "obstacles": GridCellTypes.OBSTACLE,
        }
        for category, cell_type in cell_types.items():
            for rectangle in self.road_network[category]:
                row, column = rectangle["position"]
                row_count, column_count = rectangle["size"]
                for x in range(column, column + column_count):
                    for y in range(row, row + row_count):
                        self.grid.set(x, y, GridCell(cell_type=cell_type))

        for agent in self.agents:
            self.place_agent(agent)

    def place_obj(
        self,
        obj: WorldObj | None,
        top: tuple[int, int] | None = None,
        size: tuple[int, int] | None = None,
        reject_fn: Callable[[MultiGridEnv, tuple[int, int]], bool] | None = None,
        max_tries=math.inf,
    ) -> tuple[int, int]:
        """Sample an unoccupied walkable spawn cell, preserving its terrain.

        Other objects retain the parent implementation's empty-cell placement.
        """
        if obj is not None:
            return super().place_obj(obj, top, size, reject_fn, max_tries)

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0
        while True:
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')
            num_tries += 1

            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            # Reject blocking terrain.
            if is_statically_blocked(self.grid, pos):
                continue

            # Don't place where agents are
            if np.all(self.agent_states.pos == pos, axis=1).any():
                continue

            # Custom rejection
            if reject_fn and reject_fn(self, pos):
                continue

            break

        return pos

    def is_move_allowed(self, grid, agent, current_pos, intended_pos) -> bool:
        """
        Domain-specific movement hook.
        Issue #6
        overrides this to reject wrong-direction moves on one-way lanes.
        """
        return True

    def handle_actions(self, actions: dict[AgentID, MovementActions]):
        movement_results = handle_simultaneous_movement(
            agents=self.agents,
            grid=self.grid,
            actions=actions,
            width=self.width,
            height=self.height,
            is_move_allowed=self.is_move_allowed,
        )
        self.last_movement_results = movement_results

        # Placeholder rewards -- actual SnowPlow rewards are for future issues
        rewards = {agent.index: 0 for agent in self.agents}
        return rewards
