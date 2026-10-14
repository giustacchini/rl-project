"""Snow-plow environment with a configured traffic grid and shared movement."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from mosaic_multigrid.base import AgentID, MultiGridEnv
from mosaic_multigrid.core.grid import Grid
from mosaic_multigrid.core.world_object import WorldObj

from Enums import MovementActions
from grid_cell import GridCell
from shared.grid import is_statically_blocked
from shared.movement import handle_simultaneous_movement
from traffic import build_road_network, generate_actions_from_traffic_rules


class SnowPlowEnv(MultiGridEnv):
    """Build traffic terrain and resolve simultaneous agent movement."""

    def __init__(self, road_network=None, **kwargs):
        rows, columns = road_network["size"]

        # Keep a usable empty grid before reset; agent placement happens on reset.
        if road_network is None:
            raise ValueError("road_network must be provided for SnowPlowEnv")

        self.road_network = road_network
        self.snow_layer_count = 0
        super().__init__(width=columns, height=rows, **kwargs)

    def _gen_grid(self, width: int, height: int):
        """Expand configured rectangles and place agents on walkable terrain."""
        self.grid = Grid(width, height)
        if self.road_network is None:
            self.grid.wall_rect(0, 0, width, height)
            for agent in self.agents:
                self.place_agent(agent)
            return

        build_road_network(self.road_network, self.grid)

        generate_actions_from_traffic_rules(self.grid)

        for agent in self.agents:
            self.place_agent(agent)

    def get_observable_grid(self):
        """Grid that the agents may see."""
        return self.grid.state.copy()

    def place_obj(
        self,
        obj: WorldObj | None,
        top: tuple[int, int] | None = None,
        size: tuple[int, int] | None = None,
        reject_fn: Callable[[MultiGridEnv, tuple[int, int]], bool] | None = None,
        max_tries=math.inf,
    ) -> tuple[int, int]:
        """
        Sample an unoccupied walkable spawn cell, preserving its terrain.
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
                raise RecursionError("rejection sampling failed in place_obj")
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

    # TODO: REMOVE THIS?
    # def is_move_allowed(self, grid, agent, current_pos, intended_pos) -> bool:
    #     """
    #     Domain-specific movement hook.
    #     Issue #6
    #     overrides this to reject wrong-direction moves on one-way lanes.
    #     """
    #     return True

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

    def let_it_snow_let_it_snow_let_it_snow(self, seed=None):

        np.random.seed(seed)

        # 0:(1x1) or 1:(3x3) or 2:(5x5)
        clouds_size = np.random.randint(0, 3)

        print("clouds_size:", clouds_size)

        width = self.grid.width
        height = self.grid.height

        row = np.random.randint(0, height)
        column = np.random.randint(0, width)

        center_cell = self.grid.get(column, row)

        snow_intensity = np.random.randint(1, 4)
        print("snow_intensity:", snow_intensity)

        for row_offset in range(-clouds_size, clouds_size + 1):
            for col_offset in range(-clouds_size, clouds_size + 1):
                row = center_cell.row + row_offset
                col = center_cell.column + col_offset

                # Prevent snow to be placed where the agents are positioned.
                if (col, row) in [agent.pos for agent in self.agents]:
                    continue

                # Stop at grid boundaries
                if not (0 <= row < height and 0 <= col < width):
                    continue

                cell: GridCell = self.grid.get(col, row)

                # Cover tiles that are not covered in snow
                if cell.snow_level == 0:
                    self.snow_layer_count += snow_intensity
                    print(self.snow_layer_count)
                    cell.set_snow_level(snow_intensity)
