"""Snow-plow environment with a configured traffic grid and shared movement."""

from __future__ import annotations
import math
from typing import Callable

import numpy as np
from mosaic_multigrid.base import AgentID, MultiGridEnv
from mosaic_multigrid.core.grid import Grid
from mosaic_multigrid.core.world_object import WorldObj

from Enums import GridCellTypes, MovementActions
from grid_cell import GridCell
from shared.grid import in_bounds, is_statically_blocked
from shared.movement import handle_simultaneous_movement

class SnowPlowEnv(MultiGridEnv):
    """Build traffic terrain and resolve simultaneous agent movement."""

    def __init__(self, agents, max_steps, render_mode=None, road_network=None):
        rows, columns = road_network["size"]

        # if road_network is not None and road_network["size"] != (height, width):
        #     raise ValueError(
        #         "Road network size (rows, columns) must match (height, width)."
        #     )

        # Keep a usable empty grid before reset; agent placement happens on reset.
        if road_network is None:
            raise ValueError("road_network must be provided for SnowPlowEnv")
            # self.grid.wall_rect(0, 0, columns, rows)

        self.road_network = road_network
        super().__init__(
            agents=agents,
            width=columns,
            height=rows,
            max_steps=max_steps,
            render_mode=render_mode,
        )

    def _gen_grid(self, width: int, height: int):
        """Expand configured rectangles and place agents on walkable terrain."""
        self.grid = Grid(width, height)
        if self.road_network is None:
            self.grid.wall_rect(0, 0, width, height)
            for agent in self.agents:
                self.place_agent(agent)
            return

        for category, cell_type in {grid_type.value: grid_type for grid_type in GridCellTypes}.items():
            print(category, cell_type)
            for rectangle in self.road_network[category]:
                print(self.road_network[category])
                row, column = rectangle["position"]
                row_count, column_count = rectangle["size"]
                for x in range(column, column + column_count):
                    for y in range(row, row + row_count):
                        grid_cell = GridCell(row=y, column=x, cell_type=cell_type)
                        self.grid.set(x, y, grid_cell)
                        print(grid_cell.color)

        self.generate_actions_from_traffic_rules()

        for agent in self.agents:
            self.place_agent(agent)

    # def is_out_of_bounds(curr_cell: np.array[int, int], action: MovementActions, width, height):
    #     return (
    #         ((curr_cell.pos + action.value) < width)
    #         or ((curr_cell.pos + action.value) < height)
    #         or ((curr_cell.pos + action.value) > width)
    #         or ((curr_cell.pos + action.value) > height)
    #     )

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

    def let_it_snow_let_it_snow_let_it_snow(self):

        # 1-3 clouds
        num_clouds = np.random.randint(1, 3)
        # 50/50 -> 3x3 or 5x5
        clouds_size = np.random.randint(1, 2)

        width = self.grid.width
        height = self.grid.height

        for _ in range(num_clouds):

            row = np.random.randint(0, height)
            column = np.random.randint(0, width)

            center_cell = self.grid.get(column, row)

            for row_offset in range(-clouds_size, clouds_size + 1):
                for col_offset in range(-clouds_size, clouds_size + 1):
                    row = center_cell.row + row_offset
                    col = center_cell.column + col_offset

                    # Stop at grid boundaries
                    if not (0 <= row < self.grid.height and 0 <= col < self.grid.width):
                        continue

                    cell = self.grid.get(row, col)

                    cell.set_snow_level()
                    # Do something with cell

    def generate_actions_from_traffic_rules(self):
        # TODO: Setup traffic rules (Agent actions) based on the road network configuration. This can include one-way streets, intersections, etc.
        for cell in self.grid.grid:
            if cell is None:
                raise ValueError("Grid cell is None. Check road network configuration.")

            # (COLUMN, ROW) / (X, Y)
            up = (cell.column, cell.row - 1)
            right = (cell.column + 1, cell.row)
            left = (cell.column - 1, cell.row)
            down = (cell.column, cell.row + 1)

            up_allowed = in_bounds(self.width, self.height, up)
            right_allowed = in_bounds(self.width, self.height, right)
            left_allowed = in_bounds(self.width, self.height, left)
            down_allowed = in_bounds(self.width, self.height, down)
            print(up)
            above_cell = self.grid.get(*up)  # Unpack tuple
            right_cell = self.grid.get(*right)
            left_cell = self.grid.get(*left)
            lower_cell = self.grid.get(*down)

            adjacent_cells = (above_cell, right_cell, left_cell, lower_cell)
            allowed_actions = (up_allowed, right_allowed, left_allowed, down_allowed)

            # Default to all directions except DO_NOTHING
            if cell.cell_type == GridCellTypes.HIGHWAY:
                movement_actions = []
                # Highway traffic rules
                self.highway_traffic_rules(movement_actions, adjacent_cells, allowed_actions)
            else:
                movement_actions = [MovementActions.UP, MovementActions.DOWN, MovementActions.LEFT, MovementActions.RIGHT]
                if cell.cell_type == GridCellTypes.INTERSECTION:
                    # Intersection traffic rules
                    # TODO: Prevent Agent from going back and fourth inside of an intersection. Maybe store last visited grid_type.
                    self.intersection_traffic_rules(movement_actions, adjacent_cells, allowed_actions)
                if cell.cell_type == GridCellTypes.ROAD:
                    # Road traffic rules
                    self.road_traffic_rules(movement_actions, adjacent_cells, allowed_actions)

            self.grid.set(
                cell.column,
                cell.row,
                GridCell(row=cell.row, column=cell.column, cell_type=GridCellTypes.OBSTACLE, movement_actions=movement_actions),
            )

    def highway_traffic_rules(self, movement_actions: list[MovementActions], adjacent_cells, allowed_actions):
        above_cell, right_cell, left_cell, lower_cell = adjacent_cells
        up_allowed, right_allowed, left_allowed, down_allowed = allowed_actions

        if above_cell == GridCellTypes.OBSTACLE and up_allowed:
            movement_actions.append(MovementActions.LEFT)
        elif right_cell == GridCellTypes.OBSTACLE and right_allowed:
            movement_actions.append(MovementActions.UP)
        elif left_cell == GridCellTypes.OBSTACLE and left_allowed:
            movement_actions.append(MovementActions.DOWN)
        elif lower_cell == GridCellTypes.OBSTACLE and down_allowed:
            movement_actions.append(MovementActions.RIGHT)

    def intersection_traffic_rules(self, movement_actions: list[MovementActions], adjacent_cells, allowed_actions):
        above_cell, right_cell, left_cell, lower_cell = adjacent_cells
        up_allowed, right_allowed, left_allowed, down_allowed = allowed_actions

        filter_movement_action = None
        if above_cell == GridCellTypes.INTERSECTION and up_allowed:
            filter_movement_action = MovementActions.LEFT
        elif right_cell == GridCellTypes.INTERSECTION and right_allowed:
            filter_movement_action = MovementActions.UP
        elif left_cell == GridCellTypes.INTERSECTION and left_allowed:
            filter_movement_action = MovementActions.DOWN
        elif lower_cell == GridCellTypes.INTERSECTION and down_allowed:
            filter_movement_action = MovementActions.RIGHT
        movement_actions = filter(lambda x: x != filter_movement_action, movement_actions)

    def road_traffic_rules(self, movement_actions: list[MovementActions], adjacent_cells, allowed_actions):
        above_cell, right_cell, left_cell, lower_cell = adjacent_cells
        up_allowed, right_allowed, left_allowed, down_allowed = allowed_actions

        filter_movement_action = []
        # Avoid Obstacle and out of bounds
        if above_cell == GridCellTypes.OBSTACLE and up_allowed:  # HACK: DOES THIS WORK / IS IT NEEDED?
            filter_movement_action = filter_movement_action.append(MovementActions.UP)
        if right_cell == GridCellTypes.OBSTACLE and right_allowed:
            filter_movement_action = filter_movement_action.append(MovementActions.RIGHT)
        if left_cell == GridCellTypes.OBSTACLE and left_allowed:
            filter_movement_action = filter_movement_action.append(MovementActions.LEFT)
        if lower_cell == GridCellTypes.OBSTACLE and down_allowed:
            filter_movement_action = filter_movement_action.append(MovementActions.DOWN)
        movement_actions = filter(lambda x: x != filter_movement_action, movement_actions)
