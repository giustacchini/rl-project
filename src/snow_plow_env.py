"""Snow-plow environment with a configured traffic grid and shared movement."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from mosaic_multigrid.base import AgentID, MultiGridEnv
from mosaic_multigrid.core.grid import Grid
from mosaic_multigrid.core.world_object import WorldObj

from Enums import EventType, GridCellTypes, MovementActions
from grid_cell import GridCell
from shared.grid import in_bounds, is_statically_blocked
from shared.events import EventLog
from shared.movement import handle_simultaneous_movement
from snowplow.rewards import make_snowplow_reward_calculator


class SnowPlowEnv(MultiGridEnv):
    """Build traffic terrain and resolve simultaneous agent movement."""

    def __init__(self, agents, max_steps, render_mode=None, road_network=None):
        if road_network is None:
            raise ValueError("road_network must be provided for SnowPlowEnv")

        rows, columns = road_network["size"]

        # Keep a usable empty grid before reset; agent placement happens on reset.
        self.road_network = road_network
        super().__init__(
            agents=agents,
            width=columns,
            height=rows,
            max_steps=max_steps,
            render_mode=render_mode,
        )
        self.event_log = EventLog()

    def _gen_grid(self, width: int, height: int):
        """Expand configured rectangles and place agents on walkable terrain."""
        self.grid = Grid(width, height)
        if self.road_network is None:
            self.grid.wall_rect(0, 0, width, height)
            for agent in self.agents:
                self.place_agent(agent)
            return

        for category, cell_type in {grid_type.value: grid_type for grid_type in GridCellTypes}.items():
            for rectangle in self.road_network[category]:
                row, column = rectangle["position"]
                row_count, column_count = rectangle["size"]
                for x in range(column, column + column_count):
                    for y in range(row, row + row_count):
                        grid_cell = GridCell(row=y, column=x, cell_type=cell_type)
                        self.grid.set(x, y, grid_cell)

        self.generate_actions_from_traffic_rules()

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

    def is_move_allowed(self, grid, current_pos, intended_pos) -> bool:
        """Return whether the current traffic cell permits this movement."""
        current_cell = grid.get(*current_pos)
        if current_cell is None:
            return False

        delta = tuple(
            intended_coordinate - current_coordinate
            for current_coordinate, intended_coordinate in zip(current_pos, intended_pos)
        )
        try:
            action = MovementActions(delta)
        except ValueError:
            return False

        return action in current_cell.movement_actions

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

        self.event_log.clear()
        collision_statuses = {
            "blocked_same_destination",
            "blocked_swap",
            "blocked_occupied_cell",
        }
        for agent_id, result in movement_results.items():
            status = result["status"]
            if status == "moved":
                event_type = EventType.MOVED
            elif status == "waited":
                event_type = EventType.DO_NOTHING
            elif status in collision_statuses:
                event_type = EventType.AGENT_COLLISION
            else:
                event_type = EventType.INVALID_MOVE

            self.event_log.emit(
                event_type,
                agent_id=agent_id,
                status=status,
                current_position=tuple(result["current_position"]),
                resolved_position=tuple(result["resolved_position"]),
            )

        team_of_agent = {agent.index: None for agent in self.agents}
        reward_result = make_snowplow_reward_calculator().calculate_rewards(
            self.event_log.events(),
            team_of_agent,
        )
        return reward_result.individual

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
        """
        Setup traffic rules (Agent actions) based on the road network configuration (environment).
        This includes neighborhood streets, intersections and highways.
        """
        for cell in self.grid.grid:
            if cell is None:
                raise ValueError("Grid cell is None. Check road network configuration.")
            elif cell.cell_type == GridCellTypes.OBSTACLE:
                continue

            # (COLUMN, ROW) / (X, Y)
            up = (cell.column, cell.row - 1)
            right = (cell.column + 1, cell.row)
            left = (cell.column - 1, cell.row)
            down = (cell.column, cell.row + 1)

            adjacent_cells = {"up": up, "right": right, "left": left, "down": down}

            for dir, value in adjacent_cells.items():
                try:
                    adj_cell = self.grid.get(*value)  # Unpack tuple
                    adjacent_cells[dir] = adj_cell.cell_type
                except (TypeError, IndexError):
                    # Out of bounds
                    adjacent_cells[dir] = None

            movement_actions = []
            if cell.cell_type == GridCellTypes.HIGHWAY:
                movement_actions = self.highway_traffic_rules(adjacent_cells)
            elif cell.cell_type == GridCellTypes.INTERSECTION:
                # TODO: Prevent Agent from going back and fourth inside of an intersection. Maybe store last visited grid_type.
                movement_actions = self.intersection_traffic_rules(adjacent_cells)
            elif cell.cell_type == GridCellTypes.ROAD:
                movement_actions = self.road_traffic_rules(adjacent_cells)

            # Add default action of DO_NOTHING
            movement_actions.append(MovementActions.DO_NOTHING)

            # Add new actions to grid cell
            self.grid.set(
                cell.column,
                cell.row,
                GridCell(row=cell.row, column=cell.column, cell_type=cell.cell_type, movement_actions=movement_actions),
            )

    def highway_traffic_rules(self, adjacent_cells: dict):
        allowed_action = None

        for dir, cell in adjacent_cells.items():
            if cell is not None and cell == GridCellTypes.OBSTACLE:
                if dir == "up":
                    allowed_action = MovementActions.LEFT
                elif dir == "right":
                    allowed_action = MovementActions.UP
                elif dir == "left":
                    allowed_action = MovementActions.DOWN
                elif dir == "down":
                    allowed_action = MovementActions.RIGHT

                if allowed_action is not None:
                    break

        return [allowed_action]

    def intersection_traffic_rules(self, adjacent_cells: dict):
        movement_actions = [MovementActions.UP, MovementActions.DOWN, MovementActions.LEFT, MovementActions.RIGHT]

        filter_movement_action = None

        for dir, cell in adjacent_cells.items():
            # Remove blocked movements
            if cell is None:
                if dir == "up":
                    filter_movement_action = MovementActions.UP
                elif dir == "right":
                    filter_movement_action = MovementActions.RIGHT
                elif dir == "left":
                    filter_movement_action = MovementActions.LEFT
                elif dir == "down":
                    filter_movement_action = MovementActions.DOWN

            # Prevent agents from going the opposite direction of the current lane.
            elif cell == GridCellTypes.INTERSECTION:
                if dir == "up":
                    filter_movement_action = MovementActions.LEFT
                elif dir == "right":
                    filter_movement_action = MovementActions.UP
                elif dir == "left":
                    filter_movement_action = MovementActions.DOWN
                elif dir == "down":
                    filter_movement_action = MovementActions.RIGHT

            # FIlter invalid actions
            movement_actions = list(filter(lambda x: x != filter_movement_action, movement_actions))

        return movement_actions

    def road_traffic_rules(self, adjacent_cells: dict):
        movement_actions = []
        for dir, cell in adjacent_cells.items():
            # Add movement actions if cell exists is not an obstacle.
            if cell is not None and not cell == GridCellTypes.OBSTACLE:
                if dir == "up":
                    movement_actions.append(MovementActions.UP)
                elif dir == "right":
                    movement_actions.append(MovementActions.RIGHT)
                elif dir == "left":
                    movement_actions.append(MovementActions.LEFT)
                elif dir == "down":
                    movement_actions.append(MovementActions.DOWN)
                # movement_actions = list(filter(lambda x: x != filter_movement_action, movement_actions))
        return movement_actions
