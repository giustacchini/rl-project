<<<<<<< HEAD
from mosaic_multigrid.core import Color

from Enums import GridCellTypes, MovementActions
from grid_cell import GridCell

Color.add_color("snow_level3", (255, 255, 255))  # White for snow-covered cells
Color.add_color("snow_level2", (230, 238, 252))  # White for snow-covered cells
Color.add_color("snow_level1", (199, 220, 255))  # White for snow-covered cells


def build_road_network(road_network, grid):
    for category, cell_type in {grid_type.value: grid_type for grid_type in GridCellTypes}.items():
        for rectangle in road_network[category]:
            row, column = rectangle["position"]
            row_count, column_count = rectangle["size"]
            for x in range(column, column + column_count):
                for y in range(row, row + row_count):
                    grid_cell = GridCell(row=y, column=x, cell_type=cell_type)
                    grid.set(x, y, grid_cell)


def generate_actions_from_traffic_rules(grid):
    """
    Setup traffic rules (Agent actions) based on the road network configuration (environment).
    This includes neighborhood streets, intersections and highways.
    """
    cell: GridCell = None
    for cell in grid.grid:
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
                adj_cell: GridCell = grid.get(*value)  # Unpack tuple
                adjacent_cells[dir] = adj_cell.cell_type
            except (TypeError, IndexError):
                # Out of bounds
                adjacent_cells[dir] = None

        movement_actions = []
        if cell.cell_type == GridCellTypes.HIGHWAY:
            movement_actions = highway_traffic_rules(adjacent_cells)
        elif cell.cell_type == GridCellTypes.INTERSECTION:
            # TODO: Prevent Agent from going back and fourth inside of an intersection. Maybe store last visited grid_type.
            movement_actions = intersection_traffic_rules(adjacent_cells)
        elif cell.cell_type == GridCellTypes.ROAD:
            movement_actions = road_traffic_rules(adjacent_cells)

        # Add default action of DO_NOTHING
        movement_actions.append(MovementActions.DO_NOTHING)

        # Add new actions to grid cell
        cell.set_movement_action(movement_actions)
=======
"""Display the configured traffic grid until the window is closed."""

import pygame

from snow_plow_env import SnowPlowEnv
from Traffic_configs import road_network_1


def main():
    rows, columns = road_network_1["size"]
    env = SnowPlowEnv(
        agents=2,
        width=columns,
        height=rows,
        max_steps=500,
        render_mode="human",
        road_network=road_network_1,
    )
    env.highlight = False
    try:
        env.reset(seed=42)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if running:
                env.render()
    finally:
        env.close()
>>>>>>> 0b6c6e5 (evol(visualization): applied modifications in order to see the environment in human render mode)


def highway_traffic_rules(adjacent_cells: dict):
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


def intersection_traffic_rules(adjacent_cells: dict):
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


def road_traffic_rules(adjacent_cells: dict):
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
