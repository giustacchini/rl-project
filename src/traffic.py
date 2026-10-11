from mosaic_multigrid.core.world_object import Floor, Goal, Wall, WorldObj

from Enums import GridCellTypes, SnowPlowActions
from snow_plow_env import SnowPlowEnv


def main():
    env = SnowPlowEnv(agents=2, width=10, height=10, max_steps=50, render_mode="human")
    env.reset()

    for step in range(env.max_steps):
        actions = {agent.index: agent.sample_action() for agent in env.agents}
        print(f"Step {step}: Actions: {actions}")
        rewards = env.handle_actions(actions)
        print(f"Rewards: {rewards}")
        print(f"Agent positions: {[agent.state.pos for agent in env.agents]}")
        print(f"Movement results: {env.last_movement_results}")


class GridCell(WorldObj):
    def __init__(self, pos: tuple[int, int], cell_type: GridCellTypes, snow_level: int = 1):
        super().__init__(pos)
        self.cell_type = cell_type

        if not isinstance(snow_level, int) or snow_level < 0:
            raise ValueError("snow_level must be a non-negative integer")
        elif snow_level > 3:
            raise ValueError("snow_level must be between 0 and 3")
        else:
            self.snow_level = snow_level

        if cell_type in GridCellTypes.OBSTACLE:
            self.blocked = True
            self.color = (30, 173, 2)  # Green for obstacle
        elif cell_type in GridCellTypes.ROAD:
            self.blocked = False
            self.color = (128, 128, 128)  # Gray for road
        elif cell_type in GridCellTypes.INTERSECTION:
            self.blocked = False
            self.color = (100, 100, 100)  # Darker gray than regular road to specify an intersection
        elif cell_type in GridCellTypes.HIGHWAY:
            self.blocked = False
            self.color = (128, 128, 128)  # Gray for road

        if snow_level > 0 and cell_type is not GridCellTypes.OBSTACLE:
            self.color = (255, 255, 255)  # White for snow-covered cells


if __name__ == "__main__":
    main()
