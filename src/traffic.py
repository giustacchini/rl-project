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

        if cell_type in GridCellTypes.ROAD:
            self.blocked = False
            self.color = (128, 128, 128)  # Gray for road
        elif cell_type in GridCellTypes.INTERSECTION:
            self.blocked = False
            self.color = (128, 128, 128)  # Gray for road
        elif cell_type in GridCellTypes.HIGHWAY:
            self.blocked = False
            self.color = (128, 128, 128)  # Gray for road
        elif cell_type in GridCellTypes.OBSTACLE:
            self.blocked = True
            self.color = (128, 128, 128)  # Gray for road


if __name__ == "__main__":
    main()
