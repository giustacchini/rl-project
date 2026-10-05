# python
# snow_plow_env.py

import enum

from mosaic_multigrid.base import MultiGridEnv, AgentID
from mosaic_multigrid.core.grid import Grid

from shared.movement import handle_simultaneous_movement
from shared.entities import Obstacle


class SnowPlowAction(enum.IntEnum):
    do_nothing = 0
    up = enum.auto()
    down = enum.auto()
    right = enum.auto()
    left = enum.auto()


# Movement is represented on the grid using (x, y) coordinates
SNOWPLOW_ACTION_DELTAS = {
    SnowPlowAction.do_nothing: (0, 0),
    SnowPlowAction.up: (0, -1),
    SnowPlowAction.down: (0, 1),
    SnowPlowAction.right: (1, 0),
    SnowPlowAction.left: (-1, 0),
}


class SnowPlowEnv(MultiGridEnv):
    """
    `is_move_allowed` is the extension point issue #6 (road network and
    traffic rules) will fill in, RIGHT NOW  it currently allows every move that
    passes the shared grid/bounds checks
    """

    def _gen_grid(self, width: int, height: int):
        # Required by MultiGridEnv.reset(): must set self.grid and give
        # every agent a valid (non-negative) position and direction
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for agent in self.agents:
            self.place_agent(agent)

    def is_move_allowed(self, grid, agent, current_pos, intended_pos) -> bool:
        """
        Domain-specific movement hook.
        Issue #6
        overrides this to reject wrong-direction moves on one-way lanes.
        """
        return True

    def handle_actions(self, actions: dict[AgentID, SnowPlowAction]):
        movement_results = handle_simultaneous_movement(
            agents=self.agents,
            grid=self.grid,
            actions=actions,
            action_deltas=SNOWPLOW_ACTION_DELTAS,
            width=self.width,
            height=self.height,
            invalid_action_value=SnowPlowAction.do_nothing,
            is_move_allowed=self.is_move_allowed,
        )

        self.last_movement_results = movement_results

        # Placeholder rewards -- actual SnowPlow rewards are for future issues
        rewards = {agent.index: 0 for agent in self.agents}
        return rewards
