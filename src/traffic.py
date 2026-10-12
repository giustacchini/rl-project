"""Display the configured traffic grid until the window is closed."""

import pygame

from snow_plow_env import SnowPlowEnv
from Traffic_configs import road_network_1


def main():
    # env = SnowPlowEnv(agents=2, width=10, height=10, max_steps=50, render_mode="human")
    # env.reset()

    # for step in range(env.max_steps):
    #     actions = {agent.index: agent.sample_action() for agent in env.agents}
    #     print(f"Step {step}: Actions: {actions}")
    #     rewards = env.handle_actions(actions)
    #     print(f"Rewards: {rewards}")
    #     print(f"Agent positions: {[agent.state.pos for agent in env.agents]}")
    #     print(f"Movement results: {env.last_movement_results}")

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


if __name__ == "__main__":
    main()
