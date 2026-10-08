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


if __name__ == "__main__":
    main()
