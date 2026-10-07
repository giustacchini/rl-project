from Enums import SnowPlowActions
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


if __name__ == "__main__":
    main()
