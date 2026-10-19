"""Rewards for the current fixed-horizon snowplow task.

Only CELL_CLEARED pays: +1/+2/+3 for a cleaned level-1/2/3 cell, shared in
proportion to contributed work. Movement, waiting, plowing, collisions, street
completion, and a temporarily clean map have no additional reward.
"""

from Enums import SnowplowEvent
from shared.rewards import EventRewardCalculator

# The event's reward_scale supplies the snow-level value and contributor share.
SNOWPLOW_REWARDS = {SnowplowEvent.CELL_CLEARED: 1}

# Reward for cleaning a cell, by the snow level the cell had when it became clean.
SNOW_LEVEL_REWARDS = {1: 1.0, 2: 2.0, 3: 3.0}


def snow_level_reward(level: int) -> float:
    """Team reward for completely cleaning a cell of `level` (0 for a clean cell).

    Levels above the table (if max_snow is raised) pay the reward of the highest listed level.
    """
    if level <= 0:
        return 0.0
    return SNOW_LEVEL_REWARDS.get(int(level), SNOW_LEVEL_REWARDS[max(SNOW_LEVEL_REWARDS)])


# Clearing rewards are already apportioned to individual contributors.
SNOWPLOW_SHARED_EVENTS: set[SnowplowEvent] = set()


def make_snowplow_reward_calculator() -> EventRewardCalculator:
    return EventRewardCalculator(
        SNOWPLOW_REWARDS,
        shared_events=SNOWPLOW_SHARED_EVENTS,
    )
