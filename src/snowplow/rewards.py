"""Reward values and calculator for the snowplow environment.

Per step, a plow receives:
    -1      it moved                       (time costs)
     0      it spent the step plowing      (SnowplowEvent.PLOWING, staying on a snowy cell)
    +1/2/3  it cleaned a cell of snow level 1/2/3 (SnowplowEvent.CELL_CLEARED, see
            SNOW_LEVEL_REWARDS). A cell cleaned by several plows together is split between
            them in proportion to the work each one did, so the team earns the same for a
            cell no matter who cleaned it: working together only pays off because it is
            FASTER (more cells cleaned before the episode ends), never by double counting.

Street-completion events (SECONDARY_STREET_CLEARED / MAIN_STREET_CLEARED) are still emitted
for the metrics, but they no longer give a reward.
"""

from Enums import EventType, SnowplowEvent
from shared.rewards import EventRewardCalculator

# Rewards and penalties for snowplow events
SNOWPLOW_REWARDS = {
    # EventType.MOVED: -1,
    # EventType.DO_NOTHING: -2,
    # EventType.INVALID_MOVE: -5,  # chose a wall / wrong-way lane (impossible with action masking)
    # EventType.AGENT_COLLISION: -10,
    # The reward of a cleaned cell comes from SNOW_LEVEL_REWARDS (via the event's reward_scale),
    # so this base value stays 1.
    SnowplowEvent.CELL_CLEARED: 1,
    # SnowplowEvent.PLOWING: 1,
    # Removed: the old per-cell and street-completion rewards.
    # SnowplowEvent.PARTIALLY_CLEARED: 1,
    # SnowplowEvent.SECONDARY_STREET_CLEARED: 2,
    # SnowplowEvent.MAIN_STREET_CLEARED: 4,
    # SnowplowEvent.ALL_CLEAN: 100,
}

# Reward for cleaning a cell, by the snow level the cell had when it became clean.
SNOW_LEVEL_REWARDS = {1: 1.0, 2: 2.0, 3: 3.0}


def snow_level_reward(level: int) -> float:
    """Team reward for completely cleaning a cell of `level` (0 for a clean cell).

    Levels above the table (if max_snow is raised) pay the reward of the highest listed level.
    """
    if level <= 0:
        return 0.0
    return SNOW_LEVEL_REWARDS.get(int(level), SNOW_LEVEL_REWARDS[max(SNOW_LEVEL_REWARDS)])


# Shared rewards/penalties: they belong to all plows, not to one plow.
SNOWPLOW_SHARED_EVENTS = {
    # EventType.AGENT_COLLISION,  # collision penalty currently disabled
    SnowplowEvent.ALL_CLEAN,
}


def make_snowplow_reward_calculator() -> EventRewardCalculator:
    return EventRewardCalculator(
        SNOWPLOW_REWARDS,
        shared_events=SNOWPLOW_SHARED_EVENTS,
    )
