"""Reward values and calculator for the snowplow environment.

Per step, a plow receives:
    -1   it moved                       (time costs)
     0   it spent the step plowing      (SnowplowEvent.PLOWING, staying on a snowy cell)
    +1   per snow LAYER of a cell it cleaned. A cell cleaned by k plows together pays
         layers / k to each of them, so the team earns the same for a cell no matter
         who cleaned it: working together only pays off because it is FASTER
         (more cells cleaned before the episode ends), never by double counting.
    +2   a secondary street became completely snow-free
    +4   a main street became completely snow-free
Shared by all plows (split equally between them, see RewardResult.per_agent):
    +100 every road is clean (episode success)
"""

from Enums import EventType, SnowplowEvent
from shared.rewards import EventRewardCalculator

# Rewards and penalties for snowplow events
SNOWPLOW_REWARDS = {
    # EventType.MOVED: -1,
    # EventType.DO_NOTHING: -2,
    # EventType.INVALID_MOVE: -5,  # chose a wall / wrong-way lane (impossible with action masking)
    # EventType.AGENT_COLLISION: -10,
    SnowplowEvent.PARTIALLY_CLEARED: 1,  # x reward_scale = layers removed / plows that did the work
    SnowplowEvent.PLOWING: 0,
    SnowplowEvent.SECONDARY_STREET_CLEARED: 2,
    SnowplowEvent.MAIN_STREET_CLEARED: 4,
    # SnowplowEvent.ALL_CLEAN: 100,
}


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
