"""Local observations, action masks in observations, per-agent reward split, episode metrics."""

from types import SimpleNamespace

import numpy as np

from Enums import EventType
from shared.events import Event
from shared.metrics import EpisodeMetrics
from shared.observations import FullObservationProvider, LocalObservationProvider
from shared.rewards import EventRewardCalculator


def fake_env(positions, channels=2, height=6, width=8, with_mask=True):
    grid = np.arange(channels * height * width, dtype=np.float32).reshape(channels, height, width) + 1  # all non-zero
    agents = [SimpleNamespace(index=i, state=SimpleNamespace(pos=p, terminated=False)) for i, p in enumerate(positions)]
    env = SimpleNamespace(agents=agents, get_observable_grid=lambda: grid.copy())
    if with_mask:
        env.get_action_mask = lambda agent_id: np.array([True, False, True])
    return env, grid


def test_local_window_is_centred_and_zero_padded_at_the_border():
    env, grid = fake_env([(0, 0)])
    obs = LocalObservationProvider(3).get_observations(env)[0]
    window = obs["grid"]
    assert window.shape == (2, 3, 3)
    assert (window[:, 0, :] == 0).all() and (window[:, :, 0] == 0).all()  # row/column outside the map
    assert (window[:, 1:, 1:] == grid[:, 0:2, 0:2]).all()  # the part inside the map
    assert obs["view_origin"] == (-1, -1)


def test_local_window_in_the_middle_is_a_plain_crop():
    env, grid = fake_env([(4, 3)])
    window = LocalObservationProvider(5).get_observations(env)[0]["grid"]
    assert (window == grid[:, 1:6, 2:7]).all()


def test_local_view_only_reports_plows_inside_the_window():
    env, _ = fake_env([(4, 3), (5, 4), (7, 0)])
    visible = LocalObservationProvider(3).get_observations(env)[0]["agent_positions"]
    assert visible == {0: (4, 3), 1: (5, 4)}  # (7, 0) is too far away


def test_action_mask_is_part_of_the_observation_when_the_env_has_one():
    env, _ = fake_env([(1, 1)])
    for provider in (FullObservationProvider(), LocalObservationProvider(3)):
        assert provider.get_observations(env)[0]["action_mask"].tolist() == [True, False, True]
    env, _ = fake_env([(1, 1)], with_mask=False)
    assert "action_mask" not in FullObservationProvider().get_observations(env)[0]


def test_shared_reward_is_split_equally_inside_a_team():
    calc = EventRewardCalculator({EventType.MOVED: -1.0, EventType.AGENT_COLLISION: -10.0}, {EventType.AGENT_COLLISION})
    result = calc.calculate_rewards(
        [Event(EventType.MOVED, agent_id=0), Event(EventType.AGENT_COLLISION, agent_id=1)],
        {0: None, 1: None, 2: 7},
    )
    per_agent = result.per_agent()
    assert per_agent == {0: -1.0 - 5.0, 1: -5.0, 2: 0.0}
    assert sum(per_agent[a] for a in (0, 1)) == result.team_total(None)


def test_episode_metrics_count_events_and_extras():
    m = EpisodeMetrics()
    m.record_step({0: -1.0, 1: 2.0}, [Event(EventType.MOVED, agent_id=0), Event(EventType.MOVED, agent_id=1)])
    m.record_step({0: -2.0, 1: 0.0}, [Event(EventType.DO_NOTHING, agent_id=0)])
    m.add("snow_cleared", 2)
    m.add("snow_cleared", 1)
    summary = m.summary()
    assert summary["steps"] == 2 and summary["total_reward"] == -1.0
    assert summary["events/moved"] == 2 and summary["events/do_nothing"] == 1 and summary["snow_cleared"] == 3
    m.reset()
    assert m.summary() == {"steps": 0, "total_reward": 0.0}
