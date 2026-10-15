"""A few PPO/MAPPO updates on the small map. Skipped automatically when PyTorch is not installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from algorithms import MAPPO, PPO, PPOConfig  # noqa: E402
from snowplow.config import SnowplowConfig  # noqa: E402
from snowplow.factory import adapter_kwargs, make_env_fn  # noqa: E402


def make_trainer(cls, n_plows, observation="full", recurrent=False):
    config = SnowplowConfig(n_plows=n_plows, observation=observation, max_steps=40)
    cfg = PPOConfig(n_envs=2, rollout_len=16, n_minibatches=2, epochs=2, hidden=32, recurrent=recurrent)
    return cls(make_env_fn("small", config), cfg, adapter_kwargs(config))


@pytest.mark.parametrize("cls,n_plows,observation,recurrent", [
    (PPO, 1, "full", False),
    (PPO, 1, "local", True),
    (MAPPO, 2, "full", False),
    (MAPPO, 2, "local", True),
])
def test_training_runs_and_losses_are_finite(cls, n_plows, observation, recurrent):
    trainer = make_trainer(cls, n_plows, observation, recurrent)
    history = trainer.learn(2 * 16 * 2, log_every=0)
    assert len(history) == 2
    for record in history:
        assert all(np.isfinite(record[k]) for k in ("pi_loss", "v_loss", "entropy"))


def test_the_policy_never_samples_an_illegal_action():
    trainer = make_trainer(PPO, 1)
    data = trainer._collect()
    chosen_is_legal = data["mask"].gather(-1, data["act"].unsqueeze(-1)).squeeze(-1)
    assert chosen_is_legal.all()
    assert (trainer.evaluate(2)["mean_length"] > 0)


def test_save_and_load_roundtrip(tmp_path):
    trainer = make_trainer(MAPPO, 2)
    trainer.save(str(tmp_path / "m.pt"))
    other = make_trainer(MAPPO, 2)
    other.load(str(tmp_path / "m.pt"))
    for p, q in zip(trainer.model.parameters(), other.model.parameters()):
        assert torch.equal(p, q)
