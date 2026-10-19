"""Regression tests for live metrics and restartable, self-contained checkpoints."""

import json

import pytest

torch = pytest.importorskip("torch")

from algorithms import PPO, PPOConfig
from main import load_trainer, main
from snowplow.config import SnowConfig, SnowplowConfig
from snowplow.factory import adapter_kwargs, config_to_dict, make_env_fn


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trainer(max_steps=3):
    config = SnowplowConfig(max_steps=max_steps, snow=SnowConfig(initial_clouds=1))
    cfg = PPOConfig(n_envs=2, rollout_len=4, hidden=16, epochs=1, n_minibatches=1)
    result = PPO(make_env_fn("network_1", config), cfg, adapter_kwargs(config))
    result.run_config = {
        "algo": "ppo", "map": "network_1", "snowplow": config_to_dict(config),
        "ppo": vars(cfg), "steps": 16,
    }
    return result


def test_update_history_contains_raw_environment_metrics():
    agent = trainer()
    callbacks = []
    history = agent.learn(9, log_every=0, callback=lambda stats: callbacks.append(stats.copy()))
    assert agent.env_steps == 16  # round up, rather than silently undershooting the budget
    assert len(history) == len(callbacks) == 2
    metrics = history[-1]["episode_metrics"]
    assert history[-1]["total_reward"] == history[-1]["mean_return"] == metrics["total_reward"]
    assert {"snow_cleared", "snow_remaining", "fraction_clean", "cooperation_rate", "plowing_steps"} <= metrics.keys()
    assert history[-1]["episodes_completed"] > 0
    json.dumps(history, allow_nan=False)


def test_updates_before_episode_completion_use_null_not_nan():
    history = trainer(max_steps=100).learn(8, log_every=1)
    assert history[0]["mean_return"] is None
    assert history[0]["total_reward"] is None
    assert history[0]["episode_metrics"] == {}
    json.dumps(history, allow_nan=False)


def test_standalone_checkpoint_restores_optimizer_history_and_progress(tmp_path):
    original = trainer()
    original.learn(8, log_every=0)
    checkpoint = tmp_path / "standalone.pt"
    original.save(str(checkpoint))
    restored, name, config = load_trainer(checkpoint, resume=True)
    assert name == "network_1" and config.n_plows == 1
    assert restored.env_steps == 8 and restored.updates == 1
    assert restored.history == original.history
    assert restored.opt.state_dict()["param_groups"] == original.opt.state_dict()["param_groups"]
    for key, state in original.opt.state_dict()["state"].items():
        for field, value in state.items():
            assert torch.equal(value, restored.opt.state_dict()["state"][key][field])
    assert restored.first.eq(1).all()  # new episodes, no stale recurrent rollout state
    history = restored.learn(16, log_every=0)
    assert [row["update"] for row in history] == [1, 2]
    assert [row["env_steps"] for row in history] == [8, 16]
    assert not checkpoint.with_suffix(".pt.tmp").exists()


def test_legacy_weights_still_load_but_cannot_resume(tmp_path):
    original = trainer()
    path = tmp_path / "legacy.pt"
    torch.save({"model": original.model.state_dict(), "config": vars(original.cfg)}, path)
    original.load(str(path))
    with pytest.raises(ValueError, match="weights only"):
        original.load(str(path), resume=True)


def test_cli_writes_history_during_training_and_can_resume(tmp_path, monkeypatch):
    original_learn = PPO.learn
    seen_updates = []

    def checked_learn(self, *args, **kwargs):
        callback = kwargs["callback"]

        def inspect(stats):
            callback(stats)
            history = json.loads((tmp_path / "history.json").read_text())
            assert history[-1]["update"] == stats["update"]
            seen_updates.append(stats["update"])

        kwargs["callback"] = inspect
        return original_learn(self, *args, **kwargs)

    monkeypatch.setattr(PPO, "learn", checked_learn)
    main([
        "train", "--experiment", "exp1", "--map", "network_1", "--steps", "8",
        "--n-envs", "2", "--rollout-len", "4", "--hidden", "16", "--max-steps", "3",
        "--checkpoint-every", "8", "--eval-episodes", "1", "--out", str(tmp_path),
    ])
    assert (tmp_path / "checkpoints/model_step_8.pt").exists()
    main(["train", "--resume", str(tmp_path), "--steps", "16", "--eval-episodes", "1"])
    assert seen_updates == [1, 2]
    assert (tmp_path / "checkpoints/model_step_16.pt").exists()
    assert json.loads((tmp_path / "config.json").read_text())["steps"] == 16


def test_history_survives_interruption_after_an_update(tmp_path, monkeypatch):
    original_learn = PPO.learn

    def interrupted(self, *args, **kwargs):
        callback = kwargs["callback"]

        def stop(stats):
            callback(stats)
            raise KeyboardInterrupt

        kwargs["callback"] = stop
        return original_learn(self, *args, **kwargs)

    monkeypatch.setattr(PPO, "learn", interrupted)
    with pytest.raises(KeyboardInterrupt):
        main([
            "train", "--steps", "16", "--map", "network_1", "--n-envs", "1",
            "--rollout-len", "4", "--hidden", "16", "--max-steps", "3",
            "--checkpoint-every", "4", "--out", str(tmp_path),
        ])
    assert len(json.loads((tmp_path / "history.json").read_text())) == 1
    load_trainer(tmp_path, "model_step_4.pt")
