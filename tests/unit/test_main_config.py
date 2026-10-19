from argparse import Namespace

import pytest

from main import create_run_dir, resolve_environment


def make_args(**overrides):
    values = {
        "experiment": None,
        "map": None,
        "plows": None,
        "obs": None,
        "view": None,
        "max_steps": None,
        "spawn_prob": None,
        "initial_clouds": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_resolver_uses_config_defaults_when_options_are_omitted():
    map_name, config = resolve_environment(make_args())

    assert map_name == "network_3"
    assert config.max_steps == 500
    assert config.view_size == 5


def test_explicit_map_overrides_experiment_map():
    map_name, config = resolve_environment(make_args(experiment="exp1", map="network_1"))

    assert map_name == "network_1"
    assert config.n_plows == 1
    assert config.observation == "full"


def test_experiment_map_is_used_when_map_is_omitted():
    map_name, _ = resolve_environment(make_args(experiment="exp1"))

    assert map_name == "network_1"


def test_default_run_directories_are_unique(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = Namespace(out=None, experiment="exp1", seed=0)

    first = create_run_dir(args, "network_3", "ppo")
    second = create_run_dir(args, "network_3", "ppo")

    assert first != second
    assert first.name.startswith("exp1_network_3_ppo_seed0_")
    assert second.name.startswith("exp1_network_3_ppo_seed0_")


def test_checkpoint_callback_saves_every_n_steps(tmp_path):
    """--checkpoint-every: one file per multiple of N, named after the env step reached."""
    from main import make_checkpoint_callback

    saved = []

    class FakeTrainer:
        def save(self, path):
            saved.append(path)

    callback = make_checkpoint_callback(FakeTrainer(), tmp_path, every=1000)
    for steps in (512, 1024, 1536, 2048, 4096):  # 4096 jumps over 3000: only one file for it
        callback({"env_steps": steps})
    assert [p.rsplit("_", 1)[-1] for p in saved] == ["1024.pt", "2048.pt", "4096.pt"]
    assert make_checkpoint_callback(FakeTrainer(), tmp_path, every=None) is None


@pytest.mark.parametrize("experiment,n_plows,observation,map_name", [
    ("exp1", 1, "full", "network_1"), ("exp2", 1, "local", "network_1"),
    ("exp3", 5, "full", "network_3"), ("exp4", 5, "local", "network_3"),
])
def test_all_presets_resolve_consistently(experiment, n_plows, observation, map_name):
    actual_map, config = resolve_environment(make_args(experiment=experiment))
    assert (actual_map, config.n_plows, config.observation) == (map_name, n_plows, observation)
    assert config.max_steps == 1000
    assert config.allow_shared_cells == (n_plows > 1)


def test_environment_overrides_preserve_zero_and_false_values():
    _, config = resolve_environment(make_args(
        experiment="exp1", plows=2, obs="local", view=3,
        max_steps=7, spawn_prob=0, initial_clouds=0, exclusive_cells=False,
        plow_time_single=[1, 2, 3],
    ))
    assert config.n_plows == 2 and config.observation == "local" and config.view_size == 3
    assert config.max_steps == 7 and config.allow_shared_cells
    assert config.snow.cloud_spawn_prob == 0 and config.snow.initial_clouds == 0
    assert config.snow.plow_time_single == (0, 1, 2, 3)


def test_snow_preset_and_new_probability_spelling_are_honored():
    _, config = resolve_environment(make_args(experiment="exp3"))
    assert config.snow.scale_spawn_with_plows is False
    _, config = resolve_environment(make_args(
        experiment="exp3", cloud_spawn_prob=0.1, fixed_spawn_prob=False,
    ))
    assert config.snow.cloud_spawn_prob == 0.1
    assert config.snow.scale_spawn_with_plows is True


def test_training_presets_and_explicit_checkpoint_disable():
    from main import resolve_training

    args = make_args(experiment="exp4", checkpoint_every=0, rollout_len=4, steps=17, recurrent=False)
    _, config = resolve_environment(args)
    training, ppo = resolve_training(args, config)
    assert training == {"algo": "mappo", "steps": 17, "checkpoint_every": 0}
    assert ppo.rollout_len == 4 and ppo.recurrent is False


def test_resumed_checkpoint_schedule_starts_after_restored_progress(tmp_path):
    from main import make_checkpoint_callback

    class Trainer:
        env_steps = 2048
        saved = []

        def save(self, path):
            self.saved.append(path)

    trainer = Trainer()
    callback = make_checkpoint_callback(trainer, tmp_path, 1000)
    callback({"env_steps": 2560})
    callback({"env_steps": 3072})
    assert len(trainer.saved) == 1 and trainer.saved[0].endswith("model_step_3072.pt")


def test_interrupted_run_selects_latest_checkpoint_numerically(tmp_path):
    from main import checkpoint_path

    directory = tmp_path / "checkpoints"
    directory.mkdir()
    for step in (20, 100, 9):
        (directory / f"model_step_{step}.pt").touch()
    assert checkpoint_path(tmp_path).name == "model_step_100.pt"
    assert checkpoint_path(tmp_path, "model_step_20.pt").name == "model_step_20.pt"
    (tmp_path / "model.pt").touch()
    assert checkpoint_path(tmp_path).name == "model.pt"
