from argparse import Namespace

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

    assert map_name == "network_3"


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
