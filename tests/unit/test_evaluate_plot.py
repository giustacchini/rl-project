import json

import pytest

from evaluate import main, plot_histories, plot_history, read_history


def test_plot_reads_current_and_legacy_reward_history(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps([
        {"update": 1, "mean_return": None},
        {"update": 2, "mean_return": 3},
        {"update": 3, "episode_metrics": {"total_reward": 7}},
    ]))
    assert read_history(tmp_path) == ([1, 2, 3], [None, 3.0, 7.0])


def test_history_without_finished_episodes_has_a_clear_error(tmp_path):
    path = tmp_path / "history.json"
    path.write_text('[{"update": 1, "total_reward": null}]')
    with pytest.raises(ValueError, match="complete at least one"):
        read_history(path)


def test_plot_exports_reward_and_environment_metric(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    path = tmp_path / "history.json"
    path.write_text(json.dumps([
        {"update": 1, "total_reward": 2, "episode_metrics": {"snow_cleared": 4}},
        {"update": 2, "total_reward": 5, "episode_metrics": {"snow_cleared": 9}},
    ]))
    output = plot_history(path, tmp_path / "reward.png")
    assert output.read_bytes().startswith(b"\x89PNG")
    output = plot_history(path, tmp_path / "snow.svg", "snow_cleared")
    assert "<svg" in output.read_text()


def test_comparison_preserves_each_history_length_and_labels(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    import matplotlib.axes

    first, second = tmp_path / "ppo.json", tmp_path / "mappo.json"
    first.write_text(json.dumps([
        {"update": 1, "env_steps": 8, "total_reward": None},
        {"update": 2, "env_steps": 16, "total_reward": 4},
    ]))
    second.write_text(json.dumps([
        {"update": 1, "env_steps": 32, "total_reward": 2},
        {"update": 2, "env_steps": 64, "total_reward": 5},
        {"update": 3, "env_steps": 96, "total_reward": 7},
    ]))
    original_plot = matplotlib.axes.Axes.plot
    captured = []

    def record(self, x, y, **kwargs):
        captured.append((x, y, kwargs["label"]))
        return original_plot(self, x, y, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", record)
    output = plot_histories(
        [first, second], tmp_path / "comparison.png",
        labels=["PPO", "MAPPO"], x_axis="env_steps",
    )
    assert output.read_bytes().startswith(b"\x89PNG")
    assert captured[0][0] == [8, 16] and captured[0][2] == "PPO"
    assert captured[1] == ([32, 64, 96], [2.0, 5.0, 7.0], "MAPPO")
    assert str(captured[0][1][0]) == "nan"  # missing reward remains a gap


def test_cli_accepts_multiple_histories_and_automatic_run_labels(tmp_path, monkeypatch):
    import evaluate

    for name in ("exp1", "exp3"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "history.json").write_text('[{"update": 1, "total_reward": 2}]')
    monkeypatch.chdir(tmp_path)
    # A real CLI call covers the default comparison filename and legend labels.
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    original = evaluate.read_history
    loaded = []

    def read(path, *args):
        loaded.append(path)
        return original(path, *args)

    monkeypatch.setattr(evaluate, "read_history", read)
    main(["exp1", "exp3/history.json"])
    assert loaded == [type(tmp_path)("exp1"), type(tmp_path)("exp3/history.json")]
    assert (tmp_path / "comparison_total_reward.png").exists()


def test_comparison_reports_bad_label_count_and_missing_axis(tmp_path):
    path = tmp_path / "history.json"
    path.write_text('[{"update": 1, "total_reward": 2}]')
    with pytest.raises(ValueError, match="one label per history"):
        plot_histories([path, path], tmp_path / "unused.png", labels=["only one"])
    with pytest.raises(ValueError, match="history.json.*env_steps"):
        plot_histories([path], tmp_path / "unused.png", x_axis="env_steps")
