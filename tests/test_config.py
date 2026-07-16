from pathlib import Path
from lunarlander import config


def test_core_constants():
    assert config.ENV_ID == "LunarLander-v3"
    assert config.SEEDS == [0, 1, 2, 3, 4]
    assert config.SOLVED_THRESHOLD == 200
    assert config.N_EVAL_EPISODES == 100


def test_paths_are_paths():
    for p in (config.RESULTS_DIR, config.STUDIES_DIR, config.MODELS_DIR, config.METRICS_DIR):
        assert isinstance(p, Path)


def test_ensure_dirs_creates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "STUDIES_DIR", tmp_path / "results" / "studies")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "results" / "models")
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "results" / "metrics")
    config.ensure_dirs()
    assert (tmp_path / "results" / "studies").is_dir()
