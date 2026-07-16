import numpy as np
from lunarlander.train import train


def test_train_returns_model_and_history(tmp_path):
    model, hist_path = train(
        "dqn",
        {"learning_rate": 1e-3, "learning_starts": 100},
        seed=0,
        timesteps=1500,
        log_dir=tmp_path,
    )
    assert model is not None
    assert hist_path.exists()
    data = np.load(hist_path)
    assert "timesteps" in data and "results" in data
    assert len(data["timesteps"]) >= 1


def test_train_passes_tensorboard_log(tmp_path):
    model, _ = train(
        "dqn",
        {"learning_starts": 100},
        seed=0,
        timesteps=1500,
        log_dir=tmp_path / "run",
        tensorboard_log=str(tmp_path / "tb"),
    )
    assert model.tensorboard_log == str(tmp_path / "tb")
