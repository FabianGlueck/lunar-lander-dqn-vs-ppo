import matplotlib.pyplot as plt
import numpy as np
import pytest
from lunarlander import plots


def _make_npz(path, seed):
    np.savez(path, timesteps=np.array([10, 20, 30]),
             results=np.random.default_rng(seed).normal(0, 1, size=(3, 5)))
    return path


def test_learning_curve_saves(tmp_path):
    paths = {"dqn": [_make_npz(tmp_path / "d0.npz", 0)],
             "ppo": [_make_npz(tmp_path / "p0.npz", 1)]}
    out = plots.learning_curve(paths, tmp_path / "lc.png")
    assert out.exists()


def test_learning_curve_baseline_adds_reference_line(tmp_path):
    paths = {"dqn": [_make_npz(tmp_path / "d0.npz", 0)]}
    fig, ax = plots._build_learning_curve(paths, baseline=-120)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)
    assert any("-120" in label for label in labels)


def test_learning_curve_no_baseline_by_default(tmp_path):
    paths = {"dqn": [_make_npz(tmp_path / "d0.npz", 0)]}
    fig, ax = plots._build_learning_curve(paths)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)
    assert not any("random" in label.lower() for label in labels)


def test_learning_curve_marks_best_mean_checkpoint_per_algorithm(tmp_path):
    d0 = tmp_path / "d0.npz"
    d1 = tmp_path / "d1.npz"
    timesteps = np.array([10, 20, 30])
    np.savez(d0, timesteps=timesteps,
             results=np.array([[1, 1], [6, 6], [3, 3]]))
    np.savez(d1, timesteps=timesteps,
             results=np.array([[2, 2], [4, 4], [9, 9]]))

    fig, ax = plots._build_learning_curve(
        {"dqn": [d0, d1]}, mark_best_checkpoints=True)
    offsets = [collection.get_offsets() for collection in ax.collections]
    plt.close(fig)

    assert any(np.array_equal(points, np.array([[30, 6]]))
               for points in offsets)


def test_learning_curve_beschriftung_ist_englisch(tmp_path):
    paths = {"dqn": [_make_npz(tmp_path / "d0.npz", 0)]}
    fig, ax = plots._build_learning_curve(paths, baseline=-120)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    ylabel = ax.get_ylabel()
    plt.close(fig)
    assert ylabel == "Mean eval return"
    assert any("solved" in label for label in labels)
    assert any("random baseline" in label for label in labels)


def test_comparison_plot_saves(tmp_path):
    rewards = {"dqn": np.array([200.0, 210.0, 190.0]),
               "ppo": np.array([150.0, 160.0, 140.0])}
    out = plots.comparison_plot(rewards, tmp_path / "cmp.png")
    assert out.exists()


def test_comparison_plot_with_points_and_baseline_saves(tmp_path):
    rewards = {"dqn": np.array([200.0, 210.0, 190.0]),
               "ppo": np.array([150.0, 160.0, 140.0])}
    out = plots.comparison_plot(rewards, tmp_path / "cmp2.png", baseline=-120, show_points=True)
    assert out.exists()


def test_efficiency_plot_saves(tmp_path):
    out = plots.efficiency_plot({"dqn": 200_000, "ppo": 150_000}, tmp_path / "eff.png")
    assert out.exists()


def test_efficiency_plot_handles_none(tmp_path):
    out = plots.efficiency_plot({"dqn": 200_000, "ppo": None}, tmp_path / "eff2.png")
    assert out.exists()


def test_gap_plot_saves(tmp_path):
    out = plots.gap_plot({"dqn": 236.0, "ppo": 240.0},
                         {"dqn": 152.0, "ppo": 235.0}, tmp_path / "gap.png")
    assert out.exists()


def test_reward_histogram_saves(tmp_path):
    out = plots.reward_histogram(np.array([200.0, 210.0, 190.0, 205.0, 150.0]),
                                 tmp_path / "hist.png")
    assert out.exists()


def _runs():
    return {"dqn": np.array([200.0, 220.0, 240.0]),
            "ppo": np.array([250.0, 260.0, 270.0]),
            "random": np.array([-150.0, -100.0, -200.0])}


def test_mean_runs_plot_saves(tmp_path):
    out = plots.mean_runs_plot(_runs(), tmp_path / "g.png")
    assert out.exists()


def test_mean_runs_balkenhoehe_ist_der_mittelwert():
    fig, ax = plots._build_mean_runs(_runs())
    hoehen = [b.get_height() for b in ax.containers[0]]
    plt.close(fig)
    assert hoehen == pytest.approx([220.0, 260.0, -150.0])


def test_mean_runs_zeigt_alle_einzelwerte():
    """Bei n=10 ist der Balken allein unehrlich — die Rohwerte müssen sichtbar sein."""
    from matplotlib.collections import PathCollection

    fig, ax = plots._build_mean_runs(_runs())
    punkte = sum(c.get_offsets().shape[0]
                 for c in ax.collections if isinstance(c, PathCollection))
    plt.close(fig)
    assert punkte == 9  # 3 Labels x 3 Durchläufe


def test_mean_runs_wertlabel_steht_frei():
    """Am Balkenende kollidiert das Label mit Errorbar und Punkten — es muss
    jenseits der gesamten Säule sitzen (oben bei positiv, unten bei negativ)."""
    fig, ax = plots._build_mean_runs(_runs())
    ypos = {t.get_text(): t.xy[1] for t in ax.texts}
    plt.close(fig)
    assert ypos["220 ± 16"] >= 240.0    # dqn: über dem höchsten Einzelwert
    assert ypos["260 ± 8"] >= 270.0     # ppo
    assert ypos["-150 ± 41"] <= -200.0  # random: unter dem tiefsten Einzelwert


def test_mean_runs_hat_geloest_linie():
    fig, ax = plots._build_mean_runs(_runs())
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)
    assert any("200" in label for label in labels)


def test_mean_runs_beschriftung_ist_englisch():
    fig, ax = plots._build_mean_runs(_runs())
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    ylabel = ax.get_ylabel()
    plt.close(fig)
    assert ylabel == "Return (mean ± std)"
    assert any("solved" in label for label in labels)
