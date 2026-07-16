import numpy as np
from lunarlander import stats


def test_welch_identical_data_high_p():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t, p = stats.welch_t_test(a, a)
    assert abs(t) < 1e-9
    assert p > 0.99


def test_welch_separated_data_low_p():
    a = np.array([200.0, 205.0, 198.0, 202.0, 199.0])
    b = np.array([100.0, 95.0, 105.0, 98.0, 102.0])
    _, p = stats.welch_t_test(a, b)
    assert p < 0.05


def test_confidence_interval_brackets_mean():
    data = np.array([10.0, 12.0, 11.0, 9.0, 13.0])
    mean, low, high = stats.confidence_interval(data)
    assert low < mean < high
    assert abs(mean - 11.0) < 1e-9


def test_cohens_d_sign_and_magnitude():
    a = np.array([10.0, 11.0, 9.0, 10.0, 10.0])
    b = np.array([0.0, 1.0, -1.0, 0.0, 0.0])
    d = stats.cohens_d(a, b)
    assert d > 2.0  # großer Effekt


def test_summarize_keys():
    a = np.array([200.0, 210.0, 190.0])
    b = np.array([150.0, 160.0, 140.0])
    out = stats.summarize(a, b, "dqn", "ppo")
    for key in ("mean_dqn", "mean_ppo", "p_value", "cohens_d", "ci_dqn", "ci_ppo"):
        assert key in out


def test_confidence_interval_diff_brackets_difference():
    a = np.array([200.0, 205.0, 198.0, 202.0, 199.0])
    b = np.array([100.0, 95.0, 105.0, 98.0, 102.0])
    diff, low, high = stats.confidence_interval_diff(a, b)
    assert abs(diff - (a.mean() - b.mean())) < 1e-9
    assert low < diff < high
    assert low > 0  # clearly separated groups → CI excludes 0


def test_summarize_includes_ci_diff():
    a = np.array([200.0, 210.0, 190.0])
    b = np.array([150.0, 160.0, 140.0])
    out = stats.summarize(a, b, "dqn", "ppo")
    assert "ci_diff" in out


def test_mann_whitney_separated_low_p():
    a = np.array([200.0, 205.0, 198.0, 202.0, 199.0])
    b = np.array([100.0, 95.0, 105.0, 98.0, 102.0])
    _, p = stats.mann_whitney(a, b)
    assert p < 0.05


def test_mann_whitney_identical_high_p():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _, p = stats.mann_whitney(a, a)
    assert p > 0.05


def test_steps_to_threshold_found():
    ts = np.array([10_000, 20_000, 30_000, 40_000])
    means = np.array([-50.0, 150.0, 205.0, 230.0])
    assert stats.steps_to_threshold(ts, means, threshold=200) == 30_000


def test_steps_to_threshold_not_reached():
    ts = np.array([10_000, 20_000])
    means = np.array([50.0, 120.0])
    assert stats.steps_to_threshold(ts, means, threshold=200) is None


def test_describe_keys_and_values():
    rewards = np.array([227.0, 227.0, 250.0, 233.0, 180.0])
    d = stats.describe(rewards, threshold=200)
    for k in ("mean", "std", "ci_low", "ci_high", "min", "max", "n_solved", "pct_solved"):
        assert k in d
    assert d["n_solved"] == 4        # 4 von 5 >= 200
    assert d["pct_solved"] == 80.0
    assert d["max"] == 250.0
