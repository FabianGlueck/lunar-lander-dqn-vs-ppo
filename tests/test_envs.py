import gymnasium as gym
from lunarlander.envs import make_env


def test_make_env_spaces():
    env = make_env(seed=0)
    assert env.observation_space.shape == (8,)
    assert env.action_space.n == 4
    env.close()


def test_make_env_reset_is_seeded():
    obs_a, _ = make_env(seed=42).reset()
    obs_b, _ = make_env(seed=42).reset()
    assert (obs_a == obs_b).all()


def test_make_env_passes_env_kwargs():
    env = make_env(seed=0, enable_wind=True, wind_power=10.0)
    assert env.unwrapped.enable_wind is True
    assert env.unwrapped.wind_power == 10.0
    env.close()
