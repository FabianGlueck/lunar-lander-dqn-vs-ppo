import numpy as np
import pytest
from stable_baselines3 import DQN, PPO
from lunarlander.agents import make_agent, load_agent, make_random_agent
from lunarlander.envs import make_env


def test_make_agent_dqn():
    env = make_env(seed=0)
    model = make_agent("dqn", {"learning_rate": 1e-3}, env, seed=0)
    assert isinstance(model, DQN)
    env.close()


def test_make_agent_ppo():
    env = make_env(seed=0)
    model = make_agent("ppo", {"learning_rate": 3e-4}, env, seed=0)
    assert isinstance(model, PPO)
    env.close()


def test_make_agent_net_arch():
    env = make_env(seed=0)
    model = make_agent("dqn", {"net_arch": [64, 64]}, env, seed=0)
    assert isinstance(model, DQN)
    env.close()


def test_make_agent_unknown():
    env = make_env(seed=0)
    with pytest.raises(ValueError):
        make_agent("sarsa", {}, env, seed=0)
    env.close()


def test_make_agent_tensorboard_log(tmp_path):
    env = make_env(seed=0)
    model = make_agent("dqn", {}, env, seed=0, tensorboard_log=str(tmp_path))
    assert model.tensorboard_log == str(tmp_path)
    env.close()


def test_make_agent_no_tensorboard_by_default():
    env = make_env(seed=0)
    model = make_agent("dqn", {}, env, seed=0)
    assert model.tensorboard_log is None
    env.close()


def test_load_agent_roundtrip(tmp_path):
    env = make_env(seed=0)
    model = make_agent("dqn", {}, env, seed=0)
    model.save(tmp_path / "m")
    loaded = load_agent("dqn", tmp_path / "m")
    assert isinstance(loaded, DQN)
    env.close()


def test_load_agent_unknown(tmp_path):
    with pytest.raises(ValueError):
        load_agent("sarsa", tmp_path / "m")


def test_random_agent_predicts_actions_aus_dem_aktionsraum():
    env = make_env(seed=0)
    agent = make_random_agent(env, seed=0)
    actions, state = agent.predict(np.zeros((1, 8), dtype=np.float32))
    assert actions.shape == (1,)
    assert env.action_space.contains(int(actions[0]))
    assert state is None
    env.close()


def test_random_agent_folgt_der_batch_groesse():
    env = make_env(seed=0)
    agent = make_random_agent(env, seed=0)
    actions, _ = agent.predict(np.zeros((4, 8), dtype=np.float32))
    assert actions.shape == (4,)
    env.close()


def test_random_agent_gleicher_seed_gleiche_aktionen():
    env = make_env(seed=0)
    obs = np.zeros((8, 8), dtype=np.float32)
    erste, _ = make_random_agent(env, seed=42).predict(obs)
    zweite, _ = make_random_agent(env, seed=42).predict(obs)
    assert np.array_equal(erste, zweite)
    env.close()


def test_random_agent_ignoriert_deterministic():
    """Die Zufalls-Policy bleibt zufällig — sonst wäre sie als Baseline wertlos."""
    env = make_env(seed=0)
    agent = make_random_agent(env, seed=0)
    obs = np.zeros((200, 8), dtype=np.float32)
    actions, _ = agent.predict(obs, deterministic=True)
    assert len(np.unique(actions)) > 1
    env.close()
