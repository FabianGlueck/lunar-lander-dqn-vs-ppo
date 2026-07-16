import numpy as np
from lunarlander.agents import make_agent, make_random_agent
from lunarlander.envs import make_env
from lunarlander.evaluate import evaluate


def test_evaluate_shape():
    env = make_env(seed=0)
    model = make_agent("dqn", {"learning_starts": 100}, env, seed=0)
    rewards = evaluate(model, seed=123, n_episodes=3)
    assert isinstance(rewards, np.ndarray)
    assert rewards.shape == (3,)
    assert rewards.dtype.kind == "f"
    env.close()


def test_evaluate_akzeptiert_zufallsagent():
    """evaluate bleibt algo-blind: der Zufallsagent geht durch dieselbe Funktion."""
    env = make_env(seed=0)
    agent = make_random_agent(env, seed=0)
    rewards = evaluate(agent, seed=123, n_episodes=2)
    assert rewards.shape == (2,)
    assert rewards.dtype.kind == "f"
    env.close()
