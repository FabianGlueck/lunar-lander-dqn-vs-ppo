"""Deterministische Bewertung eines trainierten Modells."""

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from lunarlander.envs import make_env


def evaluate(model, seed, n_episodes):
    """Spielt `n_episodes` mit deterministischer Policy und liefert die Rewards.

    Args:
        model: ein trainiertes SB3-Modell.
        seed: Seed der frischen Bewertungs-Umgebung (getrennt vom Training halten,
            damit auf ungesehenen Startzuständen bewertet wird).
        n_episodes: Anzahl der Bewertungs-Episoden.

    Returns:
        np.ndarray der Form `(n_episodes,)` mit dem Gesamt-Reward je Episode —
        die Rohdaten für Mittelwert, Lernkurve und Signifikanztests.
    """
    eval_env = make_env(seed=seed)
    rewards, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        return_episode_rewards=True,   # Liste je Episode statt nur Mittelwert
    )
    eval_env.close()
    return np.asarray(rewards, dtype=float)
