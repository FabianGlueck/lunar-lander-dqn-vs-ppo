"""Agent-Factory.

Dies ist die **einzige** Stelle der Pipeline, die den Unterschied zwischen den
Algorithmen kennt. Alle anderen Module (train, evaluate, tune) arbeiten
algorithmus-blind auf dem zurückgegebenen SB3-Modell — deshalb lässt sich später
ein weiterer Algorithmus (z. B. PPO) hier "einstecken", ohne sie anzufassen.
"""

import copy

import numpy as np
from stable_baselines3 import DQN, PPO

# Kürzel -> SB3-Klasse. Um einen Algorithmus zu ergänzen, hier eintragen.
_ALGOS = {"dqn": DQN, "ppo": PPO}


def make_agent(algo: str, hyperparams: dict, env, seed: int, tensorboard_log=None):
    """Erzeugt ein untrainiertes SB3-Modell für den gewählten Algorithmus.

    Args:
        algo: "dqn" oder "ppo" (Groß-/Kleinschreibung egal).
        hyperparams: an den SB3-Konstruktor durchgereichte Hyperparameter.
            Sonderfall `net_arch`: wird in `policy_kwargs` übersetzt (SB3-Konvention).
            Das übergebene Dict wird nicht verändert (es wird kopiert).
        env: die (ggf. vektorisierte) Trainingsumgebung.
        seed: Seed des Modells für reproduzierbare Initialisierung/Exploration.
        tensorboard_log: Ordner für TensorBoard-Logs (None = kein TB-Logging).

    Returns:
        Ein SB3-`DQN`- oder `PPO`-Objekt mit `MlpPolicy`.

    Raises:
        ValueError: bei unbekanntem `algo`.
    """
    algo = algo.lower()
    if algo not in _ALGOS:
        raise ValueError(f"Unbekannter Algorithmus: {algo!r}. Erlaubt: {list(_ALGOS)}")

    params = dict(hyperparams)  # Kopie, Original nicht mutieren
    net_arch = params.pop("net_arch", None)
    if net_arch is not None:
        params["policy_kwargs"] = {"net_arch": net_arch}

    cls = _ALGOS[algo]
    return cls("MlpPolicy", env, seed=seed, verbose=0,
               tensorboard_log=tensorboard_log, **params)


def load_agent(algo: str, path):
    """Lädt ein gespeichertes SB3-Modell des passenden Algorithmus.

    Args:
        algo: "dqn" oder "ppo" — bestimmt die SB3-Klasse zum Laden.
        path: Pfad zur `.zip` (mit oder ohne Endung).

    Returns:
        Das geladene SB3-Modell.

    Raises:
        ValueError: bei unbekanntem `algo`.
    """
    algo = algo.lower()
    if algo not in _ALGOS:
        raise ValueError(f"Unbekannter Algorithmus: {algo!r}. Erlaubt: {list(_ALGOS)}")
    return _ALGOS[algo].load(path)


class _RandomAgent:
    """Policy, die gleichverteilt aus dem Aktionsraum zieht.

    Bietet nur `predict` — genau das, was SB3s `evaluate_policy` von einem Modell
    verlangt. Dadurch läuft die Baseline durch dasselbe algo-blinde `evaluate()`
    wie ein echtes DQN/PPO-Modell, ohne dort eine Sonderbehandlung zu brauchen.
    """

    def __init__(self, action_space, seed: int):
        # Kopie: das Seeden darf den Aktionsraum der übergebenen Env nicht verstellen.
        self.action_space = copy.deepcopy(action_space)
        self.action_space.seed(seed)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Je Beobachtung im Batch eine Zufallsaktion.

        `deterministic` wird bewusst ignoriert: eine deterministische Zufalls-Policy
        wäre als Baseline sinnlos. Die Signatur bleibt trotzdem SB3-kompatibel,
        weil `evaluate_policy` das Argument immer mitgibt.

        Returns:
            `(actions, None)` — actions der Form `(batch,)`, kein rekurrenter Zustand.
        """
        n = len(observation)
        actions = np.array([self.action_space.sample() for _ in range(n)])
        return actions, None


def make_random_agent(env, seed: int):
    """Erzeugt die Zufalls-Baseline: der Boden, den jeder Algorithmus schlagen muss.

    Args:
        env: Env, deren Aktionsraum die Policy bespielt (wird nicht verändert).
        seed: Startwert → reproduzierbare Aktionsfolge.

    Returns:
        Ein Objekt mit SB3-`predict`-Schnittstelle, direkt an `evaluate()` übergebbar.
    """
    return _RandomAgent(env.action_space, seed)
