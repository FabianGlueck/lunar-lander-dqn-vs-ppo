"""Trainings-Loop (algorithmus-agnostisch) mit periodischem Eval-Logging."""

from pathlib import Path

from stable_baselines3.common.callbacks import EvalCallback

from lunarlander import config
from lunarlander.agents import make_agent
from lunarlander.envs import make_env


def train(algo, hyperparams, seed, timesteps, log_dir, eval_callback_fn=None,
          tensorboard_log=None):
    """Trainiert ein Modell und protokolliert dabei periodische Evaluationen.

    Args:
        algo: "dqn" oder "ppo".
        hyperparams: Hyperparameter-Dict (siehe `make_agent`).
        seed: Seed für Training; die Eval-Umgebung nutzt `seed + 1000`.
        timesteps: Anzahl der Trainings-Steps.
        log_dir: Zielordner für die besten Modelle und `evaluations.npz`.
        eval_callback_fn: optionale Fabrik `(eval_env, eval_freq) -> Callback`.
            Ohne sie wird der Standard-`EvalCallback` genutzt. Mit ihr kann der
            Aufrufer einen eigenen Callback einschleusen (z. B. trial-bewusst
            fürs Optuna-Pruning) — `train` selbst bleibt dadurch Optuna-frei.
        tensorboard_log: Ordner für TensorBoard-Logs (None = kein TB-Logging).

    Returns:
        (model, pfad_zu_evaluations.npz). Die .npz enthält `timesteps` und
        `results` (Rewards je Eval-Zeitpunkt) und speist die Lernkurve.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(seed=seed)
    eval_env = make_env(seed=seed + 1000)  # separater Seed für faire Evaluation

    model = make_agent(algo, hyperparams, env, seed=seed, tensorboard_log=tensorboard_log)

    # Bei sehr kurzen Läufen (Tests) so kappen, dass mind. eine Eval stattfindet.
    eval_freq = min(config.EVAL_FREQ, max(timesteps // 2, 1))
    if eval_callback_fn is None:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(log_dir),
            log_path=str(log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=config.CALLBACK_EVAL_EPISODES,
            deterministic=True,
            verbose=0,
        )
    else:
        # Optuna-agnostisch: die Fabrik baut den Eval-Callback (z. B. trial-bewusst
        # fürs Pruning). train.py kennt Optuna nicht.
        eval_cb = eval_callback_fn(eval_env, eval_freq)
    model.learn(total_timesteps=timesteps, callback=eval_cb)

    env.close()
    eval_env.close()
    return model, log_dir / "evaluations.npz"
