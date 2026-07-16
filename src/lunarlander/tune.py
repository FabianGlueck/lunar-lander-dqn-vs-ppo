"""Hyperparameter-Tuning mit Optuna (algorithmus-agnostisch).

Definiert die Suchräume für DQN/PPO, eine Zielfunktion (kurzes Training +
Bewertung) und einen Study-Runner mit TPE-Sampler und MedianPruner, der
aussichtslose Trials früh abbricht und in eine SQLite-DB schreibt.
"""

import tempfile
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.callbacks import EvalCallback

from lunarlander import config
from lunarlander.evaluate import evaluate
from lunarlander.train import train


class TrialEvalCallback(EvalCallback):
    """Meldet die periodische Eval-Leistung an einen Optuna-Trial, damit der
    Pruner der Study aussichtslose Trials früh stoppen kann.

    Nach jeder periodischen Evaluation wird der mittlere Reward gemeldet; hält
    der Pruner den Trial für nicht vielversprechend, wird ``is_pruned`` gesetzt
    und das Training gestoppt (``_on_step`` gibt False zurück). Der Aufrufer
    prüft danach ``is_pruned`` und wirft ``optuna.TrialPruned``.
    """

    def __init__(self, eval_env, trial, log_path=None, n_eval_episodes=20,
                 eval_freq=10_000, deterministic=True, verbose=0):
        super().__init__(
            eval_env,
            log_path=log_path,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self):
        # Elternklasse führt bei Fälligkeit die Evaluation aus und aktualisiert
        # self.last_mean_reward.
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False   # Training sauber abbrechen
        return continue_training


def sample_hyperparams(trial, algo):
    """Zieht für einen Trial einen Hyperparameter-Satz aus dem Suchraum.

    Der Suchraum unterscheidet sich je Algorithmus (DQN wert-basiert,
    PPO policy-gradient). Unbekannter `algo` → `ValueError`.
    """
    algo = algo.lower()
    if algo == "dqn":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "gamma": trial.suggest_float("gamma", 0.98, 0.9999),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
            "target_update_interval": trial.suggest_categorical("target_update_interval", [250, 500, 1000]),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.3),
            "net_arch": trial.suggest_categorical("net_arch", [[64, 64], [128, 128], [256, 256]]),
        }
    if algo == "ppo":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "gamma": trial.suggest_float("gamma", 0.98, 0.9999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
            "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
            "net_arch": trial.suggest_categorical("net_arch", [[64, 64], [128, 128]]),
        }
    raise ValueError(f"Unbekannter Algorithmus: {algo!r}")


def sample_hyperparams_ppo_extended(trial):
    """Erweiterter PPO-Suchraum (experimentell, separat von der Standardsuche).

    Weitet die Dimensionen, die im ersten Tuning an den Suchraum-Rand liefen:
    kleinere `n_steps`, mehr `n_epochs`, größere Netze, breiteres `gae_lambda`.
    Nur für optionale „ppo_v2"-Experimente — nicht für die berichteten Ergebnisse.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.98, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.85, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
        "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20, 30]),
        "net_arch": trial.suggest_categorical("net_arch", [[64, 64], [128, 128], [256, 256]]),
    }


def objective(trial, algo, timesteps, log_root, sample_fn=None):
    """Optuna-Zielfunktion: kurz trainieren, bewerten, mittlere Rendite zurückgeben.

    Schleust einen `TrialEvalCallback` ins Training ein, damit der Pruner greifen
    kann. Wird der Trial geprunt, wird `optuna.TrialPruned` geworfen; sonst ist
    der Zielwert die mittlere Rendite einer frischen Schluss-Evaluation.

    `sample_fn` erlaubt einen alternativen Suchraum (Callable `trial -> dict`);
    ohne Angabe wird `sample_hyperparams(trial, algo)` genutzt.
    """
    params = sample_fn(trial) if sample_fn is not None else sample_hyperparams(trial, algo)
    log_dir = Path(log_root) / f"trial_{trial.number}"

    # Kleiner Kniff: die Fabrik legt den erzeugten Callback in `holder` ab, damit
    # wir nach dem Training an `is_pruned` herankommen (train gibt ihn nicht zurück).
    holder = {}

    def eval_callback_fn(eval_env, eval_freq):
        cb = TrialEvalCallback(
            eval_env,
            trial,
            log_path=str(log_dir),
            n_eval_episodes=config.CALLBACK_EVAL_EPISODES,
            eval_freq=eval_freq,
        )
        holder["cb"] = cb
        return cb

    model, _ = train(algo, params, seed=0, timesteps=timesteps, log_dir=log_dir,
                     eval_callback_fn=eval_callback_fn)

    if holder["cb"].is_pruned:
        raise optuna.TrialPruned()

    rewards = evaluate(model, seed=999, n_episodes=config.CALLBACK_EVAL_EPISODES)
    return float(np.mean(rewards))


def run_study(algo, n_trials, db_path, timesteps=None, study_name=None, sample_fn=None):
    """Startet/ergänzt eine Optuna-Study und gibt sie zurück.

    Nutzt TPE-Sampler + MedianPruner, maximiert die Rendite und persistiert in
    eine SQLite-DB. `load_if_exists=True` erlaubt Fortsetzen und parallele
    Prozesse auf derselben `db_path` (mehrere Terminals = echte Parallelität).

    Args:
        algo: "dqn" oder "ppo".
        n_trials: Anzahl der Trials in diesem Aufruf.
        db_path: Pfad der SQLite-Datei.
        timesteps: Steps pro Trial (Default: `config.TUNE_TIMESTEPS`).
        study_name: Name der Study (Default: `"<algo>_lunarlander"`).
        sample_fn: optionaler alternativer Suchraum (Callable `trial -> dict`).
    """
    timesteps = timesteps or config.TUNE_TIMESTEPS
    study_name = study_name or f"{algo}_lunarlander"
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=0),
        pruner=MedianPruner(),
        load_if_exists=True,
    )
    # Pro Trial schreibt der EvalCallback Logs; wir sammeln sie in einem Temp-Ordner.
    log_root = Path(tempfile.mkdtemp(prefix=f"tune_{algo}_"))
    study.optimize(
        lambda t: objective(t, algo, timesteps, log_root, sample_fn=sample_fn),
        n_trials=n_trials,
    )
    return study


def best_params(algo, db_path=None, study_name=None):
    """Lädt die beste Hyperparameter-Konfiguration aus einer gespeicherten Study.

    So muss man die getunten Params nicht von Hand kopieren — der finale Lauf
    liest sie direkt aus der `.db`.

    Args:
        algo: "dqn" oder "ppo".
        db_path: Pfad der SQLite-DB (Default: `config.STUDIES_DIR/<algo>.db`).
        study_name: Name der Study (Default: `"<algo>_lunarlander"`).

    Returns:
        Dict der besten Hyperparameter (`study.best_params`).
    """
    db_path = db_path or (config.STUDIES_DIR / f"{algo}.db")
    study_name = study_name or f"{algo}_lunarlander"
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
    return dict(study.best_params)
