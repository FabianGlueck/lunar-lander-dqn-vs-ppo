"""CLI: Optuna-Hyperparameter-Suche für einen Algorithmus starten.

Beispiel:
    uv run python -m scripts.run_tuning --algo dqn --trials 30

Schreibt die Study nach results/studies/<algo>.db und gibt die beste
Konfiguration aus. Läufe sind fortsetzbar: derselbe Aufruf ergänzt weitere
Trials in dieselbe DB (mehrere Terminals parallel = echte Parallelität).
"""

import argparse

from lunarlander import config
from lunarlander.tune import run_study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["dqn", "ppo"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=config.TUNE_TIMESTEPS)
    args = parser.parse_args()

    config.ensure_dirs()
    db_path = config.STUDIES_DIR / f"{args.algo}.db"
    study = run_study(args.algo, n_trials=args.trials, db_path=db_path,
                      timesteps=args.timesteps)
    # Beste Params notieren – sie sind der Input für run_final_eval.
    print(f"Beste Rendite: {study.best_value:.1f}")
    print(f"Beste Params: {study.best_params}")
    print(f"Study gespeichert in: {db_path}")


if __name__ == "__main__":
    main()
