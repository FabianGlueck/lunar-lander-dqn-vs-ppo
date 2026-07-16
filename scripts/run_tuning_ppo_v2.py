"""CLI: experimentelles PPO-Tuning mit erweitertem Suchraum (separate Study).

Läuft in eine EIGENE DB (`results/studies/ppo_v2.db`, Study-Name "ppo_v2") und lässt
die Standard-`ppo.db` unangetastet — nur aus Interesse, NICHT für die berichteten
Poster-Ergebnisse.

Der erweiterte Suchraum (`sample_hyperparams_ppo_extended`) weitet die Dimensionen,
die im ersten Tuning an den Rand liefen: kleinere `n_steps`, mehr `n_epochs`, größere
Netze, breiteres `gae_lambda`.

Beispiel:
    uv run python -m scripts.run_tuning_ppo_v2 --trials 40

Fortsetzbar (load_if_exists): derselbe Befehl ergänzt weitere Trials in dieselbe DB.

Beste Params später auslesen:
    from lunarlander.tune import best_params
    best_params("ppo", db_path="results/studies/ppo_v2.db", study_name="ppo_v2")
"""

import argparse

from lunarlander import config
from lunarlander.tune import run_study, sample_hyperparams_ppo_extended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--timesteps", type=int, default=config.TUNE_TIMESTEPS)
    args = parser.parse_args()

    config.ensure_dirs()
    db_path = config.STUDIES_DIR / "ppo_v2.db"
    study = run_study("ppo", n_trials=args.trials, db_path=db_path,
                      timesteps=args.timesteps, study_name="ppo_v2",
                      sample_fn=sample_hyperparams_ppo_extended)
    print(f"Beste Rendite (ppo_v2): {study.best_value:.1f}")
    print(f"Beste Params: {study.best_params}")
    print(f"Study gespeichert in: {db_path}")
    print("(Vergleich: Standard-PPO-Tuning erreichte 212.6)")


if __name__ == "__main__":
    main()
