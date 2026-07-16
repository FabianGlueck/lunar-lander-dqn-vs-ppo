"""CLI: finaler Mehr-Seed-Lauf und Bewertung.

Beispiele:
    uv run python -m scripts.run_final_eval --algo dqn                 # trainieren + bestes Modell bewerten
    uv run python -m scripts.run_final_eval --algo dqn --eval-only     # nur neu bewerten (kein Training)
    uv run python -m scripts.run_final_eval --algo dqn --which final   # das finale statt beste Modell bewerten

Trainiert (sofern nicht --eval-only) die getunte Konfiguration über alle
`config.SEEDS` voll aus und bewertet je Seed ein Modell auf `config.N_EVAL_EPISODES`
Episoden. Standardmäßig wird das **beste** Modell bewertet (`best_model.zip`, vom
EvalCallback als bester Checkpoint gesichert) — das ist bei DQN wegen möglicher
Instabilität am Trainingsende (catastrophic forgetting) die fairere, stabilere
Metrik. Ergebnis: `results/metrics/<algo>.csv` (Spalten: seed, mean_reward), der
Input für den Signifikanzvergleich.
"""

import argparse
import csv
import json

import numpy as np

from lunarlander import config
from lunarlander.agents import load_agent
from lunarlander.evaluate import evaluate
from lunarlander.train import train
from lunarlander.tune import best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["dqn", "ppo"])
    parser.add_argument("--params", default=None,
                        help="JSON-String der Hyperparameter (Override). Ohne Angabe "
                             "werden die besten Params aus der Study geladen.")
    parser.add_argument("--timesteps", type=int, default=config.FINAL_TIMESTEPS)
    parser.add_argument("--which", choices=["best", "final"], default="best",
                        help="Welches Modell bewertet wird: best (Early Stopping, Default) "
                             "oder final (Stand am Trainingsende).")
    parser.add_argument("--eval-only", action="store_true",
                        help="Nicht trainieren — vorhandene gespeicherte Modelle neu bewerten.")
    parser.add_argument("--tag", default=None,
                        help="Suffix für alle Ausgaben (isoliertes Experiment), z.B. "
                             "v2_1m → metrics/ppo_v2_1m.csv, models/ppo_v2_1m_seed*. "
                             "Schützt die Standard-Ergebnisse vor Überschreiben.")
    args = parser.parse_args()

    config.ensure_dirs()

    # Params nur fürs Training nötig (bei --eval-only werden gespeicherte Modelle geladen).
    if not args.eval_only:
        params = json.loads(args.params) if args.params else best_params(args.algo)
        print(f"Verwende Params: {params}")

    suffix = f"_{args.tag}" if args.tag else ""   # isoliert Experimente von den Standard-Ausgaben
    out_csv = config.METRICS_DIR / f"{args.algo}{suffix}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mean_reward"])
        for seed in config.SEEDS:
            log_dir = config.MODELS_DIR / f"{args.algo}{suffix}_seed{seed}"
            model_path = log_dir / f"{args.which}_model"

            if args.eval_only:
                if not model_path.with_suffix(".zip").exists():
                    raise SystemExit(
                        f"Modell fehlt: {model_path}.zip — erst ohne --eval-only trainieren."
                    )
                model = load_agent(args.algo, model_path)
            else:
                # TensorBoard-Logs je Seed unter results/tb/ (live: tensorboard --logdir results/tb)
                tb_log = config.TB_DIR / f"{args.algo}{suffix}_seed{seed}"
                trained, _ = train(args.algo, params, seed=seed,
                                   timesteps=args.timesteps, log_dir=log_dir,
                                   tensorboard_log=str(tb_log))
                trained.save(log_dir / "final_model")
                # Das gewählte Modell laden (best_model.zip stammt vom EvalCallback).
                model = load_agent(args.algo, model_path)

            # Bewertung mit einem seed-abhängigen, aber vom Training getrennten Seed.
            rewards = evaluate(model, seed=10_000 + seed,
                               n_episodes=config.N_EVAL_EPISODES)
            mean_r = float(np.mean(rewards))
            writer.writerow([seed, mean_r])
            print(f"[{args.algo}] seed={seed} ({args.which}) mean_reward={mean_r:.1f}")

    print(f"Ergebnisse ({args.which}) gespeichert in: {out_csv}")


if __name__ == "__main__":
    main()
