"""Erzeugt die beiden RESULTS-Abbildungen fürs Poster nach docs/figures/.

- `comparison_boxplot` — 5 Seeds je Algorithmus als Boxplot mit Einzelpunkten.
  Zeigt genau die Daten, über die der Signifikanztest rechnet (bestes Modell,
  je 100 Episoden). Ohne Zufalls-Baseline: die würde die y-Achse auf 400 Punkte
  aufziehen und die Boxen unlesbar quetschen — die Lernkurve daneben zeigt den
  Zufalls-Boden ohnehin.
- `gap_slope` — je Seed eine Linie best → final. Trägt Seed-Streuung,
  Überlappung und Trainingsverfall in einem Bild.

`best` kommt aus results/metrics/<algo>.csv (dieselbe Rechnung wie im Notebook:
100 Episoden je Seed, Eval-Seed 10_000+s), `final` wird frisch ausgewertet.

Aufruf:
    uv run python -m scripts.plot_results_figures
"""

import numpy as np
import pandas as pd

from lunarlander import config, plots
from lunarlander.agents import load_agent
from lunarlander.evaluate import evaluate

ALGOS = ["dqn", "ppo"]
OUT = config.DOCS_DIR / "figures"


def best_je_seed(algo: str) -> np.ndarray:
    """Bestes Modell je Seed — die eingecheckten Poster-Zahlen."""
    df = pd.read_csv(config.METRICS_DIR / f"{algo}.csv").sort_values("seed")
    return df["mean_reward"].to_numpy()


def final_je_seed(algo: str) -> np.ndarray:
    """Finales Modell je Seed, wie im Notebook: 100 Episoden ab Eval-Seed 10_000+s."""
    werte = []
    for s in config.SEEDS:
        m = load_agent(algo, config.MODELS_DIR / f"{algo}_seed{s}" / "final_model")
        werte.append(evaluate(m, seed=10_000 + s, n_episodes=config.N_EVAL_EPISODES).mean())
    return np.array(werte)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    best = {a: best_je_seed(a) for a in ALGOS}
    print("best je Seed (aus metrics/):")
    for a in ALGOS:
        print(f"  {a.upper()}: {np.round(best[a], 1)}  -> {best[a].mean():.1f} ± {best[a].std(ddof=1):.1f}")

    print(f"\nfinal je Seed (je {config.N_EVAL_EPISODES} Episoden, dauert etwas)...")
    final = {a: final_je_seed(a) for a in ALGOS}
    for a in ALGOS:
        print(f"  {a.upper()}: {np.round(final[a], 1)}  -> {final[a].mean():.1f} ± {final[a].std(ddof=1):.1f}"
              f"   gelöst: {int((final[a] >= config.SOLVED_THRESHOLD).sum())}/{len(config.SEEDS)}")

    # PNG für PowerPoint, PDF für den Druck — wie die übrigen Figuren.
    for endung in ("png", "pdf"):
        plots.comparison_plot(best, OUT / f"comparison_boxplot.{endung}", show_points=True)
        plots.gap_slope_plot(best, final, OUT / f"gap_slope.{endung}")
    print(f"\nGeschrieben: {OUT}/comparison_boxplot.{{png,pdf}} und gap_slope.{{png,pdf}}")


if __name__ == "__main__":
    main()
