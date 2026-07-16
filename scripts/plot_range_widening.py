"""Lektions-Figur: hohe Importance + Bestwert am Suchraum-Rand → Range erweitern lohnt sich.

Liest die beiden PPO-Studies (Standard `ppo.db` und erweitert `ppo_v2.db`) und
erzeugt eine 2-Panel-Grafik:
  A) Parameter-Importance der Standard-Study; Parameter, deren *bester* Wert am
     Rand ihrer Range lag, sind markiert (↑ oberer / ↓ unterer Rand).
  B) Bester Objective-Wert Standard vs. erweitert — der Beleg, dass das Erweitern
     der markierten Ranges den Tuning-Wert hebt.

Eigenständig (fasst die Pipeline-Module nicht an). Ausgabe: docs/figures/.

    uv run python -m scripts.plot_range_widening
"""

import math
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.distributions import CategoricalDistribution

from lunarlander import config
from lunarlander import plots as P  # Import setzt das Lunar-Theme (rcParams) + Palette

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = config.STUDIES_DIR.parent.parent / "docs" / "figures"


def edge_of(value, dist, tol=0.05):
    """Is `value` at the edge of distribution `dist`? → 'low' / 'high' / None."""
    if isinstance(dist, CategoricalDistribution):
        if value == dist.choices[0]:
            return "low"
        if value == dist.choices[-1]:
            return "high"
        return None
    lo, hi = dist.low, dist.high
    if getattr(dist, "log", False):           # evaluate log-scale in log space
        lo, hi, value = math.log(lo), math.log(hi), math.log(value)
    width = hi - lo
    if width <= 0:
        return None
    if (value - lo) <= tol * width:
        return "low"
    if (hi - value) <= tol * width:
        return "high"
    return None


def fmt(value):
    """Format a best-run hyperparameter value compactly for the axis label."""
    if isinstance(value, (list, tuple)):
        return str(list(value))
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != 0 and abs(value) < 0.01:
            return f"{value:.1e}"
        return f"{value:.3g}"
    return str(value)


def _range(dist):
    return list(dist.choices) if isinstance(dist, CategoricalDistribution) else [dist.low, dist.high]


def _inside(value, dist):
    if isinstance(dist, CategoricalDistribution):
        return value in dist.choices
    return dist.low <= value <= dist.high


def render_comparison_table(v1, v2, out_base):
    """Table figure: v1 vs. v2 best hyperparameters and whether widening was used."""
    b1, b2 = v1.best_trial, v2.best_trial
    rows, rowcolors = [], []
    for p in b1.params:
        d1, d2 = b1.distributions[p], b2.distributions[p]
        widened = _range(d1) != _range(d2)
        moved = widened and not _inside(b2.params[p], d1)   # v2 uses newly opened range?
        rows.append([p, fmt(b1.params[p]), fmt(b2.params[p]),
                     "yes" if widened else "no",
                     "YES →" if moved else ("no" if widened else "—")])
        rowcolors.append("#183a2a" if moved else ("#3a3218" if widened else "#141b2b"))

    header = ["Parameter", "v1 best\n(standard)", "v2 best\n(extended)",
              "range\nwidened?", "moved into\nnew range?"]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.4))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    for cell in tbl.get_celld().values():        # dunkles Theme: helle Schrift, dezente Ränder
        cell.set_edgecolor(P.MUTED)
        cell.get_text().set_color(P.INK)
    for j in range(len(header)):
        tbl[(0, j)].set_facecolor("#2a3350")
        tbl[(0, j)].set_text_props(fontweight="bold")
    for i, color in enumerate(rowcolors):
        for j in range(len(header)):
            tbl[(i + 1, j)].set_facecolor(color)
    ax.set_title("PPO best hyperparameters: standard vs. extended search space\n"
                 "(green = value moved into the newly opened range → widening paid off)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_final_improvement(out_base):
    """Bar figure: final 5-seed performance, standard vs. extended search space.

    Fair comparison at the SAME 500k training budget (ppo.csv vs. ppo_v2_500k.csv)
    — isolates the search-space effect. Skipped if the CSVs are missing.
    """
    std_csv = config.METRICS_DIR / "ppo.csv"
    ext_csv = config.METRICS_DIR / "ppo_v2_500k.csv"
    if not (std_csv.exists() and ext_csv.exists()):
        print("skip search-space figure (need ppo.csv and ppo_v2_500k.csv)")
        return
    labels = ["Standard\nsearch space", "Extended\nsearch space"]
    data = [pd.read_csv(std_csv)["mean_reward"].values,
            pd.read_csv(ext_csv)["mean_reward"].values]
    means = [d.mean() for d in data]
    stds = [d.std(ddof=1) for d in data]
    x = np.arange(2)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x, means, 0.55, color=[P.SERIES[1], P.SERIES[2]], alpha=0.9,
           edgecolor=P.INK, linewidth=0.5, zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=P.INK, capsize=5, zorder=4)
    for i, d in enumerate(data):     # individual seeds — honest at n=5
        ax.scatter(np.random.default_rng(i).normal(i, 0.05, size=len(d)), d,
                   color=P.INK, alpha=0.7, s=20, zorder=3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.annotate(f"{m:.0f} ± {s:.0f}", (i, max(m + s, data[i].max())),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")
    ax.annotate(f"+{means[1] - means[0]:.0f}", xy=(1, means[1]),
                xytext=(0.5, max(means) + 42), ha="center",
                color=P.SERIES[2], fontweight="bold", fontsize=13)
    ax.axhline(200, ls="--", color=P.SOLVED, lw=1.5, label="solved (200)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final return (best model, 5 seeds)")
    ax.set_title("PPO: widening the search space improves performance\n(same 500k training budget)",
                 fontweight="bold", fontsize=11)
    ax.margins(y=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    v1 = optuna.load_study(study_name="ppo_lunarlander", storage=f"sqlite:///{config.STUDIES_DIR}/ppo.db")
    v2 = optuna.load_study(study_name="ppo_v2", storage=f"sqlite:///{config.STUDIES_DIR}/ppo_v2.db")

    importances = optuna.importance.get_param_importances(v1)      # {param: importance}, desc
    best = v1.best_trial
    edges = {p: edge_of(best.params[p], best.distributions[p]) for p in importances}

    for p, imp in importances.items():
        mark = {"low": "v lower edge", "high": "^ upper edge", None: ""}[edges[p]]
        print(f"  {p:16s} importance={imp:.3f}  best={best.params[p]}  {mark}")

    # --- figure ---
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.5),
                                   gridspec_kw={"width_ratios": [2, 1]})

    # Panel A: importance (ascending for barh → most important on top), edge params in red.
    # y-labels carry the exact best value found by the study.
    params = list(importances)[::-1]
    vals = [importances[p] for p in params]
    colors = [P.BASELINE_COLOR if edges[p] else P.SERIES[0] for p in params]
    ylabels = [f"{p} = {fmt(best.params[p])}" for p in params]
    axA.barh(range(len(params)), vals, color=colors)
    axA.set_yticks(range(len(params)))
    axA.set_yticklabels(ylabels)
    for i, p in enumerate(params):
        if edges[p]:
            arrow = "↑" if edges[p] == "high" else "↓"
            side = "upper" if edges[p] == "high" else "lower"
            axA.text(vals[i], i, f"  {arrow} {side} edge", va="center", ha="left",
                     color=P.BASELINE_COLOR, fontsize=9, fontweight="bold")
    axA.set_xlabel("Parameter importance (Optuna fANOVA)")
    axA.set_title("A) Diagnosis: important AND at the range edge?\n(label = best value found)")
    axA.margins(x=0.28)

    # Panel B: best objective, standard vs. extended search space
    labels = ["Standard\nrange", "Extended\nrange (v2)"]
    values = [v1.best_value, v2.best_value]
    bars = axB.bar(labels, values, color=[P.SERIES[1], P.SERIES[2]],
                   edgecolor=P.INK, linewidth=0.5)
    for bar, v in zip(bars, values):
        axB.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v:.0f}", ha="center", va="bottom", fontweight="bold")
    axB.annotate(f"+{v2.best_value - v1.best_value:.0f}",
                 xy=(1, values[1]), xytext=(0.5, max(values) * 1.08),
                 ha="center", color=P.SERIES[2], fontweight="bold")
    axB.set_ylabel("Best objective value")
    axB.set_title("B) Consequence: widen the range")
    axB.set_ylim(0, max(values) * 1.2)

    fig.suptitle("Lesson: high importance + best value at the range edge → widen the range and re-tune",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"range_widening_lesson.{ext}", dpi=150)
    plt.close(fig)

    # Separate table: did the best values change after widening the ranges?
    render_comparison_table(v1, v2, OUT / "range_widening_comparison")

    # Final proof: extended vs standard search space at equal 500k budget.
    render_final_improvement(OUT / "search_space_improvement")

    print(f"\nSaved: {OUT/'range_widening_lesson.png'}")
    print(f"Saved: {OUT/'range_widening_comparison.png'}")
    print(f"Saved: {OUT/'search_space_improvement.png'}")


if __name__ == "__main__":
    main()
