"""Poster-Grafiken im Lunar-Lander-Theme (dunkler Weltraum-Look).

Nutzt das headless `Agg`-Backend, damit Plots ohne Display (Skript/CI) speicherbar
sind. Beim Import wird ein dunkles Theme gesetzt (tiefer Weltraum-Hintergrund, helle
Schrift). Die Serien-Palette (Teal/Lila/Amber) ist mit dem dataviz-Validator geprüft
(colorblind-safe auf dunkler Fläche). Jede Funktion schreibt eine PNG und gibt den Pfad zurück.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, kein Display nötig
import matplotlib.pyplot as plt
import numpy as np

# --- Lunar-Lander-Theme (Palette geprüft: colorblind-safe, OKLCH-Lightness-Band, Kontrast) ---
SURFACE = "#0c1018"       # tiefer Weltraum
INK = "#e7eaf3"           # helle Schrift/Achsen (Terrain-weiß)
MUTED = "#3a4358"         # Achsenrahmen / Nulllinie
GRID = "#1e2637"          # dezentes Gitter
SERIES = ["#0a9fb0", "#7d54d6", "#b57d18"]   # DQN Teal · PPO Lila · 3. Amber
SOLVED = "#d3d9e6"        # helle gestrichelte Ziel-Linie (200)
BASELINE_COLOR = "#e0685a"  # gedämpftes Rot für den Zufalls-Boden

plt.rcParams.update({
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.5, "grid.linewidth": 0.8,
    "axes.prop_cycle": plt.cycler(color=SERIES),
    "legend.facecolor": "#141b2b", "legend.edgecolor": MUTED, "legend.framealpha": 0.9,
    "savefig.dpi": 150,
})


def _build_learning_curve(history_paths, baseline=None, mark_best_checkpoints=False):
    """Baut die Lernkurven-Figur (ohne zu speichern) und gibt (fig, ax) zurück.

    Als eigene Funktion herausgezogen, damit sich die Achse in Tests inspizieren
    lässt; `learning_curve` kümmert sich ums Speichern.

    Args:
        history_paths: Dict `{algo: [pfad_zu_evaluations.npz, ...]}` — je Seed eine .npz.
        baseline: optionaler y-Wert für eine gestrichelte Referenzlinie
            (z. B. der gemessene Zufalls-Boden). None = keine Linie.
        mark_best_checkpoints: Markiert für jeden Seed den Evaluationspunkt, an
            dem die gemittelte Kurve ihr Maximum erreicht.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for algo, paths in history_paths.items():
        # Alle Seeds auf gemeinsame Timesteps (erste Datei) mitteln.
        first = np.load(paths[0])
        timesteps = first["timesteps"]
        per_seed_means = []
        for p in paths:
            data = np.load(p)
            per_seed_means.append(data["results"].mean(axis=1))  # Mittel je Eval-Zeitpunkt
        stacked = np.vstack(per_seed_means)
        mean = stacked.mean(axis=0)   # Mittel über die Seeds
        std = stacked.std(axis=0)     # Streuung über die Seeds (Unsicherheitsband)
        line, = ax.plot(timesteps, mean, lw=2, label=algo.upper())
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.18,
                        color=line.get_color())   # Band in Linienfarbe
        if mark_best_checkpoints:
            best_index = mean.argmax()
            ax.scatter(timesteps[best_index], mean[best_index],
                       marker="X", s=60, color=line.get_color(), edgecolor=INK,
                       linewidth=0.7, zorder=4,
                       label=f"{algo.upper()} best model")
    ax.axhline(200, ls="--", color=SOLVED, lw=1.5, label="solved (200)")
    if baseline is not None:
        ax.axhline(baseline, ls="--", color=BASELINE_COLOR, lw=1.5,
                   label=f"random baseline ({baseline:.0f})")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean eval return")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def learning_curve(history_paths, out_path, baseline=None, mark_best_checkpoints=False):
    """Zeichnet die mittlere Eval-Rendite über die Steps (±Std-Band über Seeds).

    Args wie `_build_learning_curve`, plus `out_path` (Ziel-PNG).

    Returns:
        Path der gespeicherten PNG.
    """
    out_path = Path(out_path)
    fig, _ = _build_learning_curve(history_paths, baseline=baseline,
                                   mark_best_checkpoints=mark_best_checkpoints)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def comparison_plot(rewards_by_algo, out_path, baseline=None, show_points=False):
    """Boxplot der finalen Renditen je Algorithmus (die Poster-Hauptgrafik).

    Args:
        rewards_by_algo: Dict `{algo: rewards_array}` (ein Wert je Seed).
        out_path: Ziel-PNG.
        baseline: optionaler y-Wert für eine gestrichelte Referenzlinie (z. B. Zufalls-Boden).
        show_points: bei True die einzelnen Seed-Punkte über den Box legen
            (ehrlicher bei kleinem n — man sieht die tatsächlichen Werte).

    Returns:
        Path der gespeicherten PNG.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(rewards_by_algo.keys())
    data = [np.asarray(rewards_by_algo[k], float) for k in labels]
    bp = ax.boxplot(data, patch_artist=True)          # patch_artist → Boxen einfärbbar
    for patch, color in zip(bp["boxes"], SERIES):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(INK)
    for element in ("whiskers", "caps", "medians"):   # sonst schwarz → unsichtbar auf dunkel
        for line in bp[element]:
            line.set_color(INK)
    ax.set_xticklabels([l.upper() for l in labels])
    if show_points:
        for i, d in enumerate(data, start=1):
            jitter = np.random.default_rng(i).normal(i, 0.04, size=len(d))  # leicht streuen
            ax.scatter(jitter, d, color=INK, alpha=0.8, s=25, zorder=3)
    ax.axhline(200, ls="--", color=SOLVED, lw=1.5, label="solved (200)")
    if baseline is not None:
        ax.axhline(baseline, ls="--", color=BASELINE_COLOR, lw=1.5,
                   label=f"random baseline ({baseline:.0f})")
    ax.set_ylabel("Final return (100 episodes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def efficiency_plot(steps_by_algo, out_path):
    """Balkendiagramm der Sample-Effizienz: Steps bis Rendite ≥ 200 je Algorithmus.

    Args:
        steps_by_algo: Dict `{algo: steps_int_oder_None}` (None = nie erreicht).
        out_path: Ziel-PNG.
    """
    out_path = Path(out_path)
    labels = list(steps_by_algo.keys())
    values = [(steps_by_algo[k] or 0) for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([l.upper() for l in labels], values,
                  color=SERIES[:len(labels)], edgecolor=INK, linewidth=0.5)
    for k, bar in zip(labels, bars):
        txt = "never" if steps_by_algo[k] is None else f"{steps_by_algo[k]:,}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), txt,
                ha="center", va="bottom")
    ax.set_ylabel("Steps to return ≥ 200")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def gap_plot(best_by_algo, final_by_algo, out_path):
    """Gruppierte Balken: bestes vs. finales Modell je Algorithmus.

    Zeigt den Stabilitäts-/Forgetting-Effekt (großer Gap = starkes Vergessen).

    Args:
        best_by_algo, final_by_algo: Dicts `{algo: mittlere_rendite}`.
        out_path: Ziel-PNG.
    """
    out_path = Path(out_path)
    labels = list(best_by_algo.keys())
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, [best_by_algo[k] for k in labels], w, label="best model",
           color=SERIES[0], edgecolor=INK, linewidth=0.5)
    ax.bar(x + w / 2, [final_by_algo[k] for k in labels], w, label="final model",
           color=SERIES[1], edgecolor=INK, linewidth=0.5)
    ax.axhline(200, ls="--", color=SOLVED, lw=1.5, label="solved (200)")
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in labels])
    ax.set_ylabel("Mean return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _build_mean_runs(rewards_by_label):
    """Baut die Balkenfigur „Mittelwert über N Durchläufe" (ohne zu speichern).

    Als eigene Funktion herausgezogen, damit sich die Achse in Tests inspizieren
    lässt; `mean_runs_plot` kümmert sich ums Speichern.

    Args:
        rewards_by_label: Dict `{label: rewards_array}` mit den Renditen der N
            Episoden je Eintrag. Die Reihenfolge bestimmt Position und Farbe —
            eine Zufalls-Baseline gehört ans Ende und bekommt damit Rot.

    Returns:
        (fig, ax)
    """
    labels = list(rewards_by_label)
    data = [np.asarray(rewards_by_label[k], float) for k in labels]
    means = np.array([d.mean() for d in data])
    stds = np.array([d.std() for d in data])
    x = np.arange(len(labels))
    # Positionsbasiert statt nach Namen — plots.py bleibt so algo-blind.
    colors = [SERIES[0], SERIES[1], BASELINE_COLOR]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(x, means, 0.55, color=[colors[i % len(colors)] for i in x],
                  alpha=0.9, edgecolor=INK, linewidth=0.5, zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=INK, capsize=5, zorder=4)
    # Rohwerte über die Balken: bei kleinem N sagt der Mittelwert allein zu wenig.
    for i, d in enumerate(data):
        jitter = np.random.default_rng(i).normal(i, 0.05, size=len(d))
        ax.scatter(jitter, d, color=INK, alpha=0.7, s=20, zorder=3)
    # Direkte Wertlabels (statt sie nur aus der Achse abzulesen). Sie sitzen
    # jenseits der ganzen Säule — am Balkenende lägen sie auf Errorbar und Punkten.
    for bar, m, s, d in zip(bars, means, stds, data):
        oben = m >= 0
        y = max(m + s, d.max()) if oben else min(m - s, d.min())
        ax.annotate(f"{m:.0f} ± {s:.0f}",
                    (bar.get_x() + bar.get_width() / 2, y),
                    textcoords="offset points", xytext=(0, 8 if oben else -16),
                    ha="center", fontweight="bold")
    ax.margins(y=0.18)  # Luft, damit die Labels nicht am Rand abgeschnitten werden
    ax.axhline(200, ls="--", color=SOLVED, label="solved (200)", zorder=1)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in labels])
    ax.set_ylabel("Return (mean ± std)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def mean_runs_plot(rewards_by_label, out_path):
    """Balken je Eintrag: Mittelwert ± Std über N Durchläufe, mit den Einzelwerten.

    Die Poster-Grafik zu „G": zeigt den Mittelwert, seine Streuung *und* die
    einzelnen Episoden — inklusive Zufalls-Baseline als Vergleichsboden.

    Args:
        rewards_by_label: wie `_build_mean_runs`.
        out_path: Ziel-PNG.

    Returns:
        Path der gespeicherten PNG.
    """
    out_path = Path(out_path)
    fig, _ = _build_mean_runs(rewards_by_label)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def reward_histogram(rewards, out_path, label=None):
    """Histogramm der Episoden-Renditen (Konsistenz einer Policy).

    Args:
        rewards: Array der Renditen (z. B. 100 Eval-Episoden eines Seeds).
        out_path: Ziel-PNG.
        label: optionaler Titel.
    """
    out_path = Path(out_path)
    r = np.asarray(rewards, float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r, bins=20, color=SERIES[0], alpha=0.85, edgecolor=INK, linewidth=0.4)
    ax.axvline(200, ls="--", color=SOLVED, lw=1.5, label="solved (200)")
    ax.set_xlabel("Episode return")
    ax.set_ylabel("Frequency")
    if label:
        ax.set_title(label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
