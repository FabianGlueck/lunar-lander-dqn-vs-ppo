"""Statistik für den Signifikanzvergleich zweier Algorithmen.

Alle Funktionen sind reine Zahlen-Funktionen (kein RL, keine Environments) und
arbeiten auf Arrays von finalen Renditen — je ein Wert pro Seed.
"""

import numpy as np
from scipy import stats as sp


def welch_t_test(a, b):
    """Welch's t-Test (ungleiche Varianzen) auf zwei Stichproben.

    Returns:
        (t-Statistik, p-Wert). Kleiner p-Wert = Unterschied unwahrscheinlich zufällig.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = sp.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def confidence_interval(data, confidence=0.95):
    """t-basiertes Konfidenzintervall für den Mittelwert einer Stichprobe.

    Returns:
        (mittelwert, untere_grenze, obere_grenze). Bei n < 2 kollabiert das
        Intervall zum Mittelwert (keine Streuung schätzbar).
    """
    a = np.asarray(data, float)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return mean, mean, mean
    se = sp.sem(a)                                   # Standardfehler des Mittelwerts
    h = se * sp.t.ppf((1 + confidence) / 2, n - 1)   # halbe Intervallbreite
    return mean, float(mean - h), float(mean + h)


def cohens_d(a, b):
    """Effektstärke (Cohen's d) mit gepoolter Standardabweichung.

    Sagt, *wie groß* der Unterschied ist (nicht nur ob signifikant):
    ~0.2 klein, ~0.5 mittel, ~0.8+ groß.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def confidence_interval_diff(a, b, confidence=0.95):
    """Welch-Konfidenzintervall für die *Differenz* der Mittelwerte (a − b).

    Das ist das Intervall für den eigentlichen Effekt im Vergleich. Schließt es
    die 0 nicht ein, ist der Unterschied auf dem gewählten Niveau signifikant.

    Returns:
        (differenz, untere_grenze, obere_grenze).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    diff = float(a.mean() - b.mean())
    if na < 2 or nb < 2:
        return diff, diff, diff
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return diff, diff, diff
    # Welch–Satterthwaite-Freiheitsgrade (für ungleiche Varianzen/Größen)
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    h = se * sp.t.ppf((1 + confidence) / 2, df)
    return diff, float(diff - h), float(diff + h)


def mann_whitney(a, b):
    """Mann-Whitney-U-Test (nicht-parametrisch, zweiseitig).

    Robuste Alternative zum t-Test: setzt keine Normalverteilung voraus — sinnvoll
    bei kleinen Stichproben (n=5), wo die Normalannahme des Welch-Tests fragwürdig ist.

    Returns:
        (U-Statistik, p-Wert).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    u, p = sp.mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)


def steps_to_threshold(timesteps, mean_rewards, threshold=200):
    """Erster Timestep, bei dem die mittlere Rendite die Schwelle erreicht.

    Sample-Effizienz-Maß: „wann wird zum ersten Mal ≥ threshold erreicht?".

    Args:
        timesteps: Array der Bewertungs-Zeitpunkte.
        mean_rewards: zugehörige mittlere Renditen (gleiche Länge).
        threshold: Schwelle (Default 200 = „gelöst").

    Returns:
        Der Timestep als int, oder None wenn die Schwelle nie erreicht wird.
    """
    ts = np.asarray(timesteps)
    mr = np.asarray(mean_rewards, float)
    reached = mr >= threshold
    if not reached.any():
        return None
    return int(ts[np.argmax(reached)])   # argmax = erster True-Index


def describe(rewards, threshold=200):
    """Kennzahlen einer Rendite-Stichprobe (ein Wert je Seed) für die Ergebnistabelle.

    Returns:
        Dict mit mean, std, ci_low, ci_high, min, max, n_solved (Seeds ≥ threshold)
        und pct_solved (Prozent).
    """
    r = np.asarray(rewards, float)
    mean, low, high = confidence_interval(r)
    n_solved = int((r >= threshold).sum())
    return {
        "mean": mean,
        "std": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
        "ci_low": low,
        "ci_high": high,
        "min": float(r.min()),
        "max": float(r.max()),
        "n_solved": n_solved,
        "pct_solved": 100.0 * n_solved / len(r),
    }


def summarize(a, b, label_a, label_b):
    """Bündelt alle Kennzahlen fürs Poster in ein Dict.

    Args:
        a, b: Rendite-Arrays der beiden Algorithmen (ein Wert je Seed).
        label_a, label_b: Kurznamen, z. B. "dqn"/"ppo" (bilden die Dict-Schlüssel).

    Returns:
        Dict mit `mean_<label>`, `ci_<label>` (pro Gruppe), `p_value`, `cohens_d`
        und `ci_diff` (Konfidenzintervall der Differenz).
    """
    _, p = welch_t_test(a, b)
    return {
        f"mean_{label_a}": float(np.mean(a)),
        f"mean_{label_b}": float(np.mean(b)),
        f"ci_{label_a}": confidence_interval(a),
        f"ci_{label_b}": confidence_interval(b),
        "p_value": p,
        "cohens_d": cohens_d(a, b),
        "ci_diff": confidence_interval_diff(a, b),
    }
