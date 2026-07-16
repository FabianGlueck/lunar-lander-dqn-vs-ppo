# Poster Content — DQN vs. PPO for Lunar Lander

Paste-fertige Kurztexte fürs Poster (Englisch). Zahlen aus 5 Seeds, 100 Eval-Episoden,
bestes Modell (Early Stopping). Figuren siehe unten.

**Title:** Comparative Analysis of Deep RL Algorithms: DQN vs. PPO for Lunar Lander Control
**Authors:** Fabian Glück, Hendrick Fischer, Stefanie Kirchner
**Program:** EU4DUAL Master Class AI · Mondragon Unibertsitatea · DHBW CAS Heilbronn

---

## ABSTRACT

We compare two Deep RL algorithms on `LunarLander-v3` (8D continuous state, 4 discrete
actions): **DQN** (value-based) vs. **PPO** (policy-gradient). Both are tuned with Optuna and
run over 5 seeds. Both solve the task (return > 200) with **no significant difference in final
score**, but **PPO is more sample-efficient and far more stable**, while DQN collapses late in
training (catastrophic forgetting) — so best-checkpoint selection is essential.

---

## METHODS

- **Environment:** `LunarLander-v3` (Box2D). **Solved = mean return ≥ 200** over 100 episodes.
  Continuous state → neural-network approximator (no tabular Q-table).
- **DQN:** Q-network + target network + experience replay (off-policy).
- **PPO:** actor-critic with clipped surrogate objective (on-policy).
- **Training:** Optuna (30 trials, TPE + pruner) at 150k steps → final **500k steps × 5 seeds**,
  laptop CPU. Metric = **best checkpoint** (early stopping).

**Tuned hyperparameters:**

| Parameter | DQN | PPO |
|---|---|---|
| learning rate | 9.0 × 10⁻⁴ | 6.1 × 10⁻⁴ |
| discount γ | 0.982 | 0.994 |
| batch size | 256 | 128 |
| network (hidden) | [256, 256] | [128, 128] |
| target update / expl. frac. | 250 / 0.18 | — |
| clip / n_epochs / GAE λ | — | 0.12 / 20 / 0.92 |

---

## RESULTS

| Metric | DQN | PPO |
|---|---|---|
| best model (mean ± std) | 236 ± 10 | 240 ± 18 |
| seeds solved (≥ 200) | 5 / 5 | 5 / 5 |
| final model (end of training) | 152 ± 65 (1/5) | 219 ± 39 (4/5) |
| steps to reach mean ≥ 200 | ≈ 220k | ≈ 180k |

- **Significance (best model):** Welch p = 0.68, Cohen's d = −0.27, 95% CI (−26, +19);
  Mann-Whitney-U p = 1.00 → **no significant difference in final score.**
- **Stability:** best → final: DQN 236 → 152, PPO 240 → 219. Over 10 runs: DQN 219 ± 97
  (2 crashes), PPO 267 ± 13 (none) → **PPO clearly more reliable.**
- **Learning curves** (main figure): both climb from ≈ −120 to ≈ 200–250; DQN wobbles and
  several seeds collapse late, PPO rises smoothly.

---

## LEARNINGS

- **Equal final performance** — difference not significant (p = 0.68).
- **PPO is more sample-efficient** (≈ 180k vs. 220k steps) **and more stable.**
- **DQN → catastrophic forgetting:** final model far worse than best checkpoint →
  **early stopping essential.**
- **Why?** DQN's "deadly triad" + unconstrained `argmax` → sudden collapses; PPO's **clipping
  limits each policy update** → small, safe steps.
- **Search space > compute:** key PPO params sat at their range edges; widening ranges lifted
  PPO **240 → 254** at the *same* 500k budget (see PPO v2).

**Reading significance (n = 5):** low statistical power — *non-significant ≠ equal*, only
"no difference detectable". We therefore also report effect size (d) and CI, not just p.

---

## CONCLUSION

Both methods solve Lunar Lander; final scores are statistically indistinguishable, but **PPO
is more efficient and more stable.** Classic trade-off: DQN's off-policy value learning is
unstable (deadly triad, overestimation), PPO trades flexibility for **stability via clipping**.

**PPO v2 — search-space lesson.** Optuna importance showed key params at their range edges.
Widening the ranges and re-tuning (same 500k budget) improved PPO:

| PPO (best model, 5 seeds) | Return |
|---|---|
| standard search space (v1) | 240.1 ± 18.2 |
| **extended search space (v2)** | **253.9 ± 22.0** (+13.9) |

`n_epochs` 20 → 30 and `net_arch` [128,128] → [256,256] moved into the new region. Medium
effect (d = 0.69), **not significant at n = 5** → gain from the *search space*, not more compute.

**Outlook:** more seeds, Double DQN, reward shaping, robustness to wind (PPO 262 → 183 at
wind_power 15).

---

## Figuren-Platzierung (Grafik → Poster-Kasten)

| Poster-Kasten | Grafik | Datei |
|---|---|---|
| RESULTS (großes Bild) | Learning Curves DQN vs. PPO | `learning_curves.pdf/png` |
| LEARNINGS (Kasten 1) | Boxplot mit Einzel-Seeds | `comparison_boxplot.pdf/png` |
| LEARNINGS (Kasten 2) | best- vs. final-Modell (Gap) | `gap_best_vs_final.pdf/png` |
| RESULTS/METHODS (klein) | Sample-Effizienz (Balken) | `efficiency.pdf/png` |
| optional | Reward-Histogramme | `hist_dqn.pdf/png`, `hist_ppo.pdf/png` |
| CONCLUSION / OUTLOOK | Suchraum-Erweiterung → +14 (500k) | `search_space_improvement.pdf/png` |
| CONCLUSION / OUTLOOK | Range-Widening-Lektion | `range_widening_lesson.pdf/png` |
| CONCLUSION (Detail) | v1-vs-v2 beste Params | `range_widening_comparison.pdf/png` |
| CONCLUSION (Kurve) | Learning Curve inkl. v2 | `learning_curves_with_v2.pdf/png` |
| CONCLUSION (Bild) | Landing-Frame (bestes Modell) | `landing.png` (+ `landing.gif`) |

Figuren in **`docs/figures/`** (PDF = Druck, PNG = PowerPoint). Neu erzeugen über
`notebooks/03_analysis.ipynb`.
