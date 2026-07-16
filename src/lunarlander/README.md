# Lunar Lander: DQN vs. PPO — Pipeline

Eine kleine, algorithmus-agnostische Pipeline auf Basis von **Stable-Baselines3**,
um **DQN** und **PPO** auf dem diskreten `LunarLander-v3` zu tunen, über mehrere
Seeds zu trainieren und **statistisch zu vergleichen** (für Poster + Signifikanztests).

> Kernidee: Nur `agents.make_agent` kennt den Unterschied DQN/PPO. Alles andere
> (Training, Evaluation, Tuning, Statistik, Plots) ist algorithmus-blind — ein
> weiterer Algorithmus lässt sich daher "einstecken", ohne den Rest zu ändern.

---

## Aufbau

```
src/lunarlander/
  config.py     # Konstanten & Pfade (eine Quelle der Wahrheit)
  envs.py       # make_env(seed, render_mode) -> geseedete LunarLander-Env
  agents.py     # make_agent(algo, hyperparams, env, seed) -> SB3-Modell  ← einzige algo-spezifische Stelle
  train.py      # train(...) -> (model, evaluations.npz)  mit periodischem Eval-Logging
  evaluate.py   # evaluate(model, seed, n_episodes) -> Reward-Array pro Episode
  stats.py      # welch_t_test, confidence_interval(_diff), cohens_d, summarize
  plots.py      # learning_curve (optional baseline-Linie), comparison_plot
  tune.py       # Optuna: Suchräume, TrialEvalCallback (Pruning), run_study
scripts/
  run_tuning.py       # Optuna-Study starten (schreibt results/studies/<algo>.db)
  run_final_eval.py   # beste Config über alle Seeds trainieren (schreibt results/metrics/<algo>.csv)
notebooks/
  playground.ipynb    # Pipeline in Minuten antesten (Smoke-Test, GIF, Zeit-Hochrechnung)
  02_final_run.ipynb  # getunte Params + Optimization-History + finale Ergebnisse/Vergleich
results/                # Ausgaben (gitignored): studies/, models/, metrics/
```

## Setup

```bash
uv sync                     # Abhängigkeiten installieren (inkl. gymnasium[box2d], torch, SB3, optuna)
uv run pytest -q            # alle Tests (sollten grün sein)
```

Falls der Box2D-Build hakt: `brew install swig` und `uv sync` erneut.

---

## Workflow

**1. Erst antesten (Minuten):** `notebooks/playground.ipynb` öffnen und "Run All".
Zeigt, dass alles läuft, misst Steps/s und rechnet hoch, wie lange der echte Lauf dauert.

```bash
uv run jupyter lab
```

**2. Tuning (pro Algorithmus):**

```bash
uv run python -m scripts.run_tuning --algo dqn --trials 30
uv run python -m scripts.run_tuning --algo ppo --trials 30
```

Läuft in eine SQLite-DB (`results/studies/<algo>.db`), ist fortsetzbar, und der
`MedianPruner` bricht aussichtslose Trials früh ab. Tipp: denselben Befehl in
mehreren Terminals starten → echte Parallelität über dieselbe DB.

**3. Finaler Mehr-Seed-Lauf** (liest die besten Params automatisch aus der Study):

```bash
uv run python -m scripts.run_final_eval --algo dqn
uv run python -m scripts.run_final_eval --algo ppo
```

Kein Abtippen nötig — ohne `--params` holt sich das Skript `study.best_params`
aus `results/studies/<algo>.db`. Mit `--params '{...}'` kann man eine
Konfiguration manuell überschreiben.

Bewertet wird standardmäßig das **beste** Modell (`--which best`, Early Stopping
über den vom EvalCallback gesicherten besten Checkpoint) — bei DQN wegen möglicher
Instabilität am Trainingsende die fairere, stabilere Metrik. `--which final`
bewertet stattdessen den Stand am Trainingsende. Mit `--eval-only` werden
vorhandene Modelle **ohne erneutes Training** neu bewertet (z. B. um zwischen
`best` und `final` zu wechseln).

Das Notebook `notebooks/02_final_run.ipynb` zeigt die getunten Params, die
Optimization-History und (nach den Läufen) die finalen Ergebnisse an einem Ort.

Schreibt `results/metrics/<algo>.csv` mit einer finalen Rendite pro Seed und
TensorBoard-Logs nach `results/tb/<algo>_seed<n>/`. Zum Live-Beobachten (zweites
Terminal, vor/während des Laufs):

```bash
uv run tensorboard --logdir results/tb      # -> http://localhost:6006
```

Alle Läufe liegen im selben `--logdir`, also lassen sich DQN vs. PPO direkt
überlagern (`rollout/ep_rew_mean`, `eval/mean_reward`, `train/loss`, bei DQN
`rollout/exploration_rate`).

**4. Vergleich & Statistik:** die beiden CSVs laden und
`stats.summarize(dqn, ppo, "dqn", "ppo")` aufrufen → p-Wert (Welch), Konfidenz-
intervalle je Gruppe und für die Differenz, Cohen's d. Grafiken via
`plots.comparison_plot` und `plots.learning_curve`.

---

## Wichtige Einstellungen (`config.py`)

| Konstante | Bedeutung | Default |
|---|---|---|
| `ENV_ID` | Environment | `LunarLander-v3` |
| `SEEDS` | Seeds für den Endvergleich | `[0..4]` |
| `SOLVED_THRESHOLD` | "gelöst" ab dieser Rendite | `200` |
| `TUNE_TIMESTEPS` | Steps pro Optuna-Trial | `150_000` |
| `FINAL_TIMESTEPS` | Steps im finalen Lauf | `500_000` |
| `N_EVAL_EPISODES` | Episoden für finale Bewertung | `100` |

Mehr statistische Power? `SEEDS` auf z. B. `[0..9]` erweitern — die Pipeline
verarbeitet das ohne Code-Änderung.

## Tests

```bash
uv run pytest -q
```

Jedes Modul hat eine eigene Testdatei unter `tests/`. Die Tests nutzen sehr kurze
Trainings (≤ 2000 Steps), laufen also in Sekunden.
