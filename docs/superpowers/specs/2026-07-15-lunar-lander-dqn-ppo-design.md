# Lunar Lander: DQN vs. PPO — Design-Spec

**Datum:** 2026-07-15
**Team:** 3 Personen
**Kontext:** Uni-Projekt im RL-Modul (EU4DUAL). Ziel ist ein trainiertes Modell *und* das Verständnis von "was/warum/wie", plus ein Poster (PowerPoint, außerhalb dieses Repos) mit Ergebnissen, Methodik und Signifikanztests.

---

## 1. Ziel & zentrale Fragestellung

Vergleich zweier RL-Ansätze auf demselben Environment:

> **Löst DQN (wert-basiert) oder PPO (policy-gradient) den diskreten Lunar Lander besser — und ist der Unterschied statistisch signifikant?**

Der Signifikanztest sitzt genau auf diesem Vergleich. Beide Algorithmen werden fair per Optuna getunt, dann über mehrere Seeds voll trainiert und statistisch verglichen.

**Inkrementeller Plan:** Erst DQN vollständig durch die Pipeline (Meilenstein 1), dann PPO einstecken. Die Architektur ist so gebaut, dass PPO "Einstecken statt Neubauen" ist.

---

## 2. Environment

**`LunarLander-v3`** (Gymnasium, Box2D):

- **Zustandsraum:** 8-dimensional, **kontinuierlich** (x/y-Position, x/y-Geschwindigkeit, Winkel, Winkelgeschwindigkeit, 2 Bein-Kontakt-Flags).
- **Aktionsraum:** **4 diskret** (nichts / linkes Triebwerk / Haupttriebwerk / rechtes Triebwerk).
- **"Gelöst":** mittlere Rendite ≥ 200 über 100 Episoden.

Didaktischer Kern: Weil die States kontinuierlich sind, ist keine Q-Tabelle möglich → neuronaler Funktionsapproximator → Deep Q-Learning. Schöne Poster-Story: "von tabellarischem Q-Learning zu Deep Q-Learning".

**Warum DQN + diskret (nicht continuous):** DQN wählt `argmax` über eine endliche Aktionsmenge und funktioniert prinzipbedingt nicht mit kontinuierlichen Aktionen. Continuous Lunar Lander bräuchte SAC/TD3/PPO. Diese Entscheidung ist bewusst getroffen; continuous bleibt als möglicher späterer Zusatz (mit SAC) offen.

---

## 3. Implementierungs-Grundlage

**Stable-Baselines3 (SB3)** für beide Algorithmen (DQN und PPO).

Begründung: fairer Vergleich (gleiche Bibliotheksqualität), passt exakt zum Kursmaterial und zum vorhandenen CartPole-Beispiel (`03_SB3_example/`), erprobte Optuna-Integration. Das Verständnis entsteht über **Analyse** (Lernkurven, Ablationen, Hyperparameter-Wirkung), nicht über Neu-Implementierung. Optionaler späterer Bonus: ein kleines from-scratch-DQN-Notebook als didaktische Beilage.

---

## 4. Architektur & Projektstruktur

Kerngedanke: **eine algorithmus-agnostische Pipeline**; DQN ist die erste Instanz, PPO steckt später rein. `agents.py` kapselt den einzigen algorithmus-spezifischen Teil.

```
src/lunarlander/
  config.py       # Env-Name, Seeds, Pfade, Konstanten (eine Quelle der Wahrheit)
  agents.py       # make_agent(algo, hyperparams, seed) -> SB3-Modell (DQN | PPO)
  train.py        # train(algo, hyperparams, seed, timesteps) -> Modell + Logs
  tune.py         # Optuna-Study: sucht beste Hyperparameter für einen Algo
  evaluate.py     # evaluate(model, n_episodes) -> Reward-Array (für Statistik)
  stats.py        # Welch's t-Test, Confidence Intervals, Effektstärke
  plots.py        # Lernkurven, Box-/Violinplots, CI-Plots (fürs Poster)

notebooks/
  01_dqn_explore.ipynb     # erstes DQN, sanity-check, eine Lernkurve
  02_dqn_tuning.ipynb      # Optuna-Ergebnisse für DQN visualisieren
  03_ppo_tuning.ipynb      # dasselbe für PPO (später)
  04_comparison.ipynb      # DQN vs PPO: Statistik + finale Poster-Plots

scripts/
  run_tuning.py            # startet Optuna im Hintergrund, speichert in study.db
  run_final_eval.py        # trainiert beste Configs über alle Seeds, speichert Rewards

results/
  studies/*.db             # Optuna-Studies (wie im CartPole-Beispiel)
  models/                  # gespeicherte beste Modelle
  metrics/*.csv            # Rewards pro Seed → Input für Statistik
```

**Schnittstellen-Verträge (die Isolation, die PPO "reinstecken" ermöglicht):**

- `agents.make_agent(algo, hyperparams, seed)` — einziger Ort, der DQN vs. PPO kennt. Gibt ein SB3-Modell zurück.
- `train.train(algo, hyperparams, seed, timesteps)` — algorithmus-agnostisch, ruft `make_agent`.
- `evaluate.evaluate(model, n_episodes)` — nimmt ein beliebiges SB3-Modell, gibt ein Reward-Array zurück.
- `stats`/`plots` — arbeiten nur auf Reward-Arrays/CSVs, kennen keinen Algorithmus.

Lange Optuna-Läufe laufen als **Skript** (robust, im Hintergrund, speichert in `.db`). Notebooks bleiben **dünn** und zeigen nur Ergebnisse/Plots.

**Dependency-Ergänzungen in `pyproject.toml`:** `gymnasium[box2d]` (für Lunar Lander) und `torch` (via SB3).

---

## 5. Experiment-Design & Statistik

Ablauf für **jeden** Algorithmus identisch.

### Phase 1 — Tuning (pro Algorithmus)
- Optuna-Study, **~30 Trials**, TPE-Sampler + MedianPruner (wie CartPole-Beispiel).
- Jeder Trial: verkürztes Training (~150k Steps), ein Seed, Zielgröße = mittlere Eval-Rendite.
- **Suchraum DQN:** `learning_rate`, `buffer_size`, `batch_size`, `gamma`, `train_freq`, `target_update_interval`, `exploration_fraction`, Netz-Architektur.
- **Suchraum PPO:** `learning_rate`, `n_steps`, `batch_size`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`, `n_epochs`.
- Ergebnis: **eine beste Hyperparameter-Konfiguration** je Algorithmus.

### Phase 2 — Finaler Vergleich
- Beste DQN-Config → volles Training über **5 Seeds** (0–4).
- Beste PPO-Config → volles Training über **dieselben 5 Seeds**.
- Jedes finale Modell auf **100 Eval-Episoden** (deterministisch) → mittlere Rendite pro Seed.
- Ergebnis: zwei Zahlenreihen à 5 Werte (Rendite pro Seed).

### Phase 3 — Statistik
- **Primärmetrik:** finale mittlere Rendite (gelöst = ≥ 200).
- **Welch's t-Test** (ungleiche Varianzen) → p-Wert.
- **95% Confidence Intervals** pro Gruppe + für die Differenz.
- **Effektstärke** (Cohen's d): nicht nur "signifikant", sondern "wie groß".
- **Sekundärmetrik (optional):** Sample-Effizienz — Steps bis ≥ 200 erreicht.

**Bewusste Limitierung:** n=5 pro Gruppe → geringe statistische Power; nur große Effekte zuverlässig nachweisbar. Für ein Uni-Poster akzeptabel und ein guter Diskussionspunkt. Einfachster Hebel für mehr Aussagekraft: Seeds auf 8–10 erhöhen — ohne Code-Änderung möglich.

---

## 6. Richtwerte Rechenbudget (Laptop-CPU, schlanker Umfang)

- Ein DQN-Training bis "gelöst": ~300k–1M Steps ≈ 15–45 Min pro Lauf.
- Optuna 30 Trials (verkürzte Trainings): mehrere Stunden → im Hintergrund als Skript laufen lassen.
- Finaler Vergleich: beste DQN- und PPO-Config je über 5 Seeds voll trainiert.
- Kein Cluster-/GPU-Zugang angenommen.

---

## 7. Poster-Mapping

| Poster-Element | Woher es kommt |
|---|---|
| Problem & MDP | Lunar Lander: 8D kontinuierliche States, 4 diskrete Aktionen, Reward, "gelöst = ≥200" |
| Ansatz | DQN vs. PPO, Begründung der Wahl |
| Methodik | Optuna (30 Trials, TPE+Pruner), 5 Seeds, faire Bedingungen |
| Lernkurven | `plots.py` — Rendite über Steps, beide Algorithmen, CI-Band über Seeds |
| Hauptergebnis | Box/Violinplot finale Rendite + Welch's t-Test (p), 95% CI, Cohen's d |
| Sample-Effizienz | Steps bis ≥200 (Sekundärmetrik) |
| Diskussion | Limitierungen (n=5, CPU), warum was funktioniert |
| Demo | GIF einer gelandeten Episode (`imageio` in Deps) |

---

## 8. Aufgabenteilung (3 Personen)

- **Phase 0 (gemeinsam):** Pipeline-Gerüst (`agents/train/evaluate/stats`) + erstes DQN läuft. Basis für alle.
- Danach parallel:
  - **Person 1 — DQN-Owner:** Optuna-Study DQN, beste Config, Lernkurven-Analyse.
  - **Person 2 — PPO-Owner:** PPO einstecken (sobald Pipeline steht), Optuna-Study PPO.
  - **Person 3 — Eval & Statistik:** `stats.py` + `plots.py`, finaler Vergleich, Signifikanztests, Poster-Grafiken.

**Meilenstein 1 = DQN End-to-End** (Tuning → Eval → ein Plot). Erst danach kommt PPO. So gibt es früh etwas Vorzeigbares.

---

## 9. Erfolgskriterien

1. DQN läuft End-to-End durch die Pipeline und erreicht auf mind. einem Seed eine Rendite nahe/über 200.
2. Optuna-Studies für DQN und PPO liefern reproduzierbar beste Konfigurationen (gespeichert in `.db`).
3. Finaler Vergleich über 5 Seeds mit Welch's t-Test, 95% CI und Cohen's d liegt vor.
4. Poster-fertige Grafiken (Lernkurven, Box-/Violinplot, Demo-GIF) sind erzeugt.
5. Alles reproduzierbar (fixe Seeds, Skripte statt reiner Notebook-Ausführung).

---

## 10. Bewusst außerhalb des Scopes (YAGNI)

- Continuous Lunar Lander / SAC/TD3 (möglicher späterer Zusatz).
- From-scratch-Implementierung der Algorithmen (optionale didaktische Beilage).
- Weitere Algorithmen (A2C etc.).
- Hochskalieren auf viele Seeds / Cluster — Pipeline erlaubt es, aber nicht im Kern-Scope.
