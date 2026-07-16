# CLAUDE.md

Guidance for AI sessions working in this repo.

## Was dieses Repo ist
Eine Lunar-Lander-Pipeline, die **DQN vs. PPO** auf `LunarLander-v3` vergleicht
(Tuning → Multi-Seed-Training → Signifikanztest, fürs Poster). Entstanden aus einem
RL-Kurs-Repo, jetzt eigenständig unter `FabianGlueck/lunar-lander-dqn-vs-ppo` —
das Kursmaterial (PDFs, Flappy-Bird u. a.) ist hier **nicht** mehr enthalten.

## Kommunikation
Der User schreibt Deutsch → **auf Deutsch antworten**. Code-Kommentare/Docstrings ebenfalls Deutsch.

## Umgebung & Kommandos
Immer **`uv`** (nicht pip/venv direkt), Python 3.12.
```bash
uv sync                                              # Abhängigkeiten
uv run pytest -q                                     # Tests
uv run python -m scripts.run_tuning --algo dqn --trials 30
uv run python -m scripts.run_final_eval --algo dqn   # best_model (Default), --eval-only, --which {best,final}
uv run tensorboard --logdir results/tb               # finale Läufe live
uv run python -m scripts.render_demo --open          # Demo-Seite: 10 Landungen je Policy → docs/demo/
```

## Architektur (wichtig)
Die Pipeline ist **algorithmus-agnostisch**. `src/lunarlander/agents.py` (`make_agent`/`load_agent`)
ist die **einzige** Stelle, die DQN/PPO kennt. **Keine Algo-Verzweigungen** in train/evaluate/tune/stats/plots
einbauen — dort bleibt alles algo-blind, damit ein weiterer Algorithmus nur „reingesteckt" wird.

Module: `config` (Konstanten/Pfade), `envs` (`make_env`), `agents`, `train`, `evaluate`, `stats`
(Welch/CIs/Cohen's d), `plots`, `tune` (Optuna). `scripts/` = CLI für lange Läufe.
`results/` = Laufergebnisse (**gitignored**), `docs/figures/` + `docs/demo/` = eingecheckte Artefakte.

## So arbeiten wir
- **TDD immer** (superpowers:test-driven-development): erst den fehlschlagenden Test schreiben, RED sehen,
  minimal GREEN, verifizieren. Für neue Features/Bugfixes.
- **Verifizieren vor „fertig":** Kommando ausführen und Output zeigen, nicht behaupten.
- **Kleine, fokussierte Commits**, deutsche Commit-Bodies ok. Commit-Message endet mit dem
  `Co-Authored-By: Claude ...`-Trailer. Nur committen, wenn sinnvoll/gewünscht.
- **Es gibt nur `main`** — direkt darauf arbeiten. Pushen oder einen PR erstellen nur nach Nachfrage.
- **Laufendes Training nicht stören:** keine schweren Trainings/große pytest-Läufe starten, während der
  User einen echten Lauf (CPU) fahren hat.

## Tests
Jedes Modul hat eine Testdatei unter `tests/`. **Tests dürfen kein langes Training fahren** — Test-Trainings
nutzen winzige Timesteps (≤ 2000), laufen in Sekunden. Bekanntes, harmloses Rauschen (nicht jagen):
- macOS `objc[...] SDL2 implemented in both cv2 and pygame` (nur beim Import)
- Optuna `net_arch ... is of type list` (Liste statt Tuple; benigne)

## Doku-Landkarte
- `docs/notes.md` — **lebende** Konzept-/Entscheidungs-Doku. Wenn ein Konzept geklärt wird → hier ergänzen.
- `docs/poster-content.md` — paste-fertige Poster-Texte (Englisch) + Figuren-Zuordnung.
- `docs/superpowers/specs/` & `plans/` — Design-Spec und Implementierungsplan.
- `src/lunarlander/README.md` — Pipeline-Nutzung/Workflow.
- Root `README.md` — die Projekt-README. Darf gepflegt werden, aber nicht ungefragt umschreiben.

## Zentrale Entscheidungen (Details in docs/notes.md)
- **Metrik = `best_model`** (Early Stopping), nicht das finale Modell — DQN neigt am Ende zu
  catastrophic forgetting. `run_final_eval --which best` (Default).
- **Mehrere Seeds** (`config.SEEDS = [0..4]`), dieselben für DQN & PPO → fairer Signifikanztest.
- Pfade in `config.py` sind an den Repo-Root verankert (funktionieren auch aus Notebooks).
- `best_model.zip`/`final_model.zip`/`evaluations.npz` liegen in `results/models/<algo>_seed<n>/`.
