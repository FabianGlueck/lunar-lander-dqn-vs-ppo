"""Zentrale Konfiguration der Pipeline.

Einzige Quelle der Wahrheit: Environment-Name, Seeds, Trainings-/Evaluations-
Längen und alle Ausgabepfade. Jedes andere Modul liest seine Konstanten von hier,
damit ein Wert nur an *einer* Stelle geändert werden muss.
"""

from pathlib import Path

# --- Environment ---------------------------------------------------------
ENV_ID = "LunarLander-v3"      # diskrete Variante: 8D-Zustand, 4 Aktionen
SEEDS = [0, 1, 2, 3, 4]        # feste Seeds für den reproduzierbaren Endvergleich
SOLVED_THRESHOLD = 200         # mittlere Rendite ab der Lunar Lander als "gelöst" gilt

# --- Evaluation ----------------------------------------------------------
N_EVAL_EPISODES = 100          # Episoden für die finale Bewertung eines Modells
CALLBACK_EVAL_EPISODES = 20    # Episoden für die periodische Bewertung im Training
EVAL_FREQ = 10_000             # Steps zwischen zwei periodischen Bewertungen

# --- Trainingslängen -----------------------------------------------------
TUNE_TIMESTEPS = 150_000       # verkürztes Training pro Optuna-Trial (schnell)
FINAL_TIMESTEPS = 500_000      # volles Training für den finalen Vergleich

# --- Pfade ---------------------------------------------------------------
# An den Repo-Root verankert, damit sie unabhängig vom Arbeitsverzeichnis
# stimmen (z. B. auch aus einem Notebook heraus, das in notebooks/ läuft).
_REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = _REPO_ROOT / "results"     # Laufergebnisse — gitignored
STUDIES_DIR = RESULTS_DIR / "studies"    # Optuna-SQLite-Studies (.db)
MODELS_DIR = RESULTS_DIR / "models"      # gespeicherte Modelle + Eval-Logs
METRICS_DIR = RESULTS_DIR / "metrics"    # finale Rewards pro Seed (.csv)
TB_DIR = RESULTS_DIR / "tb"              # TensorBoard-Logs der finalen Läufe
DOCS_DIR = _REPO_ROOT / "docs"           # eingecheckte Artefakte (Figuren, Demo-Seite)


def ensure_dirs() -> None:
    """Legt alle Ausgabeordner an (idempotent). Vor dem Schreiben aufrufen."""
    for d in (RESULTS_DIR, STUDIES_DIR, MODELS_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
