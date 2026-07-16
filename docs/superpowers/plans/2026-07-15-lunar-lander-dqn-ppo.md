# Lunar Lander DQN vs. PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eine algorithmus-agnostische SB3-Pipeline bauen, die DQN (Meilenstein 1) und später PPO auf `LunarLander-v3` per Optuna tunt, über mehrere Seeds trainiert und statistisch vergleicht.

**Architecture:** Kleines Python-Paket `src/lunarlander/` mit einer einzigen algorithmus-spezifischen Stelle (`agents.make_agent`). Alles andere (Training, Evaluation, Tuning, Statistik, Plots) ist algorithmus-agnostisch und arbeitet auf SB3-Modellen bzw. Reward-Arrays. Lange Läufe laufen als Skripte und speichern Ergebnisse (Optuna-`.db`, CSVs); Notebooks bleiben dünn und zeigen nur Ergebnisse.

**Tech Stack:** Python 3.12, Gymnasium (Box2D), Stable-Baselines3 (DQN/PPO), Optuna (TPE + MedianPruner), NumPy, SciPy, Matplotlib, pytest, uv.

## Global Constraints

- Python `>=3.12`, Dependencies via `uv` (siehe `pyproject.toml`).
- Environment: `LunarLander-v3` (diskret, 8D-Zustand, 4 Aktionen). "Gelöst" = mittlere Rendite ≥ 200 über 100 Episoden.
- Feste Seeds für Reproduzierbarkeit: `SEEDS = [0, 1, 2, 3, 4]`.
- Nur SB3-Implementierungen; keine from-scratch-Algorithmen.
- Schlanker Umfang / Laptop-CPU: Tuning ~30 Trials, finaler Vergleich 5 Seeds.
- Alle Kommandos mit `uv run` ausführen.
- Tests dürfen keine langen Trainings ausführen: Test-Trainings nutzen sehr kleine Timestep-Zahlen (≤ 2000).

---

## File Structure

```
src/lunarlander/
  __init__.py
  config.py       # Konstanten: ENV_ID, SEEDS, Schwellen, Timesteps, Pfade
  envs.py         # make_env(seed, render_mode) -> gym.Env (Monitor-gewrappt)
  agents.py       # make_agent(algo, hyperparams, env, seed) -> SB3-Modell  (EINZIGE algo-spezifische Stelle)
  train.py        # train(algo, hyperparams, seed, timesteps, log_dir) -> (model, history_path)
  evaluate.py     # evaluate(model, seed, n_episodes) -> np.ndarray[float]  (Reward pro Episode)
  tune.py         # sample_hyperparams(trial, algo), objective(...), run_study(algo, n_trials, db_path) -> optuna.Study
  stats.py        # welch_t_test, confidence_interval, cohens_d, summarize
  plots.py        # learning_curve(history_paths), comparison_plot(rewards_by_algo)
scripts/
  run_tuning.py       # CLI: python -m scripts.run_tuning --algo dqn --trials 30
  run_final_eval.py   # CLI: trainiert beste Config über alle Seeds, schreibt CSV
notebooks/
  01_dqn_explore.ipynb
  04_comparison.ipynb
tests/
  test_config.py
  test_envs.py
  test_agents.py
  test_train.py
  test_evaluate.py
  test_tune.py
  test_stats.py
  test_plots.py
results/
  studies/        # Optuna .db
  models/         # beste Modelle + EvalCallback-Logs
  metrics/        # finale Rewards pro Seed (CSV)
```

---

### Task 1: Projekt-Setup & Konfiguration

**Files:**
- Modify: `pyproject.toml` (Dependency `gymnasium[box2d]`, `pytest`)
- Create: `src/lunarlander/__init__.py`
- Create: `src/lunarlander/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nichts (erste Task).
- Produces: Konstanten `ENV_ID: str`, `SEEDS: list[int]`, `SOLVED_THRESHOLD: float`, `N_EVAL_EPISODES: int`, `TUNE_TIMESTEPS: int`, `FINAL_TIMESTEPS: int`, `EVAL_FREQ: int`, `CALLBACK_EVAL_EPISODES: int`; Pfade `RESULTS_DIR`, `STUDIES_DIR`, `MODELS_DIR`, `METRICS_DIR` (alle `pathlib.Path`); Funktion `ensure_dirs() -> None`.

- [ ] **Step 1: Dependencies ergänzen**

Run:
```bash
uv add "gymnasium[box2d]" pandas && uv add --dev pytest
```
Expected: `pyproject.toml` enthält jetzt `gymnasium[box2d]` und `pandas` sowie eine `[dependency-groups]`/dev-Sektion mit `pytest`. `uv.lock` aktualisiert. (Box2D benötigt ggf. `swig`; falls der Build fehlschlägt: `brew install swig` und erneut ausführen.)

- [ ] **Step 2: Failing test schreiben**

Create `tests/__init__.py` (leer) und `tests/test_config.py`:
```python
from pathlib import Path
from lunarlander import config


def test_core_constants():
    assert config.ENV_ID == "LunarLander-v3"
    assert config.SEEDS == [0, 1, 2, 3, 4]
    assert config.SOLVED_THRESHOLD == 200
    assert config.N_EVAL_EPISODES == 100


def test_paths_are_paths():
    for p in (config.RESULTS_DIR, config.STUDIES_DIR, config.MODELS_DIR, config.METRICS_DIR):
        assert isinstance(p, Path)


def test_ensure_dirs_creates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "STUDIES_DIR", tmp_path / "results" / "studies")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "results" / "models")
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "results" / "metrics")
    config.ensure_dirs()
    assert (tmp_path / "results" / "studies").is_dir()
```

- [ ] **Step 3: Test ausführen (soll fehlschlagen)**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'lunarlander'`.

- [ ] **Step 4: Paket-Konfiguration + config.py schreiben**

`src/lunarlander/__init__.py` (leer lassen).

Damit `import lunarlander` funktioniert, in `pyproject.toml` unter `[tool.uv]`-Ebene ein Paketverzeichnis definieren. Ergänze in `pyproject.toml`:
```toml
[tool.setuptools.packages.find]
where = ["src"]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
```
Und installiere das Paket editierbar in die Umgebung:
```bash
uv pip install -e .
```

`src/lunarlander/config.py`:
```python
from pathlib import Path

ENV_ID = "LunarLander-v3"
SEEDS = [0, 1, 2, 3, 4]
SOLVED_THRESHOLD = 200

# Evaluation
N_EVAL_EPISODES = 100          # finale Bewertung
CALLBACK_EVAL_EPISODES = 20    # periodische Bewertung während des Trainings
EVAL_FREQ = 10_000             # Steps zwischen periodischen Bewertungen

# Trainingslängen
TUNE_TIMESTEPS = 150_000       # verkürztes Training pro Optuna-Trial
FINAL_TIMESTEPS = 500_000      # volles Training für den finalen Vergleich

# Pfade
RESULTS_DIR = Path("results")
STUDIES_DIR = RESULTS_DIR / "studies"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, STUDIES_DIR, MODELS_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 5: Test ausführen (soll bestehen)**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (3 Tests).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/lunarlander tests
git commit -m "feat: project setup and config for lunar lander pipeline"
```

---

### Task 2: Environment-Factory

**Files:**
- Create: `src/lunarlander/envs.py`
- Create: `tests/test_envs.py`

**Interfaces:**
- Consumes: `config.ENV_ID`.
- Produces: `make_env(seed: int, render_mode: str | None = None) -> gym.Env`. Rückgabe ist ein `Monitor`-gewrapptes Env mit gesetztem Seed. Observation-Space Shape `(8,)`, Action-Space `Discrete(4)`.

- [ ] **Step 1: Failing test schreiben**

`tests/test_envs.py`:
```python
import gymnasium as gym
from lunarlander.envs import make_env


def test_make_env_spaces():
    env = make_env(seed=0)
    assert env.observation_space.shape == (8,)
    assert env.action_space.n == 4
    env.close()


def test_make_env_reset_is_seeded():
    obs_a, _ = make_env(seed=42).reset()
    obs_b, _ = make_env(seed=42).reset()
    assert (obs_a == obs_b).all()
```

- [ ] **Step 2: Test ausführen (soll fehlschlagen)**

Run: `uv run pytest tests/test_envs.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'lunarlander.envs'`.

- [ ] **Step 3: envs.py schreiben**

`src/lunarlander/envs.py`:
```python
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from lunarlander import config


def make_env(seed: int, render_mode: str | None = None) -> gym.Env:
    env = gym.make(config.ENV_ID, render_mode=render_mode)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
```

- [ ] **Step 4: Test ausführen (soll bestehen)**

Run: `uv run pytest tests/test_envs.py -v`
Expected: PASS (2 Tests). (Falls `Box2D`-Importfehler: siehe Task 1 Step 1, `swig`.)

- [ ] **Step 5: Commit**

```bash
git add src/lunarlander/envs.py tests/test_envs.py
git commit -m "feat: seeded LunarLander env factory"
```

---

### Task 3: Agent-Factory (einzige algo-spezifische Stelle)

**Files:**
- Create: `src/lunarlander/agents.py`
- Create: `tests/test_agents.py`

**Interfaces:**
- Consumes: `make_env` (nur im Test), SB3 `DQN`/`PPO`.
- Produces: `make_agent(algo: str, hyperparams: dict, env, seed: int)` → SB3-Modell. `algo` ∈ {`"dqn"`, `"ppo"`}. `hyperparams` kann optional `net_arch: list[int]` enthalten (wird in `policy_kwargs` übersetzt). Unbekannter `algo` → `ValueError`.

- [ ] **Step 1: Failing test schreiben**

`tests/test_agents.py`:
```python
import pytest
from stable_baselines3 import DQN, PPO
from lunarlander.agents import make_agent
from lunarlander.envs import make_env


def test_make_agent_dqn():
    env = make_env(seed=0)
    model = make_agent("dqn", {"learning_rate": 1e-3}, env, seed=0)
    assert isinstance(model, DQN)
    env.close()


def test_make_agent_ppo():
    env = make_env(seed=0)
    model = make_agent("ppo", {"learning_rate": 3e-4}, env, seed=0)
    assert isinstance(model, PPO)
    env.close()


def test_make_agent_net_arch():
    env = make_env(seed=0)
    model = make_agent("dqn", {"net_arch": [64, 64]}, env, seed=0)
    assert isinstance(model, DQN)
    env.close()


def test_make_agent_unknown():
    env = make_env(seed=0)
    with pytest.raises(ValueError):
        make_agent("sarsa", {}, env, seed=0)
    env.close()
```

- [ ] **Step 2: Test ausführen (soll fehlschlagen)**

Run: `uv run pytest tests/test_agents.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'lunarlander.agents'`.

- [ ] **Step 3: agents.py schreiben**

`src/lunarlander/agents.py`:
```python
from stable_baselines3 import DQN, PPO

_ALGOS = {"dqn": DQN, "ppo": PPO}


def make_agent(algo: str, hyperparams: dict, env, seed: int):
    algo = algo.lower()
    if algo not in _ALGOS:
        raise ValueError(f"Unbekannter Algorithmus: {algo!r}. Erlaubt: {list(_ALGOS)}")

    params = dict(hyperparams)  # Kopie, Original nicht mutieren
    net_arch = params.pop("net_arch", None)
    if net_arch is not None:
        params["policy_kwargs"] = {"net_arch": net_arch}

    cls = _ALGOS[algo]
    return cls("MlpPolicy", env, seed=seed, verbose=0, **params)
```

- [ ] **Step 4: Test ausführen (soll bestehen)**

Run: `uv run pytest tests/test_agents.py -v`
Expected: PASS (4 Tests).

- [ ] **Step 5: Commit**

```bash
git add src/lunarlander/agents.py tests/test_agents.py
git commit -m "feat: algo-agnostic agent factory (DQN/PPO)"
```

---

### Task 4: Training mit periodischer Evaluation

**Files:**
- Create: `src/lunarlander/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: `make_env`, `make_agent`, `config`.
- Produces: `train(algo: str, hyperparams: dict, seed: int, timesteps: int, log_dir: str | Path) -> tuple[model, Path]`. Trainiert das Modell, schreibt periodische Eval-Ergebnisse via SB3 `EvalCallback` nach `log_dir/evaluations.npz` (enthält Arrays `timesteps` und `results`). Rückgabe: `(model, Path(log_dir) / "evaluations.npz")`.

- [ ] **Step 1: Failing test schreiben**

`tests/test_train.py` (nutzt winzige Timestep-Zahl):
```python
import numpy as np
from lunarlander.train import train


def test_train_returns_model_and_history(tmp_path):
    model, hist_path = train(
        "dqn",
        {"learning_rate": 1e-3, "learning_starts": 100},
        seed=0,
        timesteps=1500,
        log_dir=tmp_path,
    )
    assert model is not None
    assert hist_path.exists()
    data = np.load(hist_path)
    assert "timesteps" in data and "results" in data
    assert len(data["timesteps"]) >= 1
```

- [ ] **Step 2: Test ausführen (soll fehlschlagen)**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'lunarlander.train'`.

- [ ] **Step 3: train.py schreiben**

`src/lunarlander/train.py`:
```python
from pathlib import Path

from stable_baselines3.common.callbacks import EvalCallback

from lunarlander import config
from lunarlander.agents import make_agent
from lunarlander.envs import make_env


def train(algo, hyperparams, seed, timesteps, log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(seed=seed)
    eval_env = make_env(seed=seed + 1000)  # separater Seed für faire Evaluation

    model = make_agent(algo, hyperparams, env, seed=seed)

    eval_freq = min(config.EVAL_FREQ, max(timesteps // 2, 1))
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=config.CALLBACK_EVAL_EPISODES,
        deterministic=True,
        verbose=0,
    )
    model.learn(total_timesteps=timesteps, callback=eval_cb)

    env.close()
    eval_env.close()
    return model, log_dir / "evaluations.npz"
```

- [ ] **Step 4: Test ausführen (soll bestehen)**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS. (Dauert wenige Sekunden wegen 1500 Steps.)

- [ ] **Step 5: Commit**

```bash
git add src/lunarlander/train.py tests/test_train.py
git commit -m "feat: training loop with periodic eval logging"
```

---

### Task 5: Evaluation (Reward pro Episode)

**Files:**
- Create: `src/lunarlander/evaluate.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `make_env`.
- Produces: `evaluate(model, seed: int, n_episodes: int) -> np.ndarray` (Shape `(n_episodes,)`, float, deterministische Policy).

- [ ] **Step 1: Failing test schreiben**

`tests/test_evaluate.py`:
```python
import numpy as np
from lunarlander.agents import make_agent
from lunarlander.envs import make_env
from lunarlander.evaluate import evaluate


def test_evaluate_shape():
    env = make_env(seed=0)
    model = make_agent("dqn", {"learning_starts": 100}, env, seed=0)
    rewards = evaluate(model, seed=123, n_episodes=3)
    assert isinstance(rewards, np.ndarray)
    assert rewards.shape == (3,)
    assert rewards.dtype.kind == "f"
    env.close()
```

- [ ] **Step 2: Test ausführen (soll fehlschlagen)**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'lunarlander.evaluate'`.

- [ ] **Step 3: evaluate.py schreiben**

`src/lunarlander/evaluate.py`:
```python
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from lunarlander.envs import make_env


def evaluate(model, seed, n_episodes):
    eval_env = make_env(seed=seed)
    rewards, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )
    eval_env.close()
    return np.asarray(rewards, dtype=float)
```

- [ ] **Step 4: Test ausführen (soll bestehen)**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lunarlander/evaluate.py tests/test_evaluate.py
git commit -m "feat: per-episode deterministic evaluation"
```

---

### Task 6: Statistik (Welch t-Test, CI, Cohen's d)

**Files:**
- Create: `src/lunarlander/stats.py`
- Create: `tests/test_stats.py`

**Interfaces:**
- Consumes: NumPy, SciPy.
- Produces:
  - `welch_t_test(a, b) -> tuple[float, float]` (t-Statistik, p-Wert).
  - `confidence_interval(data, confidence=0.95) -> tuple[float, float, float]` (mean, low, high).
  - `cohens_d(a, b) -> float` (gepoolte Standardabweichung).
  - `summarize(a, b, label_a, label_b) -> dict` (aggregierte Kennzahlen für Poster).

- [ ] **Step 1: Failing test schreiben**

`tests/test_stats.py`:
```python
import numpy as np
from lunarlander import stats


def test_welch_identical_data_high_p():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t, p = stats.welch_t_test(a, a)
    assert abs(t) < 1e-9
    assert p > 0.99


def test_welch_separated_data_low_p():
    a = np.array([200.0, 205.0, 198.0, 202.0, 199.0])
    b = np.array([100.0, 95.0, 105.0, 98.0, 102.0])
    _, p = stats.welch_t_test(a, b)
    assert p < 0.05


def test_confidence_interval_brackets_mean():
    data = np.array([10.0, 12.0, 11.0, 9.0, 13.0])
    mean, low, high = stats.confidence_interval(data)
    assert low < mean < high
    assert abs(mean - 11.0) < 1e-9


def test_cohens_d_sign_and_magnitude():
    a = np.array([10.0, 11.0, 9.0, 10.0, 10.0])
    b = np.array([0.0, 1.0, -1.0, 0.0, 0.0])
    d = stats.cohens_d(a, b)
    assert d > 2.0  # großer Effekt


def test_summarize_keys():
    a = np.array([200.0, 210.0, 190.0])
    b = np.array([150.0, 160.0, 140.0])
    out = stats.summarize(a, b, "dqn", "ppo")
    for key in ("mean_dqn", "mean_ppo", "p_value", "cohens_d", "ci_dqn", "ci_ppo"):
        assert key in out
```

- [ ] **Step 2: Test ausführen (soll fehlschlagen)**

Run: `uv run pytest tests/test_stats.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'lunarlander.stats'`.

- [ ] **Step 3: stats.py schreiben**

`src/lunarlander/stats.py`:
```python
import numpy as np
from scipy import stats as sp


def welch_t_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = sp.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def confidence_interval(data, confidence=0.95):
    a = np.asarray(data, float)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return mean, mean, mean
    se = sp.sem(a)
    h = se * sp.t.ppf((1 + confidence) / 2, n - 1)
    return mean, float(mean - h), float(mean + h)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def summarize(a, b, label_a, label_b):
    _, p = welch_t_test(a, b)
    return {
        f"mean_{label_a}": float(np.mean(a)),
        f"mean_{label_b}": float(np.mean(b)),
        f"ci_{label_a}": confidence_interval(a),
        f"ci_{label_b}": confidence_interval(b),
        "p_value": p,
        "cohens_d": cohens_d(a, b),
    }
```

- [ ] **Step 4: Test ausführen (soll bestehen)**

Run: `uv run pytest tests/test_stats.py -v`
Expected: PASS (5 Tests).

- [ ] **Step 5: Commit**

```bash
git add src/lunarlander/stats.py tests/test_stats.py
git commit -m "feat: statistics (Welch t-test, CI, Cohen's d)"
```

---

### Task 7: Plots (Lernkurve + Vergleich)

**Files:**
- Create: `src/lunarlander/plots.py`
- Create: `tests/test_plots.py`

**Interfaces:**
- Consumes: `evaluations.npz`-Dateien (aus `train`), Reward-Arrays, `stats`.
- Produces:
  - `learning_curve(history_paths: dict[str, list[Path]], out_path: Path) -> Path` — mittlere Eval-Rendite über Steps je Algorithmus (mit ±Std-Band über Seeds); speichert PNG.
  - `comparison_plot(rewards_by_algo: dict[str, np.ndarray], out_path: Path) -> Path` — Boxplot der finalen Rendite je Algorithmus; speichert PNG.

- [ ] **Step 1: Failing test schreiben**

`tests/test_plots.py`:
```python
import numpy as np
from lunarlander import plots


def _make_npz(path, seed):
    np.savez(path, timesteps=np.array([10, 20, 30]),
             results=np.random.default_rng(seed).normal(0, 1, size=(3, 5)))
    return path


def test_learning_curve_saves(tmp_path):
    paths = {"dqn": [_make_npz(tmp_path / "d0.npz", 0)],
             "ppo": [_make_npz(tmp_path / "p0.npz", 1)]}
    out = plots.learning_curve(paths, tmp_path / "lc.png")
    assert out.exists()


def test_comparison_plot_saves(tmp_path):
    rewards = {"dqn": np.array([200.0, 210.0, 190.0]),
               "ppo": np.array([150.0, 160.0, 140.0])}
    out = plots.comparison_plot(rewards, tmp_path / "cmp.png")
    assert out.exists()
```

- [ ] **Step 2: Test ausführen (soll fehlschlagen)**

Run: `uv run pytest tests/test_plots.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'lunarlander.plots'`.

- [ ] **Step 3: plots.py schreiben**

`src/lunarlander/plots.py`:
```python
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, kein Display nötig
import matplotlib.pyplot as plt
import numpy as np


def learning_curve(history_paths, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    for algo, paths in history_paths.items():
        # Alle Seeds auf gemeinsame Timesteps (erste Datei) mitteln.
        first = np.load(paths[0])
        timesteps = first["timesteps"]
        per_seed_means = []
        for p in paths:
            data = np.load(p)
            per_seed_means.append(data["results"].mean(axis=1))
        stacked = np.vstack(per_seed_means)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        ax.plot(timesteps, mean, label=algo.upper())
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2)
    ax.axhline(200, ls="--", color="grey", label="gelöst (200)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mittlere Eval-Rendite")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def comparison_plot(rewards_by_algo, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(rewards_by_algo.keys())
    ax.boxplot([rewards_by_algo[k] for k in labels], labels=[l.upper() for l in labels])
    ax.axhline(200, ls="--", color="grey", label="gelöst (200)")
    ax.set_ylabel("Finale Rendite (100 Episoden)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
```

- [ ] **Step 4: Test ausführen (soll bestehen)**

Run: `uv run pytest tests/test_plots.py -v`
Expected: PASS (2 Tests).

- [ ] **Step 5: Commit**

```bash
git add src/lunarlander/plots.py tests/test_plots.py
git commit -m "feat: learning-curve and comparison plots"
```

---

### Task 8: Optuna-Tuning (algo-agnostisch)

**Files:**
- Create: `src/lunarlander/tune.py`
- Create: `tests/test_tune.py`

**Interfaces:**
- Consumes: `train`, `evaluate`, `config`, Optuna.
- Produces:
  - `sample_hyperparams(trial, algo: str) -> dict` — Suchraum je Algorithmus (DQN/PPO).
  - `objective(trial, algo, timesteps, log_root) -> float` — trainiert kurz, gibt mittlere Eval-Rendite zurück.
  - `run_study(algo, n_trials, db_path, timesteps=None, study_name=None) -> optuna.Study` — TPE + MedianPruner, speichert in SQLite unter `db_path`.

- [ ] **Step 1: Failing test schreiben**

`tests/test_tune.py`:
```python
import optuna
from lunarlander import tune


def test_sample_hyperparams_dqn():
    study = optuna.create_study()
    trial = study.ask()
    params = tune.sample_hyperparams(trial, "dqn")
    assert "learning_rate" in params
    assert "gamma" in params


def test_sample_hyperparams_ppo():
    study = optuna.create_study()
    trial = study.ask()
    params = tune.sample_hyperparams(trial, "ppo")
    assert "n_steps" in params
    assert "clip_range" in params


def test_run_study_tiny(tmp_path):
    db = tmp_path / "t.db"
    study = tune.run_study("dqn", n_trials=2, db_path=db,
                           timesteps=1500, study_name="tiny")
    assert len(study.trials) == 2
    assert study.best_value is not None
```

- [ ] **Step 2: Test ausführen (soll fehlschlagen)**

Run: `uv run pytest tests/test_tune.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'lunarlander.tune'`.

- [ ] **Step 3: tune.py schreiben**

`src/lunarlander/tune.py`:
```python
import tempfile
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from lunarlander import config
from lunarlander.evaluate import evaluate
from lunarlander.train import train


def sample_hyperparams(trial, algo):
    algo = algo.lower()
    if algo == "dqn":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "gamma": trial.suggest_float("gamma", 0.98, 0.9999),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
            "target_update_interval": trial.suggest_categorical("target_update_interval", [250, 500, 1000]),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.3),
            "net_arch": trial.suggest_categorical("net_arch", [[64, 64], [128, 128], [256, 256]]),
        }
    if algo == "ppo":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "gamma": trial.suggest_float("gamma", 0.98, 0.9999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
            "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
            "net_arch": trial.suggest_categorical("net_arch", [[64, 64], [128, 128]]),
        }
    raise ValueError(f"Unbekannter Algorithmus: {algo!r}")


def objective(trial, algo, timesteps, log_root):
    params = sample_hyperparams(trial, algo)
    log_dir = Path(log_root) / f"trial_{trial.number}"
    model, _ = train(algo, params, seed=0, timesteps=timesteps, log_dir=log_dir)
    rewards = evaluate(model, seed=999, n_episodes=config.CALLBACK_EVAL_EPISODES)
    return float(np.mean(rewards))


def run_study(algo, n_trials, db_path, timesteps=None, study_name=None):
    timesteps = timesteps or config.TUNE_TIMESTEPS
    study_name = study_name or f"{algo}_lunarlander"
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=0),
        pruner=MedianPruner(),
        load_if_exists=True,
    )
    log_root = Path(tempfile.mkdtemp(prefix=f"tune_{algo}_"))
    study.optimize(
        lambda t: objective(t, algo, timesteps, log_root),
        n_trials=n_trials,
    )
    return study
```

- [ ] **Step 4: Test ausführen (soll bestehen)**

Run: `uv run pytest tests/test_tune.py -v`
Expected: PASS (3 Tests). (Der `run_study`-Test trainiert 2×1500 Steps → wenige Sekunden.)

- [ ] **Step 5: Commit**

```bash
git add src/lunarlander/tune.py tests/test_tune.py
git commit -m "feat: algo-agnostic Optuna tuning (DQN/PPO search spaces)"
```

---

### Task 9: Skripte für lange Läufe (Tuning + finaler Eval)

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/run_tuning.py`
- Create: `scripts/run_final_eval.py`

**Interfaces:**
- Consumes: `tune.run_study`, `train`, `evaluate`, `config`.
- Produces: zwei CLI-Skripte. `run_tuning.py` schreibt Optuna-`.db` nach `results/studies/<algo>.db` und gibt beste Params aus. `run_final_eval.py` trainiert eine Config über alle `SEEDS` voll und schreibt `results/metrics/<algo>.csv` (Spalten: `seed,mean_reward`).

Kein separater Unit-Test — Verifikation durch Ausführung mit winzigen Werten (Step 3/4 unten).

- [ ] **Step 1: run_tuning.py schreiben**

`scripts/__init__.py` (leer). `scripts/run_tuning.py`:
```python
import argparse

from lunarlander import config
from lunarlander.tune import run_study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["dqn", "ppo"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=config.TUNE_TIMESTEPS)
    args = parser.parse_args()

    config.ensure_dirs()
    db_path = config.STUDIES_DIR / f"{args.algo}.db"
    study = run_study(args.algo, n_trials=args.trials, db_path=db_path,
                      timesteps=args.timesteps)
    print(f"Beste Rendite: {study.best_value:.1f}")
    print(f"Beste Params: {study.best_params}")
    print(f"Study gespeichert in: {db_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: run_final_eval.py schreiben**

`scripts/run_final_eval.py`:
```python
import argparse
import csv
import json

import numpy as np

from lunarlander import config
from lunarlander.evaluate import evaluate
from lunarlander.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["dqn", "ppo"])
    parser.add_argument("--params", required=True,
                        help="JSON-String der besten Hyperparameter")
    parser.add_argument("--timesteps", type=int, default=config.FINAL_TIMESTEPS)
    args = parser.parse_args()

    config.ensure_dirs()
    params = json.loads(args.params)
    out_csv = config.METRICS_DIR / f"{args.algo}.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mean_reward"])
        for seed in config.SEEDS:
            log_dir = config.MODELS_DIR / f"{args.algo}_seed{seed}"
            model, _ = train(args.algo, params, seed=seed,
                             timesteps=args.timesteps, log_dir=log_dir)
            model.save(log_dir / "final_model")
            rewards = evaluate(model, seed=10_000 + seed,
                               n_episodes=config.N_EVAL_EPISODES)
            mean_r = float(np.mean(rewards))
            writer.writerow([seed, mean_r])
            print(f"[{args.algo}] seed={seed} mean_reward={mean_r:.1f}")

    print(f"Ergebnisse gespeichert in: {out_csv}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-Test Tuning-Skript**

Run:
```bash
uv run python -m scripts.run_tuning --algo dqn --trials 2 --timesteps 1500
```
Expected: Läuft durch, druckt "Beste Rendite", "Beste Params", legt `results/studies/dqn.db` an.

- [ ] **Step 4: Smoke-Test finaler-Eval-Skript**

Run:
```bash
uv run python -m scripts.run_final_eval --algo dqn --params '{"learning_starts": 100}' --timesteps 1500
```
Expected: Druckt pro Seed eine Zeile, legt `results/metrics/dqn.csv` mit 5 Datenzeilen an.

- [ ] **Step 5: results/ in .gitignore, dann Commit**

`.gitignore` um `results/` ergänzen (Modelle/DBs nicht einchecken).
```bash
git add scripts .gitignore
git commit -m "feat: CLI scripts for tuning and final multi-seed eval"
```

---

### Task 10: Meilenstein 1 — DQN End-to-End (echter Lauf + Explorations-Notebook)

**Files:**
- Create: `notebooks/01_dqn_explore.ipynb`

**Interfaces:**
- Consumes: die komplette Pipeline (Tasks 1–9).
- Produces: einen echten DQN-Optuna-Lauf, gespeicherte beste Config, eine Lernkurve. Dies ist der Abschluss von Meilenstein 1.

Kein Unit-Test — dies ist ein echter (längerer) Lauf. Zwischen den Schritten kann Stunden vergehen; im Hintergrund laufen lassen.

- [ ] **Step 1: DQN-Tuning echt laufen lassen (Hintergrund)**

Run (kann mehrere Stunden dauern; ggf. Trials reduzieren):
```bash
uv run python -m scripts.run_tuning --algo dqn --trials 30
```
Expected: `results/studies/dqn.db` gefüllt, beste Params in der Konsole. **Beste Params notieren.**

- [ ] **Step 2: Finales DQN über 5 Seeds (Hintergrund)**

Run (beste Params aus Step 1 einsetzen):
```bash
uv run python -m scripts.run_final_eval --algo dqn --params '<BESTE_DQN_PARAMS_JSON>'
```
Expected: `results/metrics/dqn.csv` mit 5 Seeds; mindestens ein Seed nahe/über 200.

- [ ] **Step 3: Explorations-Notebook erstellen**

`notebooks/01_dqn_explore.ipynb` mit diesen Zellen (dünn — ruft nur die Pipeline auf):
1. Imports: `from lunarlander import config, plots`, `import optuna, numpy as np, pandas as pd`.
2. Optuna-Study laden und Verlauf zeigen:
   ```python
   study = optuna.load_study(study_name="dqn_lunarlander",
                             storage=f"sqlite:///{config.STUDIES_DIR}/dqn.db")
   print(study.best_params, study.best_value)
   optuna.visualization.matplotlib.plot_optimization_history(study)
   ```
3. Lernkurve aus den EvalCallback-Logs der 5 finalen Seeds:
   ```python
   paths = {"dqn": [config.MODELS_DIR / f"dqn_seed{s}" / "evaluations.npz"
                    for s in config.SEEDS]}
   plots.learning_curve(paths, config.RESULTS_DIR / "dqn_learning_curve.png")
   ```
4. Finale Rendite anzeigen: `pd.read_csv(config.METRICS_DIR / "dqn.csv")`.

- [ ] **Step 4: Notebook ausführen & prüfen**

Run: `uv run jupyter nbconvert --to notebook --execute notebooks/01_dqn_explore.ipynb --output 01_dqn_explore.ipynb`
Expected: Läuft fehlerfrei durch; `dqn_learning_curve.png` wird erzeugt.

- [ ] **Step 5: Commit**

```bash
git add notebooks/01_dqn_explore.ipynb
git commit -m "feat: DQN end-to-end milestone 1 (tuning, multi-seed, learning curve)"
```

---

### Task 11: Meilenstein 2 — PPO einstecken + finaler Vergleich

**Files:**
- Create: `notebooks/04_comparison.ipynb`

**Interfaces:**
- Consumes: gesamte Pipeline; `stats.summarize`, `plots.comparison_plot`, `plots.learning_curve`.
- Produces: PPO-Ergebnisse + Signifikanzvergleich DQN vs. PPO + Poster-Grafiken. **Kein neuer Pipeline-Code nötig** — PPO läuft durch dieselben Skripte.

- [ ] **Step 1: PPO-Tuning echt laufen lassen (Hintergrund)**

Run:
```bash
uv run python -m scripts.run_tuning --algo ppo --trials 30
```
Expected: `results/studies/ppo.db` gefüllt; beste PPO-Params notieren.

- [ ] **Step 2: Finales PPO über 5 Seeds (Hintergrund)**

Run:
```bash
uv run python -m scripts.run_final_eval --algo ppo --params '<BESTE_PPO_PARAMS_JSON>'
```
Expected: `results/metrics/ppo.csv` mit 5 Seeds.

- [ ] **Step 3: Vergleichs-Notebook erstellen**

`notebooks/04_comparison.ipynb` mit Zellen:
1. Rewards laden:
   ```python
   import pandas as pd, numpy as np
   from lunarlander import config, stats, plots
   dqn = pd.read_csv(config.METRICS_DIR / "dqn.csv")["mean_reward"].values
   ppo = pd.read_csv(config.METRICS_DIR / "ppo.csv")["mean_reward"].values
   ```
2. Statistik:
   ```python
   result = stats.summarize(dqn, ppo, "dqn", "ppo")
   print(result)
   ```
3. Vergleichs-Boxplot:
   ```python
   plots.comparison_plot({"dqn": dqn, "ppo": ppo},
                         config.RESULTS_DIR / "comparison.png")
   ```
4. Gemeinsame Lernkurve DQN vs. PPO:
   ```python
   paths = {a: [config.MODELS_DIR / f"{a}_seed{s}" / "evaluations.npz"
                for s in config.SEEDS] for a in ("dqn", "ppo")}
   plots.learning_curve(paths, config.RESULTS_DIR / "both_learning_curves.png")
   ```
5. Markdown-Zelle: Interpretation (p-Wert, CI, Cohen's d, Limitierung n=5) für Poster-Text.

- [ ] **Step 4: Notebook ausführen & prüfen**

Run: `uv run jupyter nbconvert --to notebook --execute notebooks/04_comparison.ipynb --output 04_comparison.ipynb`
Expected: Läuft fehlerfrei; `comparison.png` und `both_learning_curves.png` erzeugt; Statistik ausgegeben.

- [ ] **Step 5: Demo-GIF einer Episode (optional, fürs Poster)**

In einer neuen Notebook-Zelle:
```python
import imageio, numpy as np
from stable_baselines3 import DQN
from lunarlander.envs import make_env

model = DQN.load(config.MODELS_DIR / "dqn_seed0" / "final_model")
env = make_env(seed=0, render_mode="rgb_array")
frames, obs = [], env.reset(seed=0)[0]
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = env.step(int(action))
    frames.append(env.render())
    done = term or trunc
imageio.mimsave(config.RESULTS_DIR / "dqn_landing.gif", frames, fps=30)
```
Expected: `dqn_landing.gif` zeigt eine Landung.

- [ ] **Step 6: Commit**

```bash
git add notebooks/04_comparison.ipynb
git commit -m "feat: PPO comparison milestone 2 (significance test, poster plots)"
```

---

## Verifikation nach Abschluss

- [ ] Alle Unit-Tests grün: `uv run pytest -v`
- [ ] `results/metrics/dqn.csv` und `results/metrics/ppo.csv` existieren mit je 5 Seeds.
- [ ] Poster-Grafiken erzeugt: `dqn_learning_curve.png`, `comparison.png`, `both_learning_curves.png`, `dqn_landing.gif`.
- [ ] `04_comparison.ipynb` gibt p-Wert, CIs und Cohen's d aus.
