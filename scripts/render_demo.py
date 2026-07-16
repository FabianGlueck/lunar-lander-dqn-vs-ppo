"""Rendert die Demo-Landungen als GIFs und baut eine HTML-Seite mit Grid.

Gezeigt werden **exakt** die Episoden, die im Poster-Balkendiagramm als Punkte
stehen: je Algorithmus das beste Modell (bester Seed laut `results/metrics/<algo>.csv`),
bewertet über `N_RUNS` Episoden ab `EVAL_SEED`, plus die Zufalls-Baseline.

Die Werte sind reproduzierbar, weil `make_env` die Env einmalig seedet und der
Zufallsstrom danach deterministisch weiterläuft — dieselbe Mechanik wie in
`notebooks/03_analysis.ipynb`, Abschnitt G.

Aufruf:
    uv run python -m scripts.render_demo          # nach docs/demo/ (eingecheckt)
    uv run python -m scripts.render_demo --open   # und im Browser öffnen
"""

import argparse
import json
import webbrowser

import numpy as np
import pandas as pd
from PIL import Image

from lunarlander import config
from lunarlander.agents import load_agent, make_random_agent
from lunarlander.envs import make_env

# --- Muss zu notebooks/03_analysis.ipynb (Abschnitt G) passen, sonst zeigen die
# --- GIFs andere Episoden als der Plot.
EVAL_SEED = 12_345
N_RUNS = 10

OUT_DIR = config.DOCS_DIR / "demo"
SCALE = 0.5          # 600x400 -> 300x200 (Kachelbreite im 5er-Grid)
FRAME_STEP = 2       # jeder 2. Simulationsschritt -> 25 fps
FRAME_MS = 40        # 2 Schritte * 20 ms -> Echtzeit
GIF_COLORS = 64

# Farben aus src/lunarlander/plots.py, damit Website und Poster zusammenpassen.
THEME = {
    "surface": "#0c1018",
    "ink": "#e7eaf3",
    "muted": "#3a4358",
    "grid": "#1e2637",
}
POLICIES = [
    {"key": "dqn", "label": "DQN", "color": "#0a9fb0"},
    {"key": "ppo", "label": "PPO", "color": "#7d54d6"},
    {"key": "random", "label": "Random", "color": "#e0685a"},
]


def best_seed(algo: str) -> int:
    """Seed mit dem höchsten 100-Episoden-Mittel — wie `np.argmax` im Notebook."""
    df = pd.read_csv(config.METRICS_DIR / f"{algo}.csv")
    return int(df.loc[df["mean_reward"].idxmax(), "seed"])


def rollout(agent, deterministic: bool) -> list[dict]:
    """Spielt `N_RUNS` Episoden und liefert je Episode Frames + Ergebnis.

    Args:
        agent: Objekt mit SB3-`predict`-Schnittstelle.
        deterministic: an `predict` durchgereicht (bei der Zufalls-Policy egal).

    Returns:
        Liste aus Dicts mit `frames`, `reward` (Episoden-Return) und `last`
        (Reward des Schlussschritts: -100 = Crash, +100 = saubere Landung).
    """
    env = make_env(seed=EVAL_SEED, render_mode="rgb_array")
    episodes = []
    for _ in range(N_RUNS):
        obs, _ = env.reset()
        frames, total, last, step = [], 0.0, 0.0, 0
        done = False
        while not done:
            # Batch der Größe 1: die Zufalls-Policy zieht pro `predict` eine Aktion
            # je Zeile — mit rohem `obs` (8 Werte) wären es acht.
            action, _ = agent.predict(obs[None], deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(int(action[0]))
            total += reward
            last = reward
            done = terminated or truncated
            if step % FRAME_STEP == 0 or done:
                frames.append(_shrink(env.render()))
            step += 1
        episodes.append({"frames": frames, "reward": total, "last": last})
    env.close()
    return episodes


def _shrink(frame: np.ndarray) -> Image.Image:
    """Verkleinert einen gerenderten Frame auf Website-Größe."""
    img = Image.fromarray(frame)
    return img.resize((int(img.width * SCALE), int(img.height * SCALE)), Image.BILINEAR)


def write_gif(frames: list[Image.Image], pad_to: int, path) -> None:
    """Schreibt eine Endlos-GIF, aufgefüllt auf `pad_to` Frames.

    Das Auffüllen (letzter Frame eingefroren) synchronisiert alle GIFs im Grid:
    sie starten gemeinsam und springen gemeinsam zurück. Identische Folgeframes
    kostet GIF fast nichts, weil nur Differenzen gespeichert werden.
    """
    frames = frames + [frames[-1]] * (pad_to - len(frames))
    # Eine gemeinsame Palette für alle Frames -> kein Flackern, kleinere Diffs.
    palette = frames[len(frames) // 2].convert("P", palette=Image.ADAPTIVE, colors=GIF_COLORS)
    quantized = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    quantized[0].save(
        path, save_all=True, append_images=quantized[1:],
        duration=FRAME_MS, loop=0, optimize=True,
    )


def outcome(last_reward: float) -> str:
    """Übersetzt den Schluss-Reward in das Episoden-Ergebnis."""
    if last_reward == -100:
        return "crash"
    if last_reward == 100:
        return "landed"
    return "timeout"   # weder Ruhe noch Einschlag -> Abbruch nach 1000 Schritten


def build(open_browser: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for policy in POLICIES:
        key = policy["key"]
        if key == "random":
            env = make_env(seed=EVAL_SEED)
            agent, deterministic, source = make_random_agent(env, seed=EVAL_SEED), False, "uniform"
            episodes = rollout(agent, deterministic)
            env.close()
        else:
            seed = best_seed(key)
            agent = load_agent(key, config.MODELS_DIR / f"{key}_seed{seed}" / "best_model")
            episodes, source = rollout(agent, True), f"best_model, seed {seed}"
        results.append({**policy, "source": source, "episodes": episodes})
        rewards = np.array([e["reward"] for e in episodes])
        print(f"{policy['label']:7s} ({source:18s}): {rewards.mean():7.1f} ± {rewards.std():5.1f}"
              f"   {[round(r) for r in rewards]}")

    pad_to = max(len(e["frames"]) for r in results for e in r["episodes"])
    print(f"\nGIF-Länge: {pad_to} Frames ({pad_to * FRAME_MS / 1000:.1f} s Loop)")

    for r in results:
        for i, ep in enumerate(r["episodes"]):
            ep["file"] = f"{r['key']}_{i:02d}.gif"
            write_gif(ep["frames"], pad_to, OUT_DIR / ep["file"])

    (OUT_DIR / "index.html").write_text(render_html(results, pad_to), encoding="utf-8")
    print(f"Geschrieben: {OUT_DIR / 'index.html'}")
    if open_browser:
        webbrowser.open((OUT_DIR / "index.html").as_uri())


def render_html(results: list[dict], pad_to: int) -> str:
    sections = []
    for r in results:
        rewards = np.array([e["reward"] for e in r["episodes"]])
        crashes = sum(outcome(e["last"]) == "crash" for e in r["episodes"])
        tiles = []
        for i, ep in enumerate(r["episodes"]):
            res = outcome(ep["last"])
            tiles.append(f"""
        <figure class="tile {res}">
          <img src="{ep['file']}" alt="{r['label']} Episode {i + 1}">
          <figcaption><span class="ep">#{i + 1}</span>
            <span class="score">{ep['reward']:.0f}</span>
            <span class="badge {res}">{res}</span></figcaption>
        </figure>""")
        note = f"{crashes} crash{'es' if crashes != 1 else ''}" if crashes else "no crashes"
        sections.append(f"""
    <section style="--accent:{r['color']}">
      <h2><span class="dot"></span>{r['label']}
        <small>{r['source']} · {rewards.mean():.1f} ± {rewards.std():.1f} · {note}</small></h2>
      <div class="grid">{''.join(tiles)}</div>
    </section>""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lunar Lander — 10 landings per policy</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 2rem clamp(1rem, 3vw, 3rem) 4rem;
         background: {THEME['surface']}; color: {THEME['ink']};
         font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  header {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: baseline;
            justify-content: space-between; border-bottom: 1px solid {THEME['muted']};
            padding-bottom: 1rem; margin-bottom: 2rem; }}
  h1 {{ font-size: 1.5rem; margin: 0; font-weight: 600; }}
  header p {{ margin: .25rem 0 0; color: #9aa4bd; font-size: .9rem; }}
  button {{ background: {THEME['grid']}; color: {THEME['ink']};
            border: 1px solid {THEME['muted']}; border-radius: 6px;
            padding: .5rem 1rem; font-size: .9rem; cursor: pointer; }}
  button:hover {{ border-color: {THEME['ink']}; }}
  section {{ margin-bottom: 2.5rem; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; margin: 0 0 .75rem;
        display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }}
  .dot {{ width: .7rem; height: .7rem; border-radius: 50%; background: var(--accent); }}
  h2 small {{ color: #9aa4bd; font-weight: 400; font-size: .85rem; }}
  .grid {{ display: grid; gap: .7rem; grid-template-columns: repeat(5, 1fr); }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} }}
  .tile {{ margin: 0; background: #0a0e15; border: 1px solid {THEME['grid']};
           border-radius: 8px; overflow: hidden; }}
  .tile.crash {{ border-color: {THEME['muted']}; }}
  .tile img {{ display: block; width: 100%; height: auto; }}
  figcaption {{ display: flex; align-items: center; gap: .4rem;
                padding: .5rem .6rem; font-size: .85rem; }}
  .ep {{ color: #6b7590; }}
  .score {{ margin-left: auto; font-variant-numeric: tabular-nums; font-weight: 600; }}
  .badge {{ font-size: .7rem; padding: .15rem .4rem; border-radius: 3px;
            text-transform: uppercase; letter-spacing: .03em; }}
  .badge.landed {{ background: #14361f; color: #79d18f; }}
  .badge.crash {{ background: #3b1a17; color: #ef8377; }}
  .badge.timeout {{ background: #3a3115; color: #d9b04e; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>Lunar Lander — 10 landings per policy</h1>
    <p>Best model per algorithm, eval seed {EVAL_SEED} · the same episodes shown in the poster
       bar chart · {pad_to * FRAME_MS / 1000:.1f} s loop, in sync</p>
  </div>
  <button id="restart">Restart all (R)</button>
</header>
{''.join(sections)}
<script>
  // GIFs neu laden = synchroner Neustart. Cache-Buster, sonst zeigt der Browser
  // den laufenden Loop weiter.
  const restart = () => document.querySelectorAll('img').forEach(img => {{
    const src = img.src.split('?')[0];
    img.src = src + '?' + Date.now();
  }});
  document.getElementById('restart').addEventListener('click', restart);
  addEventListener('keydown', e => {{ if (e.key.toLowerCase() === 'r') restart(); }});
</script>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="Seite im Browser öffnen")
    build(open_browser=parser.parse_args().open)
