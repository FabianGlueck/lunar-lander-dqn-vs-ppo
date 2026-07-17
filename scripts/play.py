"""Lunar Lander zum Selberspielen — dieselbe Env, dieselben Aktionen wie der Agent.

Kein vereinfachtes Nachbau-Spiel: Es läuft `config.ENV_ID` über `make_env`, mit exakt
den vier diskreten Aktionen, die DQN und PPO auch bekommen, und exakt einer Aktion pro
Frame. Wer es spielt, versteht in 30 Sekunden, warum die Aufgabe nicht trivial ist.

Steuerung:
    ←  →     zur Seite schieben
    ↑ / Leer Hauptriebwerk (schiebt nach oben)
    R        neue Episode
    Esc / Q  beenden

Aufruf:
    uv run python -m scripts.play
    uv run python -m scripts.play --seed 42     # feste Startbedingung
"""

import argparse

import numpy as np

from lunarlander import config
from lunarlander.envs import make_env

# Aktionen von LunarLander-v3 — empirisch verifiziert, nicht aus der Doku übernommen:
# Aktion 1 schiebt nach links, Aktion 3 nach rechts, Aktion 2 ist das Hauptriebwerk
# (der Strahl geht nach unten, der Lander also nach oben).
NICHTS, LINKS, SCHUB, RECHTS = 0, 1, 2, 3


def action_from_keys(links: bool, rechts: bool, schub: bool) -> int:
    """Übersetzt gedrückte Tasten in *eine* Aktion — mehr gibt der Aktionsraum nicht her.

    Die Seitendüsen haben Vorrang vor dem Schub: Man hält den Schub gedrückt und tippt
    zur Korrektur kurz zur Seite; hätte der Schub Vorrang, käme der Tipper nie an.
    Links und rechts zusammen heben sich auf.

    Args:
        links, rechts, schub: ist die jeweilige Taste gerade gedrückt?

    Returns:
        Die Aktion für `env.step()`.
    """
    if links and not rechts:
        return LINKS
    if rechts and not links:
        return RECHTS
    if schub:
        return SCHUB
    return NICHTS


def _zeichne_hud(pygame, schirm, schrift, punkte: float, schritte: int,
                 ergebnis: str | None) -> None:
    """Schreibt Punktestand und Ergebnis auf den Schirm — ohne zu flippen.

    Das Flippen macht die Spielschleife *einmal* pro Frame. Zeichnet man hier schon,
    zeigt die Anzeige abwechselnd Bilder mit und ohne HUD → Flackern.
    """
    zeilen = [f"Score: {punkte:6.0f}", f"Steps: {schritte}"]
    if ergebnis:
        zeilen.append(ergebnis)
        zeilen.append("R = new episode")
    for i, zeile in enumerate(zeilen):
        farbe = (255, 255, 255) if i < 2 else (255, 210, 90)
        schirm.blit(schrift.render(zeile, True, farbe), (10, 10 + i * 26))


def main(seed: int | None = None) -> None:
    import pygame   # erst hier: der Import öffnet Fenster-Ressourcen

    rng = np.random.default_rng()
    aktueller_seed = seed if seed is not None else int(rng.integers(0, 100_000))
    # Bewusst `rgb_array` statt `human`: im human-Modus flippt `env.render()` die
    # Anzeige selbst, ein HUD danach landete nur in jedem zweiten Puffer und flackerte.
    # So holen wir das Bild als Array und zeichnen Bild + HUD in einem Durchgang.
    env = make_env(seed=aktueller_seed, render_mode="rgb_array")

    obs, _ = env.reset()
    bild = env.render()
    hoehe, breite = bild.shape[:2]

    pygame.init()
    schirm = pygame.display.set_mode((breite, hoehe))
    pygame.display.set_caption("Lunar Lander — same env, same 4 actions as the agent")
    schrift = pygame.font.Font(None, 30)
    takt = pygame.time.Clock()

    print(__doc__)
    punkte, schritte, ergebnis, laeuft = 0.0, 0, None, True

    while laeuft:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                laeuft = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    laeuft = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    punkte, schritte, ergebnis = 0.0, 0, None

        if ergebnis is None:      # nach Episodenende einfrieren, bis R gedrückt wird
            tasten = pygame.key.get_pressed()
            aktion = action_from_keys(
                links=tasten[pygame.K_LEFT],
                rechts=tasten[pygame.K_RIGHT],
                schub=tasten[pygame.K_UP] or tasten[pygame.K_SPACE],
            )
            obs, reward, beendet, abgebrochen, _ = env.step(aktion)
            punkte += reward
            schritte += 1
            if beendet or abgebrochen:
                # Der Schluss-Reward verrät den Ausgang: -100 Einschlag, +100 Landung.
                if reward == -100:
                    ergebnis = f"CRASHED — {punkte:.0f}"
                elif reward == 100:
                    ergebnis = f"LANDED — {punkte:.0f}"
                else:
                    ergebnis = f"TIME OUT — {punkte:.0f}"
                geloest = " (solved!)" if punkte >= config.SOLVED_THRESHOLD else ""
                print(f"{ergebnis}{geloest}   nach {schritte} Schritten")

        # Genau ein Zeichendurchgang pro Frame: Bild, HUD, ein Flip.
        bild = env.render()
        schirm.blit(pygame.surfarray.make_surface(bild.swapaxes(0, 1)), (0, 0))
        _zeichne_hud(pygame, schirm, schrift, punkte, schritte, ergebnis)
        pygame.display.flip()
        takt.tick(env.metadata.get("render_fps", 50))   # Echtzeit wie beim Agenten

    env.close()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lunar Lander selbst fliegen.")
    parser.add_argument("--seed", type=int, default=None,
                        help="feste Startbedingung (sonst zufällig)")
    main(seed=parser.parse_args().seed)
