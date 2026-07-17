"""Tests der Tastenbelegung fürs Spiel — die Spielschleife selbst ist untestbar (pygame)."""

from scripts.play import action_from_keys

# Aktionen von LunarLander-v3, empirisch verifiziert (nicht aus der Doku geraten):
# 0 = nichts, 1 = schiebt nach links, 2 = Hauptriebwerk (nach oben), 3 = nach rechts.
NICHTS, LINKS, SCHUB, RECHTS = 0, 1, 2, 3


def test_ohne_taste_passiert_nichts():
    assert action_from_keys(links=False, rechts=False, schub=False) == NICHTS


def test_links_und_rechts_schieben_zur_seite():
    assert action_from_keys(links=True, rechts=False, schub=False) == LINKS
    assert action_from_keys(links=False, rechts=True, schub=False) == RECHTS


def test_schub_zuendet_das_hauptriebwerk():
    assert action_from_keys(links=False, rechts=False, schub=True) == SCHUB


def test_links_und_rechts_gleichzeitig_heben_sich_auf():
    assert action_from_keys(links=True, rechts=True, schub=False) == NICHTS


def test_seitenduese_schlaegt_schub():
    # Es gibt nur *eine* Aktion pro Frame — genau wie beim Agenten. Wer den Schub
    # gedrückt hält und kurz zur Seite tippt, will dass der Tipper ankommt.
    assert action_from_keys(links=True, rechts=False, schub=True) == LINKS
    assert action_from_keys(links=False, rechts=True, schub=True) == RECHTS


def test_links_und_rechts_gleichzeitig_lassen_den_schub_durch():
    assert action_from_keys(links=True, rechts=True, schub=True) == SCHUB
