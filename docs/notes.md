# Projekt-Notizen: Lunar Lander DQN vs. PPO

Lebendes Dokument. Hier sammeln wir **Konzepte, Entscheidungen und Erkenntnisse** —
als gemeinsames Verständnis und als Rohmaterial fürs Poster. Neue Fragen, die
auftauchen, wandern als kurze Erklärung hier rein.

Inhalt:
1. Problem & Environment
2. Algorithmen (DQN vs. PPO)
3. Hyperparameter-Tuning (Optuna)
4. Training & Modellauswahl (best vs. final)
5. Evaluation & Metriken
6. Mehrere Seeds & Statistik
7. Ergebnisse (laufend)
8. Entscheidungs-Log

---

## 1. Problem & Environment

- **`LunarLander-v3`** (Gymnasium, Box2D): ein Raumschiff sanft auf einer Plattform landen.
- **Zustandsraum:** 8-dimensional, **kontinuierlich** (x/y-Position, x/y-Geschwindigkeit, Winkel, Winkelgeschwindigkeit, 2 Bein-Kontakt-Flags).
- **Aktionsraum:** **4 diskret** (nichts / linkes Triebwerk / Haupttriebwerk / rechtes Triebwerk).
- **„Gelöst":** mittlere Rendite ≥ **200** über 100 Episoden.
- **Reward grob:** Punkte fürs Annähern an die Plattform und sanftes Landen (+100–140), Bonus pro Bein-Kontakt (+10), Abzug für Spritverbrauch und Absturz (−100).

**Kernidee (Poster):** Weil die Zustände *kontinuierlich* sind, ist keine Q-Tabelle möglich → man braucht ein **neuronales Netz als Funktionsapproximator**. Das ist der Übergang von tabellarischem Q-Learning zu **Deep** Q-Learning.

---

## 2. Algorithmen: DQN vs. PPO

| | **DQN** (Deep Q-Network) | **PPO** (Proximal Policy Optimization) |
|---|---|---|
| Typ | **wert-basiert** | **policy-gradient / actor-critic** |
| Lernt | Q-Werte pro Aktion, wählt `argmax` | direkt eine Policy (Aktions-Wahrscheinlichkeiten) |
| Braucht | **diskrete** Aktionen | diskret *oder* kontinuierlich |
| Datennutzung | off-policy (Replay-Buffer) | on-policy (frische Rollouts) |
| Ruf | sample-effizient, aber **instabil** | stabiler, robuster |

**Warum DQN nur mit diskreten Aktionen?** DQN wählt die Aktion per `argmax` über die Q-Werte — das setzt eine endliche Aktionsmenge voraus. Beim *kontinuierlichen* LunarLander (2 reelle Steuerwerte) gäbe es unendlich viele Aktionen → kein `argmax`. Dafür bräuchte man SAC/TD3/PPO. Wir nutzen bewusst die **diskrete** Variante.

### Warum DQN instabil ist, PPO stabil
(Erklärt unseren Befund: DQN wird im Training zwischendurch wieder schlechter, PPO steigt gleichmäßig.)

**DQN — bricht ein:**
- **„Deadly triad"** (Bootstrapping + Off-policy + Funktionsapproximation): diese drei zusammen haben keine Konvergenzgarantie → Werte können oszillieren/divergieren.
- **`max`-Überschätzung:** `max_a Q(s',a')` greift auch verrauschte Höchstwerte ab → Bias schaukelt sich auf.
- **Kein Zaun um die Policy:** Policy = `argmax` über Q — kleine Q-Änderungen können den `argmax` in vielen Zuständen *gleichzeitig* umklappen → abrupte Einbrüche.
- **Catastrophic forgetting:** neue Buffer-Daten überschreiben gute alte Repräsentationen.

**PPO — steigt stetig:**
- **On-policy:** trainiert nur auf frischen Rollouts der aktuellen Policy (keine veralteten Daten).
- **Geclippte Zielfunktion (`clip_range`):** begrenzt *explizit*, wie weit sich die Policy pro Update bewegt → kleine, sichere Schritte statt riskanter Sprünge (Trust-Region-Idee). **Das ist der Hauptgrund.**
- **Direkte Policy-Optimierung:** Gradientenaufstieg auf die Leistung selbst → glatte, ~monotone Verbesserung.

**Einschränkungen (ehrlich):** PPO ist nicht garantiert monoton (kann plateauen/kleine Dellen), aber katastrophale Einbrüche sind selten. DQN lässt sich stabilisieren (Double DQN, größeres `target_update_interval`, Prioritized Replay …) — „nacktes" DQN ist aber fragiler, weshalb die **Best-Checkpoint-Auswahl** bei uns essenziell ist.

---

## 3. Hyperparameter-Tuning (Optuna)

- **Trial** = ein Durchlauf mit *einer* Hyperparameter-Kombination (kurz trainieren + bewerten). `--trials N` = wie viele Kombinationen probiert werden.
- **TPE-Sampler:** nicht zufällig — nutzt die Ergebnisse bisheriger Trials, um die nächste Kombination gezielt in vielversprechende Regionen zu legen.
- **MedianPruner:** bricht aussichtslose Trials **früh** ab (wenn ihr Zwischenstand schlechter ist als der Median vergleichbarer Trials). Spart viel Rechenzeit. Greift erst ab Trial 6 (die ersten 5 laufen voll durch, `n_startup_trials=5`).
- Läufe sind **fortsetzbar** und in einer SQLite-`.db` gespeichert; mehrere Terminals auf dieselbe DB = echte Parallelität.

**Beobachtung:** Beim DQN-Tuning wurden **23 von 30 Trials gepruned** — die meisten Kombinationen sind schlecht, der Pruner spart sie effizient weg.

---

## 4. Training & Modellauswahl: `best_model` vs. `final_model`

**Der Mechanismus (SB3 `EvalCallback`):** Alle `EVAL_FREQ = 10.000` Steps hält das Training kurz an, bewertet die aktuelle Policy über 20 Episoden und **speichert das Modell nur dann nach `best_model.zip`, wenn dieser Wert besser ist als der bisher beste.** Es gibt am Ende genau *eine* Datei (der beste Checkpoint), kein Snapshot pro Step.

- `best_model.zip` = Gewichte vom besten Bewertungszeitpunkt (z. B. Step 230k).
- `final_model.zip` = Gewichte am Trainingsende (Step 500k).

**Warum das wichtig ist — catastrophic forgetting:** DQN ist instabil. Es erreicht gute Leistung und **verlernt sie teilweise wieder**. Das *finale* Modell erwischt oft einen schlechteren Zustand als der beste Checkpoint.

**Unsere Entscheidung:** Wir berichten das **beste Modell (Early Stopping)** — Standard in RL, fairer und stabiler. `run_final_eval --which best` (Default). `--which final` und `--eval-only` erlauben den Vergleich ohne Neu-Training.

**Beleg (DQN, 5 Seeds, 100 Eval-Episoden):**

| Metrik | Mittel | Streuung | gelöst? |
|---|---|---|---|
| finales Modell | 152.5 | ± 65.0 | nur 1/5 Seeds > 200 |
| **bestes Modell** | **236.1** | **± 9.9** | **alle 5 Seeds** |

→ Die Wahl der Metrik kippt die Aussage komplett. Der Gap (152 → 236) ist selbst ein **Poster-Insight**: „Bei DQN ist Best-Checkpoint-Auswahl essenziell."

---

## 5. Evaluation & Metriken (TensorBoard)

- **`rollout/ep_rew_mean`** — mittlere Episoden-Rendite *während des Trainings* (mit Exploration, ε-greedy), gemittelt über die letzten 100 Episoden. Der Live-Fortschrittsbalken „lernt er?".
- **`eval/mean_reward`** — deterministische Bewertung alle 10k Steps (ohne Exploration). Die **offizielle** Zahl für „gelöst ≥ 200"; `best_model` = höchster Punkt dieser Kurve.
- **`rollout/ep_len_mean`** — mittlere Episodenlänge in Steps. Nur zusammen mit dem Reward aussagekräftig: kurz = stürzt schnell ab; ~1000 (Zeitlimit) = schwebt/kreist ohne zu landen; mittel = gezielte Landung.
- Weil `rollout/...` Explorations-Zufallsaktionen enthält, liegt es meist *unter* `eval/...`.

### Zufalls-Baseline (der Boden)

**Wozu?** „219 Punkte" ist für sich genommen bedeutungslos — die Zahl bekommt erst Sinn im
Abstand zu dem, was *ohne jedes Lernen* herauskommt. Die Zufalls-Policy beantwortet die Frage
„ist das Ergebnis gelernt oder nur die Umgebung?" und ist damit die untere Referenz, gegen die
die 200er-Schwelle oben die Gegenprobe bildet.

- `agents.make_random_agent(env, seed)` zieht gleichverteilt aus dem Aktionsraum. Sie bietet
  nur `predict()` — genau das, was SB3s `evaluate_policy` braucht. Damit läuft die Baseline
  durch **dasselbe** `evaluate()` wie DQN/PPO, ohne dort eine Sonderbehandlung zu erzwingen
  (die Pipeline bleibt algo-blind).
- `deterministic=True` wird bewusst ignoriert: eine deterministische Zufalls-Policy wäre keine.
- Gemessener Boden auf LunarLander-v3: **≈ −196** (10 Episoden, Seed 12345). Der Wert schwankt
  je nach Stichprobe; die im Repo verstreute Konstante `baseline=-120` in den Plot-Aufrufen ist
  eine ältere, gröbere Schätzung derselben Größe.
- Abstand best_model → Zufall: DQN ≈ +415, PPO ≈ +463. Beide lernen also *deutlich* — der
  Unterschied zwischen DQN und PPO ist klein gegen den Abstand zum Nichtstun.

---

## 6. Mehrere Seeds & Statistik

**Warum mehrere Seeds?** RL ist stark zufallsabhängig (Netz-Init, Exploration, Env-Resets, Replay-Sampling). Ein einzelner Lauf ist nur eine Stichprobe — Glückstreffer möglich. Mehrere Seeds geben eine **Verteilung**, die man mitteln, mit Unsicherheit versehen und **vergleichen** kann.

- Ohne mehrere Seeds **kein Signifikanztest** — der Welch-Test braucht je eine Stichprobe pro Algorithmus.
- DQN und PPO laufen über **dieselben Seeds** `[0..4]` → fairer Vergleich (keiner erwischt „glücklichere" Seeds).

**Statistik (in `stats.py`):**
- **Welch's t-Test** (ungleiche Varianzen) → *p*-Wert: „ist der Unterschied zufällig?"
- **95%-Konfidenzintervalle** je Gruppe **und für die Differenz** (schließt die Differenz die 0 nicht ein → signifikant).
- **Cohen's d** → Effektstärke: *wie groß* ist der Unterschied (0.2 klein, 0.5 mittel, 0.8+ groß).
- **Limitierung:** Mit n=5 ist die statistische Power gering — nur *große* Effekte sind sicher nachweisbar. Hebel bei Bedarf: Seeds auf 8–10 erhöhen (Pipeline kann das ohne Code-Änderung; Hardware ist schnell genug).

---

## 7. Ergebnisse

Beide getunt (30 Trials), finaler Lauf über 5 Seeds, je 100 Eval-Episoden.
Beste Tuning-Rendite (150k Steps): DQN 186.7, PPO 212.6.

### Endleistung (mittlere Rendite über 5 Seeds)

| Modell | DQN | PPO |
|---|---|---|
| **bestes** (Early Stopping) | **236.1 ± 9.9** (5/5 gelöst) | **240.1 ± 18.2** (5/5 gelöst) |
| finales (Trainingsende) | 152.5 ± 65.0 (1/5 gelöst) | 218.6 ± 39.4 (4/5 gelöst) |

→ **Beide lösen LunarLander** zuverlässig (bestes Modell, alle Seeds > 200).

### Signifikanz (bestes Modell, DQN vs. PPO)
- Welch-t-Test: **p = 0.684** → **nicht signifikant**.
- Cohen's d = −0.27 (kleiner Effekt), CI(DQN−PPO) = **(−26.5, +18.6)** → schließt die 0 ein.
- Mann-Whitney-U (robust): p = 1.000.
→ **Kein statistisch nachweisbarer Unterschied in der Endleistung.**

### Sample-Effizienz (Steps bis mittlere Rendite ≥ 200)
- DQN ≈ **220k**, PPO ≈ **180k** → **PPO erreicht das Ziel etwas früher.**

### Stabilität (bestes → finales Modell)
- DQN: 236 → **152** (−84, nur noch 1/5 gelöst) — starkes **catastrophic forgetting**.
- PPO: 240 → **219** (−21, noch 4/5 gelöst) — deutlich stabiler.
- Bestes Modell über **10 Durchläufe**: DQN **219.1 ± 96.8** (2 von 10 Episoden Crashes: 38, 20), PPO **266.9 ± 12.6** (kein Crash).
→ **PPO ist verlässlicher / robuster**, DQN gelegentlich mit Totalausfällen.

### Suchraum-Erweiterung (PPO) — belegte Tuning-Lektion
Das PPO-Tuning zeigte: die wichtigen Parameter `n_epochs`, `net_arch`, `n_steps` lagen am
**Rand** des Suchraums (siehe `docs/figures/range_widening_lesson`). Nach **Erweitern des
Suchraums** und erneutem Tuning, fair bei **gleichem 500k-Budget**:

| PPO (bestes Modell, 5 Seeds) | Rendite |
|---|---|
| Standard-Suchraum | 240.1 ± 18.2 |
| **erweiterter Suchraum** | **253.9 ± 22.0** (+13.9) |

→ Die Verbesserung kommt aus dem **Suchraum**, nicht aus mehr Rechenzeit (isoliert bei 500k).
Bei n=5 nicht signifikant (d = 0.69, CI schließt 0 ein), aber klarer Effekt. `n_epochs` (20→30)
und `net_arch` ([128,128]→[256,256]) wanderten in den neu geöffneten Bereich.
Grafiken: `search_space_improvement`, `range_widening_comparison`.

### Kernaussage fürs Poster
Wert-basiert (DQN) und policy-gradient (PPO) **lösen LunarLander beide gleich gut** —
der Unterschied in der Endleistung ist **statistisch nicht signifikant**. PPO ist
jedoch **sample-effizienter** und **stabiler** (weniger Leistungseinbruch am
Trainingsende, keine Totalausfälle). Bei DQN ist die **Best-Checkpoint-Auswahl
essenziell**, sonst fällt die Leistung durch catastrophic forgetting stark ab.

---

## 8. Entscheidungs-Log

| Entscheidung | Begründung |
|---|---|
| **Stable-Baselines3** statt from-scratch | fairer Vergleich (gleiche Bibliothek), Fokus auf Analyse/Statistik statt Debugging; deckt Kursmaterial ab |
| **Diskretes** LunarLander + DQN | DQN braucht diskrete Aktionen; deckt sich mit DQN-Kursmodul |
| DQN vs. PPO als zentraler Vergleich | wert-basiert vs. policy-gradient — lehrreich; liefert den Signifikanztest |
| Optuna, 30 Trials, TPE + MedianPruner | solider Standard; Pruner spart Rechenzeit |
| 5 Seeds, schlanker Umfang | reicht für sauberen Test; Laptop-CPU; erweiterbar auf 8–10 |
| **best_model** als Hauptmetrik | fairer/stabiler; umgeht DQNs catastrophic forgetting |
| Tuning ohne TensorBoard, finaler Lauf mit | 30 Trials = zu viele Log-Ordner; finale Läufe live beobachtbar |

---

## Offene Fragen / TODO

- [x] PPO tunen + finalen Lauf machen
- [x] Finalen Vergleich (Statistik) interpretieren (siehe Abschnitt 7)
- [ ] Ergebnisse aufs Poster übertragen (Tabellen/Grafiken aus `03_analysis.ipynb`)
- [ ] Demo-GIF vom besten Seed erzeugen (`02_final_run.ipynb` Abschnitt 6)
- [ ] Optional: Robustheits-Experiment (Rendite über Windstärke, `sandbox.ipynb`)
- [ ] Optional: Seeds für mehr statistische Power auf 8–10 erhöhen
