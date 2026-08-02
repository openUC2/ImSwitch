
## Engpass-Übersicht (pro Tile beobachtet)

| Phase | Schnell (~70 %) | Langsam (~30 %) |
|---|---|---|
| `Move to point` (XY) | **0.60 s** | **2.06 s** ← Ausreißer |
| `Move to Z base` (Z=0) | 0.09–0.10 s | 0.09–0.10 s |
| `Acquire frame` | 0.30 s | 0.30 s |
| **Summe** | **~1.0 s** | **~2.5 s** |

Mittel ≈ 1.45 s/Tile. Behebung des XY-Ausreißers allein bringt **≈ 30 % Speedup**.

---

## 1. Hauptursache: Race-Condition in mserial.py (kritisch)

Jeder `motor_act` löst auf der ESP32-Seite **drei** JSON-Antworten aus:

1. `{"steppers":[{...,"isDone":0}], "qid":N}`
2. `{"steppers":[{...,"position":X,"isDone":1}], "qid":N}`
3. `{"qid":N,"state":"done"}`

`Motor.move_stepper` setzt `nResponses = len(steppers) + 1` (= 2) und in mserial.py prüft die Warteschleife:

```python
if len(self.responses[identifier]) == nResponses:   # <-- BUG: ==
    return ...
```

**Problem:** Die Antworten 2 und 3 kommen oft im Abstand von nur ~3 ms (siehe Log: `17:05:42,813` → `17:05:42,816`). Bei `time.sleep(0.002)`-Polling springt `len(responses)` häufig von 1 direkt auf 3 — der Vergleich `== 2` wird übersprungen. Danach läuft `timeReturnReceived = 1 s` ab, der Retry-Block läuft auf `raise`, und der nächste Aufruf erbt `resetLastCommand=True` → Log-Zeile *"Communication interrupted by timeout or reset"* → **~2 s verloren**.

Vergleich Logs:
- qid 172 (schnell): Step `completed in 0.62s` exakt zwischen Append der Nachricht 2 und Append der Nachricht 3.
- qid 171 (langsam): Polling hat die Lücke verpasst → `len` wurde 3 → 2 s Timeout.

**Fixes (priorisiert):**

**a) Trivialfix in mserial.py:**
```python
if len(self.responses[identifier]) >= nResponses:
```
Allein das eliminiert die ~30 % langsamen Tiles.

**b) Sauberer Fix:** `self.use_qid_done = True` aktivieren. Der QID-Done-Pfad in mserial.py nutzt `threading.Event` — kein Polling, kein Race. Die Infrastruktur (`qid_done_events`, `qid_done_responses` werden in `_process_commands` korrekt gesetzt) ist bereits vorhanden, aber per Default deaktiviert.

>>>> activate via UC2ConfigController or detect via a firmware flag on /state_get => newer build date than X activate feature

---

## 2. Redundante Z-Bewegung pro Tile (~10 s über 100 Tiles)

Log zeigt bei *jedem* Tile:
```
Motor 3 is already at target position 0
Step Move to Z base position (0.0 µm) ... completed in 0.10s
```

Trotzdem wird ein `motor_act`-Roundtrip ausgelöst. In `experiment_normal_mode.py` den Z-WorkflowStep überspringen, wenn `abs(current_z - target_z) < tol`. Spart ~0.1 s/Tile.

>>>> fix that so that we don't have this behavior 

---

## 3. `_callback_motor_status` wirft jedes Mal `KeyError 'position'`

In motor.py: Die intermediate `isDone:0`-Antwort enthält kein `position`-Feld, der Callback wirft `KeyError`, der Log wird mit *"Error in _callback_motor_status: 'position'"* geflutet.

```python
for iMotor in range(nSteppers):
    stepper = data["steppers"][iMotor]
    if "position" not in stepper:   # intermediate isDone:0 → skip
        continue
    stepperID = stepper["stepperid"]
    self.currentPosition[stepperID] = stepper["position"] * stepSizes[stepperID]
```

Funktional harmlos, aber Print-I/O auf jeder Bewegung kostet Zeit und verschleiert echte Fehler.

>>>> fix this 
---

## 4. Sequentielle Workflow-Phasen

`Move XY → Z-check → Acquire` laufen strikt seriell. Optimierungsmöglichkeiten:

- **XY und Z in einem `motor_act`** kombinieren (Mehrachs-Move geht schon, wird teilweise genutzt z. B. qid 195).
- **Acquire-Trigger früher feuern**: Sobald `state:"done"` callback eintrifft, statt erst nach Rückkehr des WorkflowSteps.
- **Doppel-Puffer**: nächste Position schon berechnen / Frame schon abrufen, während Stage settled.

Das ist aber ein Refactor — erst Punkte 1–3 angehen.

---

## 5. Kamera-Acquire ~0.30 s

Konstant pro Tile. Wenn die Kamera frei läuft, wartest du im Schnitt eine halbe Frame-Period. Mit Hardware-Trigger oder kürzerer Belichtung (siehe `exposure`-Settings) ggf. halbierbar. Niedrigste Priorität.

---

## Empfohlene Reihenfolge

1. **`== nResponses` → `>= nResponses`** in mserial.py — *1 Zeile, ~30 % Speedup*.
2. **`use_qid_done = True`** aktivieren (oder beim Init setzen) — robuster für die Zukunft.
3. **Z-Step skippen**, wenn Position bereits stimmt.
4. **`_callback_motor_status` gegen fehlendes `position` absichern**.
5. Erst dann an Pipeline-Overlap (Punkt 4) denken.

Soll ich die Fixes 1, 3 und 4 direkt umsetzen?