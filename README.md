# OneStep — Stuckness Score System

Real-time physiological stuckness detection for the OneStep ADHD wristband.
Identifies moments when a child's brain "gets stuck" — and delivers one small
action to help them resume, within seconds.

> **Core principle:** We don't ask the child to try harder.
> We give them a system that works even when they can't.

---

## Overview

Children with ADHD repeatedly experience cognitive stuckness — a state where
they can't start or continue a task, often accompanied by physical freezing or
restless fidgeting. This system detects that state in real time from three
wrist-worn biosensors and triggers a targeted micro-action before the stuckness
becomes a full emotional breakdown.

**Detection pipeline:**

```
EDA (skin conductance)
PPG (heart rate / HRV)       →  Feature extraction (30s rolling window)
ACC (wrist accelerometer)

        ↓

Personal baseline normalization   ← calibrated per child, first session only

        ↓

Stuckness Score   0 ──────────────────── 100
                                         ▲ threshold (55)

        ↓ (if score ≥ threshold)

Classify subtype:   FREEZE   or   AGITATED

        ↓ (pass through 3 guards)

Micro-action:   vibration + on-screen Hebrew cue
```

---

## Stuckness Score

### Formula

```
S = clip(
    0.30 × norm(SCL_slope)       # EDA tonic trend — rising arousal over time
  + 0.20 × norm(SCR_count)       # EDA phasic spikes — acute frustration bursts
  + 0.30 × (1 − RMSSD_ratio)     # HRV drop — cognitive overload without movement
  + 0.20 × movement_pattern      # Freeze=1.0  /  Fidget=0.5  /  Normal=0
, 0, 100)
```

All four components are normalised to [0, 1] against the child's **personal
baseline** (captured in a 10-minute first-session calibration). Population
norms are never used — every child's physiology is measured relative to
themselves.

### Why these signals?

| Signal | What it measures | Why it matters for stuckness |
|--------|-----------------|------------------------------|
| SCL slope (EDA tonic) | Slow rise in skin conductance level | Sustained stress builds up over minutes — the most reliable long-duration signal |
| SCR count (EDA phasic) | Sudden conductance spikes | Marks acute frustration moments; bursts increase before a child gives up |
| RMSSD drop (HRV) | Decrease in beat-to-beat heart rate variability | Drops under cognitive load without physical movement — distinguishes mental stress from exercise |
| Movement pattern | Freeze or 1–3 Hz wrist oscillation | Physical signature of the two stuck subtypes |

### Two subtypes and their micro-actions

| Subtype | Physiological signature | Trigger threshold | Micro-action |
|---------|------------------------|-------------------|--------------|
| **Freeze** | Low wrist energy + rising EDA | score ≥ 55 AND freeze_flag | Vibration + **"הזז אצבע אחת"** |
| **Agitated** | Fidget 1–3 Hz + SCR bursts | score ≥ 55 AND fidget_score > threshold | Vibration + **"שאיפה אחת, לאט"** |

The micro-action type is matched to the subtype because the two states require
different interruptions: a frozen child needs a motor cue to unfreeze; an
agitated child needs a calming breath to de-escalate before resuming.

---

## Quickstart

Requires Python 3.10+ and the `py` launcher (Windows) or `python3`.

```bash
# Install dependencies
py -3 -m pip install numpy scipy matplotlib

# Run the interactive demo (text dashboard + score chart)
py -3 demo.py --plot

# Run lab validation — Phase 3a
py -3 validation/lab_protocol.py

# Run ecological validation — Phase 3b
py -3 validation/ecological_protocol.py
```

The demo simulates a full session across five states, prints a live score
dashboard, and saves a chart to `results/demo_chart.png`.

---

## Personal calibration

Every child runs a single 10-minute calibration session on first use.
No parameters are set manually — the protocol captures their physiology.

| Phase | Duration | Task | What is captured |
|-------|----------|------|-----------------|
| REST | 3 min | Sitting, watching a favourite video | Resting SCL, HR, HRV, ACC energy |
| ENGAGED | 4 min | Playing a game they enjoy | Focused-but-not-stuck reference |
| STUCK reference | 3 min | Intentionally unsolvable puzzle | Frustrated/stuck ceiling values |

The resulting `BaselineProfile` is saved as JSON and reloaded automatically
on all subsequent sessions.

---

## False-positive guards

Three mechanisms prevent the child from being over-triggered and learning to
ignore the cues:

1. **Physical activity veto** — suppresses all triggers when wrist movement
   energy exceeds 0.45 m²/s⁴ (running, gross motor activity). Fidgeting
   (~0.10–0.30) is explicitly allowed through, because it is a stuckness signal.

2. **Cooldown** — enforces a minimum 3-minute gap between consecutive triggers.
   After a trigger fires, no further trigger can occur regardless of score.

3. **Adaptive threshold** — after 3 consecutive dismissed triggers (child
   swipes away), the threshold rises by 5 points (max +20 total). Resets
   downward by 1 point on each accepted trigger.

---

## Validation results (simulated)

Simulations use synthetic physiological signals with physiologically-grounded
parameters derived from the research literature on EDA, HRV, and ADHD.

### Phase 3a — Lab study

10 simulated participants, 60 windows each (mix of all five states).
Ground truth: "stuck" label from the `PhysiologicalState` simulator.

| Metric | Result | Target |
|--------|--------|--------|
| Mean Precision | **0.896** | ≥ 0.70 |
| Mean Recall | **0.695** | ≥ 0.60 |
| Mean F1 | 0.763 | — |
| Result | **PASSED** | |

### Phase 3b — Ecological study

18 simulated participants, 14 days of continuous use, realistic daily
state distribution (20% rest, 45% engaged, 25% stuck, 10% active).

| Metric | Result | Target |
|--------|--------|--------|
| Mean FP Rate | **0.183** | < 0.30 |
| Mean Detection Rate | 0.260 | — |
| Mean Band Retention | 0.885 | — |
| Result | **PASSED** | |

---

## File structure

```
stuckness_score/
  signals.py         — Signal simulator: 5 physiological states
  features.py        — EDA, PPG, ACC feature extraction (per 30s window)
  calibration.py     — 10-min personal baseline protocol (3 phases)
  score.py           — Stuckness Score formula + ScoreWeights dataclass
  classifier.py      — Freeze / Agitated classification + micro-action text
  guards.py          — Activity veto, cooldown, adaptive threshold
  pipeline.py        — End-to-end orchestrator (features → score → action)
  logger.py          — NDJSON per-window session log

validation/
  lab_protocol.py          — Phase 3a: FIT task, confusion matrix, P/R
  ecological_protocol.py   — Phase 3b: 14-day simulation, FP rate

demo.py              — Interactive demo: text dashboard + matplotlib chart
requirements.txt     — numpy, scipy, matplotlib
results/             — Auto-created: session logs, charts, study summaries
```

---

## Connecting real hardware

`StucknessPipeline.process()` accepts any `SignalWindow` object. To connect
a physical wristband, replace `SignalSimulator.generate_window()` with a
hardware reader that fills the three arrays:

```python
from stuckness_score import SignalWindow, StucknessPipeline
from stuckness_score.calibration import BaselineProfile
from pathlib import Path

baseline = BaselineProfile.load(Path("child_001_baseline.json"))
pipeline = StucknessPipeline.from_calibration(baseline, child_id="child_001")

# Your hardware read loop:
while True:
    eda_samples = wristband.read_eda(duration_s=30)    # shape: (n_eda,)
    ppg_samples = wristband.read_ppg(duration_s=30)    # shape: (n_ppg,)
    acc_samples = wristband.read_acc(duration_s=30)    # shape: (n_acc, 3)

    window = SignalWindow(eda=eda_samples, ppg=ppg_samples, acc=acc_samples)
    result = pipeline.process(window)

    if result.trigger_allowed:
        device.vibrate(result.classification.vibration_pattern)
        display.show(result.classification.action_text)
```

**Recommended sensor specifications:**

| Sensor | Sample rate | Notes |
|--------|-------------|-------|
| EDA (electrodermal activity) | 4–8 Hz | Ventral wrist, resolution 0.01 µS |
| PPG (photoplethysmography) | 64–128 Hz | Green LED; 64 Hz minimum for RMSSD |
| Accelerometer (3-axis MEMS) | 50–100 Hz | Must cover 1–3 Hz fidget band |
