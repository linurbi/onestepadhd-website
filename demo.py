"""
OneStep — Stuckness Score Interactive Demo

Simulates a session that walks through 5 physiological states and prints
a real-time dashboard showing raw features and the score for each window.
Optionally generates a matplotlib chart of the score over time.

Usage:
    python demo.py           # text-only dashboard
    python demo.py --plot    # text + chart saved to results/demo_chart.png
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stuckness_score import (
    SignalSimulator, PhysiologicalState, SignalConfig,
    CalibrationProtocol, BaselineProfile,
    StucknessPipeline, PipelineResult,
)
from stuckness_score.classifier import StucknessType


# ---------------------------------------------------------------------------
# Demo scenario
# ---------------------------------------------------------------------------

DEMO_SCENARIO = [
    (PhysiologicalState.REST,            4, "Resting quietly"),
    (PhysiologicalState.ENGAGED,         4, "Playing a favourite game"),
    (PhysiologicalState.STUCK_FREEZE,    6, "Impossible maths task (freeze)"),
    (PhysiologicalState.ACTIVE_MOVEMENT, 3, "Running around the room"),
    (PhysiologicalState.ENGAGED,         3, "Back to an easy game"),
    (PhysiologicalState.STUCK_AGITATED,  6, "Confusing instructions (agitated)"),
    (PhysiologicalState.REST,            3, "Calm-down break"),
]

SCORE_THRESHOLD = 55.0
BAR_WIDTH = 40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_bar(score: float) -> str:
    filled = int(score / 100.0 * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"[{bar}] {score:5.1f}"


def state_label(state: PhysiologicalState) -> str:
    return {
        PhysiologicalState.REST:            "REST           ",
        PhysiologicalState.ENGAGED:         "ENGAGED        ",
        PhysiologicalState.STUCK_FREEZE:    "STUCK (freeze) ",
        PhysiologicalState.STUCK_AGITATED:  "STUCK (agitated)",
        PhysiologicalState.ACTIVE_MOVEMENT: "ACTIVE MOVEMENT",
    }[state]


def action_label(result: PipelineResult) -> str:
    if not result.trigger_allowed:
        return f"  (guard: {result.guard_reason})"
    stype = result.classification.stuckness_type
    if stype == StucknessType.FREEZE:
        return f"  *** TRIGGER: {result.classification.action_text} (Freeze) ***"
    if stype == StucknessType.AGITATED:
        return f"  *** TRIGGER: {result.classification.action_text} (Agitated) ***"
    return ""


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(plot: bool = False) -> None:
    print("\n" + "=" * 70)
    print("  OneStep - Stuckness Score Demo")
    print("=" * 70)

    # ── Calibration ──────────────────────────────────────────────────────
    print("\n[Step 1] Running 10-minute calibration (simulated)...\n")
    config = SignalConfig()
    calibrator = CalibrationProtocol(config=config, simulate=True, seed=42, verbose=True)
    baseline = calibrator.run()

    # ── Session ───────────────────────────────────────────────────────────
    print("\n\n[Step 2] Live session simulation")
    print("-" * 70)
    print(f"{'Window':>6}  {'State':<18}  {'Score bar':<{BAR_WIDTH+8}}  Action")
    print("-" * 70)

    simulator = SignalSimulator(config=config, seed=7)
    log_path = Path("results/demo_session.ndjson")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    scores: list[float] = []
    states_seq: list[str] = []
    window_idx = 0
    ts = 0.0

    with StucknessPipeline.from_calibration(
        baseline,
        trigger_threshold=SCORE_THRESHOLD,
        log_path=log_path,
        child_id="DEMO",
    ) as pipeline:
        for state, n_windows, description in DEMO_SCENARIO:
            print(f"\n  ─── {description} ({'×'.join(str(n_windows))} windows) ───")

            for _ in range(n_windows):
                window = simulator.generate_window(state, duration=30.0, timestamp=ts)
                ts += 30.0
                result = pipeline.process(window)

                scores.append(result.score)
                states_seq.append(state.value)
                window_idx += 1

                bar = score_bar(result.score)
                trigger_str = action_label(result)
                print(f"  {window_idx:4d}   {state_label(state)}  {bar}{trigger_str}")

    print("\n" + "-" * 70)
    print(f"Session log saved → {log_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n[Summary]")
    print(f"  Windows processed : {window_idx}")
    print(f"  Mean score        : {np.mean(scores):.1f}")
    print(f"  Max score         : {np.max(scores):.1f}")
    print(f"  Windows ≥ {SCORE_THRESHOLD:.0f}     : {sum(1 for s in scores if s >= SCORE_THRESHOLD)}")

    if plot:
        _save_chart(scores, states_seq)


def _save_chart(scores: list, states: list) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("\n[Chart] matplotlib not installed — skipping chart.")
        return

    state_colors = {
        "rest":             "#90CAF9",
        "engaged":          "#A5D6A7",
        "stuck_freeze":     "#EF9A9A",
        "stuck_agitated":   "#FFCC80",
        "active_movement":  "#CE93D8",
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    x = list(range(1, len(scores) + 1))

    # Background colour bands by state
    prev_state = states[0]
    band_start = 0
    for i, s in enumerate(states + [None]):
        if s != prev_state or i == len(states):
            color = state_colors.get(prev_state, "#EEEEEE")
            ax.axvspan(band_start + 0.5, i + 0.5, alpha=0.18, color=color)
            prev_state = s
            band_start = i

    ax.plot(x, scores, color="#1565C0", linewidth=2, zorder=3)
    ax.axhline(60, color="#C62828", linestyle="--", linewidth=1.5, label="Trigger threshold (60)")
    ax.fill_between(x, scores, 60, where=[s >= 60 for s in scores],
                    alpha=0.25, color="#C62828", label="Triggered zone")

    ax.set_xlabel("Window (30 s each)", fontsize=11)
    ax.set_ylabel("Stuckness Score (0–100)", fontsize=11)
    ax.set_title("OneStep — Stuckness Score Demo Session", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xlim(0.5, len(scores) + 0.5)

    legend_patches = [mpatches.Patch(color=c, alpha=0.5, label=s.replace("_", " ").title())
                      for s, c in state_colors.items()]
    ax.legend(handles=legend_patches + ax.get_lines()[-2:], loc="upper left", fontsize=9)

    out = Path("results/demo_chart.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n[Chart] Saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OneStep Stuckness Score demo")
    parser.add_argument("--plot", action="store_true", help="Save score chart to results/")
    args = parser.parse_args()
    run_demo(plot=args.plot)
