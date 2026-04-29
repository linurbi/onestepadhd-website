"""
Ecological Validation Protocol — Phase 3b

Objective: Validate the Stuckness Score in naturalistic, home/school use
           over two weeks.

Design:
  - 15–20 children with ADHD, continuous wristband use for 14 days
  - Ground truth: parents fill a daily diary noting stuck episodes
    (time, task context, severity 1–5)
  - Metrics:
      False Positive Rate  — triggers that parents marked as unwarranted
      Time-to-Detect       — lag between diary event and score crossing threshold
      Daily usage retention — % of days child wore the band ≥ 4 hours

Simulated here as 14 days × 16 active hours × 2 windows/hour = 448 windows
per participant, with state sequences that mimic a realistic day.
"""

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stuckness_score import (
    SignalSimulator, PhysiologicalState, SignalConfig,
    CalibrationProtocol, StucknessPipeline,
)


# ---------------------------------------------------------------------------
# Study parameters
# ---------------------------------------------------------------------------

N_PARTICIPANTS = 18
STUDY_DAYS = 14
ACTIVE_HOURS_PER_DAY = 8   # hours the child is actively doing tasks
WINDOWS_PER_HOUR = 2       # 2 × 30s windows = 1 minute of data per "hour slot"
                            # (compressed for simulation speed)

# Probability of each state per "hour slot" during the day
# Reflects a realistic school/homework day
_DAY_STATE_DISTRIBUTION = {
    PhysiologicalState.REST:            0.20,
    PhysiologicalState.ENGAGED:         0.45,
    PhysiologicalState.STUCK_FREEZE:    0.15,
    PhysiologicalState.STUCK_AGITATED:  0.10,
    PhysiologicalState.ACTIVE_MOVEMENT: 0.10,
}

_STUCK_STATES = {PhysiologicalState.STUCK_FREEZE, PhysiologicalState.STUCK_AGITATED}

# False positive rate acceptance ceiling (plan target: < 30%)
FP_RATE_TARGET = 0.30


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DayRecord:
    day: int
    true_stuck_events: int = 0
    detected_events: int = 0
    false_positive_triggers: int = 0
    total_triggers: int = 0
    wore_band: bool = True


@dataclass
class ParticipantResult:
    participant_id: str
    days: list = field(default_factory=list)

    @property
    def total_triggers(self) -> int:
        return sum(d.total_triggers for d in self.days)

    @property
    def total_fp(self) -> int:
        return sum(d.false_positive_triggers for d in self.days)

    @property
    def fp_rate(self) -> float:
        t = self.total_triggers
        return self.total_fp / t if t > 0 else 0.0

    @property
    def detection_rate(self) -> float:
        actual = sum(d.true_stuck_events for d in self.days)
        detected = sum(d.detected_events for d in self.days)
        return detected / actual if actual > 0 else 0.0

    @property
    def retention_rate(self) -> float:
        return sum(1 for d in self.days if d.wore_band) / len(self.days)


@dataclass
class StudyResults:
    participants: list = field(default_factory=list)
    fp_rate_target: float = FP_RATE_TARGET

    @property
    def mean_fp_rate(self) -> float:
        return float(np.mean([p.fp_rate for p in self.participants]))

    @property
    def mean_detection_rate(self) -> float:
        return float(np.mean([p.detection_rate for p in self.participants]))

    @property
    def mean_retention(self) -> float:
        return float(np.mean([p.retention_rate for p in self.participants]))

    @property
    def passed(self) -> bool:
        return self.mean_fp_rate < self.fp_rate_target


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

class EcologicalProtocol:
    """Simulates 14-day ecological study for N participants."""

    def __init__(
        self,
        n_participants: int = N_PARTICIPANTS,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.n_participants = n_participants
        self.output_dir = output_dir or Path("results/ecological")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._rng = random.Random(99)

    def run(self) -> StudyResults:
        study = StudyResults()

        for i in range(self.n_participants):
            pid = f"E{i+1:03d}"
            result = self._run_participant(pid, seed=i * 200 + 13)
            study.participants.append(result)
            if self.verbose:
                print(
                    f"  {pid}  FP_rate={result.fp_rate:.2f}  "
                    f"Detect={result.detection_rate:.2f}  "
                    f"Retention={result.retention_rate:.2f}"
                )

        self._print_study_summary(study)
        self._save_results(study)
        return study

    # ------------------------------------------------------------------

    def _run_participant(self, pid: str, seed: int) -> ParticipantResult:
        config = SignalConfig()
        simulator = SignalSimulator(config=config, seed=seed)
        calibrator = CalibrationProtocol(config=config, simulate=True, seed=seed, verbose=False)
        baseline = calibrator.run()

        log_path = self.output_dir / f"{pid}_signals.ndjson"
        p_result = ParticipantResult(participant_id=pid)
        ts = 0.0

        states = list(_DAY_STATE_DISTRIBUTION.keys())
        probs = list(_DAY_STATE_DISTRIBUTION.values())
        local_rng = np.random.RandomState(seed)

        with StucknessPipeline.from_calibration(
            baseline, log_path=log_path, child_id=pid,
            trigger_threshold=60.0,
        ) as pipeline:
            for day_num in range(1, STUDY_DAYS + 1):
                # Simulate occasional non-wear (10% chance per day)
                wore_band = local_rng.rand() > 0.10
                day = DayRecord(day=day_num, wore_band=wore_band)

                if not wore_band:
                    p_result.days.append(day)
                    continue

                for _ in range(ACTIVE_HOURS_PER_DAY * WINDOWS_PER_HOUR):
                    state = local_rng.choice(states, p=probs)
                    window = simulator.generate_window(state, duration=30.0, timestamp=ts)
                    ts += 30.0

                    ground_truth = "stuck" if state in _STUCK_STATES else "not_stuck"
                    pr = pipeline.process(window, ground_truth=ground_truth)

                    actually_stuck = (ground_truth == "stuck")
                    if actually_stuck:
                        day.true_stuck_events += 1

                    if pr.trigger_allowed:
                        day.total_triggers += 1
                        if actually_stuck:
                            day.detected_events += 1
                        else:
                            day.false_positive_triggers += 1

                p_result.days.append(day)

        return p_result

    # ------------------------------------------------------------------

    @staticmethod
    def _print_study_summary(study: StudyResults) -> None:
        print("\n" + "=" * 60)
        print("ECOLOGICAL STUDY SUMMARY (14-day)")
        print("=" * 60)
        print(f"  Mean FP Rate      : {study.mean_fp_rate:.3f}  (target < {study.fp_rate_target})")
        print(f"  Mean Detection    : {study.mean_detection_rate:.3f}")
        print(f"  Mean Retention    : {study.mean_retention:.3f}")
        status = "PASSED" if study.passed else "NEEDS TUNING"
        print(f"  Result            : {status}")
        print("=" * 60)

    def _save_results(self, study: StudyResults) -> None:
        out = {
            "mean_fp_rate": study.mean_fp_rate,
            "mean_detection_rate": study.mean_detection_rate,
            "mean_retention": study.mean_retention,
            "passed": study.passed,
            "participants": [
                {
                    "id": p.participant_id,
                    "fp_rate": p.fp_rate,
                    "detection_rate": p.detection_rate,
                    "retention_rate": p.retention_rate,
                    "total_triggers": p.total_triggers,
                    "total_fp": p.total_fp,
                }
                for p in study.participants
            ],
        }
        path = self.output_dir / "study_summary.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"\n[Ecological] Results saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OneStep — Ecological Validation Protocol (Phase 3b)")
    print("=" * 60)
    protocol = EcologicalProtocol(verbose=True)
    results = protocol.run()
