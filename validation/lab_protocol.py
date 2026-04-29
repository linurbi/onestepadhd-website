"""
Lab Validation Protocol — Phase 3a

Objective: Verify that the Stuckness Score rises with self-reported stuckness.

Design:
  - 8–12 children with ADHD, age 8–12
  - Frustration Induction Task (FIT): a computer game where certain levels
    are intentionally unsolvable (unknown to the child)
  - Ground truth: child presses a "I'm stuck" button; post-session parent report
  - Metrics: Precision, Recall, F1 against ground truth labels

This module simulates the protocol with synthetic data and produces a
confusion matrix + Precision/Recall analysis.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running from repo root or validation/ directory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stuckness_score import (
    SignalSimulator, PhysiologicalState, SignalConfig,
    CalibrationProtocol,
    StucknessPipeline,
    SignalLogger,
)


# ---------------------------------------------------------------------------
# Study parameters
# ---------------------------------------------------------------------------

N_PARTICIPANTS = 10
WINDOWS_PER_STATE = 10   # 10 × 30s = 5 minutes per task block

# Scenario sequence per participant: rest → engaged → stuck → rest → stuck_agitated → engaged
SCENARIO_SEQUENCE = [
    (PhysiologicalState.REST,            "not_stuck"),
    (PhysiologicalState.ENGAGED,         "not_stuck"),
    (PhysiologicalState.STUCK_FREEZE,    "stuck"),
    (PhysiologicalState.REST,            "not_stuck"),
    (PhysiologicalState.STUCK_AGITATED,  "stuck"),
    (PhysiologicalState.ENGAGED,         "not_stuck"),
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParticipantResult:
    participant_id: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class StudyResults:
    participants: list = field(default_factory=list)
    target_precision: float = 0.70
    target_recall: float = 0.60

    @property
    def mean_precision(self) -> float:
        return float(np.mean([p.precision for p in self.participants]))

    @property
    def mean_recall(self) -> float:
        return float(np.mean([p.recall for p in self.participants]))

    @property
    def mean_f1(self) -> float:
        return float(np.mean([p.f1 for p in self.participants]))

    @property
    def passed(self) -> bool:
        return (
            self.mean_precision >= self.target_precision
            and self.mean_recall >= self.target_recall
        )


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

class LabProtocol:
    """
    Simulates the lab frustration induction protocol for N participants.

    In production this class would read from actual wristband hardware
    and a "stuck button" event stream. Here it uses the SignalSimulator
    with ground truth state labels.
    """

    def __init__(
        self,
        n_participants: int = N_PARTICIPANTS,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.n_participants = n_participants
        self.output_dir = output_dir or Path("results/lab")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def run(self) -> StudyResults:
        study = StudyResults()

        for i in range(self.n_participants):
            pid = f"P{i+1:03d}"
            result = self._run_participant(pid, seed=i * 100 + 7)
            study.participants.append(result)
            if self.verbose:
                self._print_participant_summary(result)

        self._print_study_summary(study)
        self._save_results(study)
        return study

    # ------------------------------------------------------------------

    def _run_participant(self, pid: str, seed: int) -> ParticipantResult:
        config = SignalConfig()
        simulator = SignalSimulator(config=config, seed=seed)

        # Calibrate
        calibrator = CalibrationProtocol(config=config, simulate=True, seed=seed, verbose=False)
        baseline = calibrator.run()

        log_path = self.output_dir / f"{pid}_signals.ndjson"
        result = ParticipantResult(participant_id=pid)
        ts = 0.0

        with StucknessPipeline.from_calibration(
            baseline, log_path=log_path, child_id=pid, trigger_threshold=55.0
        ) as pipeline:
            for state, ground_truth in SCENARIO_SEQUENCE:
                for _ in range(WINDOWS_PER_STATE):
                    window = simulator.generate_window(state, duration=30.0, timestamp=ts)
                    ts += 30.0

                    pr = pipeline.process(window, ground_truth=ground_truth)

                    predicted_stuck = pr.trigger_allowed or pr.score >= pipeline.scorer.threshold
                    actually_stuck = (ground_truth == "stuck")

                    if predicted_stuck and actually_stuck:
                        result.true_positives += 1
                    elif predicted_stuck and not actually_stuck:
                        result.false_positives += 1
                    elif not predicted_stuck and actually_stuck:
                        result.false_negatives += 1
                    else:
                        result.true_negatives += 1

        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _print_participant_summary(r: ParticipantResult) -> None:
        print(
            f"  {r.participant_id}  "
            f"TP={r.true_positives:3d} FP={r.false_positives:3d} "
            f"FN={r.false_negatives:3d} TN={r.true_negatives:3d}  "
            f"Prec={r.precision:.2f}  Rec={r.recall:.2f}  F1={r.f1:.2f}"
        )

    @staticmethod
    def _print_study_summary(study: StudyResults) -> None:
        print("\n" + "=" * 60)
        print("LAB STUDY SUMMARY")
        print("=" * 60)
        print(f"  Mean Precision : {study.mean_precision:.3f}  (target ≥ {study.target_precision})")
        print(f"  Mean Recall    : {study.mean_recall:.3f}  (target ≥ {study.target_recall})")
        print(f"  Mean F1        : {study.mean_f1:.3f}")
        status = "PASSED" if study.passed else "NEEDS TUNING"
        print(f"  Result         : {status}")
        print("=" * 60)

    def _save_results(self, study: StudyResults) -> None:
        out = {
            "mean_precision": study.mean_precision,
            "mean_recall": study.mean_recall,
            "mean_f1": study.mean_f1,
            "passed": study.passed,
            "participants": [asdict(p) for p in study.participants],
        }
        path = self.output_dir / "study_summary.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"\n[Lab] Results saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OneStep — Lab Validation Protocol (Phase 3a)")
    print("=" * 60)
    protocol = LabProtocol(verbose=True)
    results = protocol.run()
