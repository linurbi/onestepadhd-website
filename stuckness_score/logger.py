"""
Signal and score logger.

Writes a newline-delimited JSON log file with one record per window.
Each record contains raw feature values, the Stuckness Score, classifier
output, and guard decision — enabling offline analysis and validation.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .features import CombinedFeatures
from .classifier import ClassificationResult


class SignalLogger:
    """
    Appends one JSON line per processed window to `log_path`.

    File format: newline-delimited JSON (ndjson) — one object per line.
    Compatible with pandas `read_json(lines=True)` for analysis.
    """

    def __init__(self, log_path: Path, child_id: str = "unknown"):
        self.log_path = log_path
        self.child_id = child_id
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = log_path.open("a", encoding="utf-8")

    def log(
        self,
        features: CombinedFeatures,
        score: float,
        classification: Optional[ClassificationResult],
        guard_allowed: bool,
        guard_reason: str,
        ground_truth: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        record = {
            "ts": timestamp or time.time(),
            "child_id": self.child_id,
            "eda": {
                "scl_slope": round(features.eda.scl_slope, 4),
                "scr_count": features.eda.scr_count,
                "scr_amplitude_mean": round(features.eda.scr_amplitude_mean, 4),
            },
            "ppg": {
                "hr_bpm": round(features.ppg.hr_bpm, 1),
                "rmssd_ms": round(features.ppg.rmssd_ms, 1),
            },
            "acc": {
                "movement_energy": round(features.acc.movement_energy, 6),
                "fidget_score": round(features.acc.fidget_score, 4),
                "freeze_flag": features.acc.freeze_flag,
            },
            "score": round(score, 2),
            "stuckness_type": classification.stuckness_type.value if classification else None,
            "micro_action": classification.micro_action.value if classification else None,
            "confidence": round(classification.confidence, 3) if classification else None,
            "guard_allowed": guard_allowed,
            "guard_reason": guard_reason,
            "ground_truth": ground_truth,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
