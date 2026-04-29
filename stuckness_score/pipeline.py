"""
Main Stuckness Score pipeline orchestrator.

Ties together:
  features → score → classifier → guards → logger

Usage:
    pipeline = StucknessPipeline.from_calibration(baseline)
    result = pipeline.process(window)
    if result.trigger_allowed:
        deliver_micro_action(result.classification)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

from .signals import SignalWindow, SignalConfig
from .features import extract_all_features, CombinedFeatures
from .calibration import BaselineProfile
from .score import StucknessScorer, ScoreWeights
from .classifier import StucknessClassifier, ClassificationResult, StucknessType
from .guards import StucknessGuard, GuardConfig
from .logger import SignalLogger


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    features: CombinedFeatures
    score: float
    classification: ClassificationResult
    trigger_allowed: bool
    guard_reason: str
    timestamp: float


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class StucknessPipeline:
    """
    Processes a single SignalWindow end-to-end and returns a PipelineResult.

    Instantiate via `StucknessPipeline.from_calibration(baseline)` for the
    standard setup, or supply individual components for custom configurations.
    """

    def __init__(
        self,
        scorer: StucknessScorer,
        classifier: StucknessClassifier,
        guard: StucknessGuard,
        config: Optional[SignalConfig] = None,
        logger: Optional[SignalLogger] = None,
    ):
        self.scorer = scorer
        self.classifier = classifier
        self.guard = guard
        self.config = config or SignalConfig()
        self.logger = logger

    # ------------------------------------------------------------------

    @classmethod
    def from_calibration(
        cls,
        baseline: BaselineProfile,
        weights: Optional[ScoreWeights] = None,
        trigger_threshold: float = 60.0,
        guard_config: Optional[GuardConfig] = None,
        signal_config: Optional[SignalConfig] = None,
        log_path: Optional[Path] = None,
        child_id: str = "unknown",
    ) -> "StucknessPipeline":
        scorer = StucknessScorer(baseline, weights, trigger_threshold)
        classifier = StucknessClassifier(baseline)
        guard = StucknessGuard(trigger_threshold, guard_config)
        logger = SignalLogger(log_path, child_id) if log_path else None
        return cls(scorer, classifier, guard, signal_config, logger)

    # ------------------------------------------------------------------

    def process(
        self,
        window: SignalWindow,
        ground_truth: Optional[str] = None,
    ) -> PipelineResult:
        """Process a single 30-second window and return a full result."""
        ts = window.timestamp or time.time()
        cfg = self.config

        # 1. Feature extraction
        # Use personal freeze threshold from baseline when available
        freeze_thr = getattr(self.scorer.baseline, "freeze_energy_threshold", None)
        features = extract_all_features(
            window.eda, window.ppg, window.acc,
            eda_fs=cfg.eda_fs, ppg_fs=cfg.ppg_fs, acc_fs=cfg.acc_fs,
            freeze_threshold=freeze_thr,
        )

        # 2. Score
        score = self.scorer.compute(features)

        # 3. Classify (only meaningful when score >= threshold)
        classification = self.classifier.classify(
            score, features, self.scorer.threshold
        )

        # 4. Guard check
        allowed, reason = self.guard.should_trigger(
            score=score,
            movement_energy=features.acc.movement_energy,
            current_time=ts,
        )

        if allowed and classification.stuckness_type != StucknessType.NONE:
            self.guard.record_trigger(current_time=ts)

        # 5. Log
        if self.logger:
            self.logger.log(
                features=features,
                score=score,
                classification=classification,
                guard_allowed=allowed,
                guard_reason=reason,
                ground_truth=ground_truth,
                timestamp=ts,
            )

        return PipelineResult(
            features=features,
            score=score,
            classification=classification,
            trigger_allowed=allowed and classification.stuckness_type != StucknessType.NONE,
            guard_reason=reason,
            timestamp=ts,
        )

    def close(self) -> None:
        if self.logger:
            self.logger.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
