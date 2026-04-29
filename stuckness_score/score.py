"""
Stuckness Score computation.

Formula (from the OneStep business plan):

    S = clip(
        w1 * norm(SCL_slope)
        + w2 * norm(SCR_count)
        + w3 * (1 - RMSSD_ratio)
        + w4 * movement_pattern_score
    , 0, 100)

All inputs are normalised to [0, 1] relative to the child's personal
BaselineProfile before the weighted sum is computed.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .calibration import BaselineProfile
from .features import CombinedFeatures, ACCFeatures


# ---------------------------------------------------------------------------
# Configurable weights
# ---------------------------------------------------------------------------

@dataclass
class ScoreWeights:
    """
    Component weights (must sum to 1.0).
    Defaults from the plan; can be tuned after lab validation.
    """
    scl_slope: float = 0.30    # EDA tonic trend — most reliable for sustained stuckness
    scr_count: float = 0.20    # EDA phasic events — acute frustration bursts
    hrv_drop:  float = 0.30    # HRV (RMSSD) decrease — cognitive load without movement
    movement:  float = 0.20    # Freeze / fidget pattern

    def __post_init__(self) -> None:
        total = self.scl_slope + self.scr_count + self.hrv_drop + self.movement
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"ScoreWeights must sum to 1.0, got {total:.3f}")


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class StucknessScorer:
    """
    Computes Stuckness Score (0–100) from combined features + personal baseline.

    A score ≥ 60 is the default trigger threshold (configurable via `threshold`).
    """

    def __init__(
        self,
        baseline: BaselineProfile,
        weights: Optional[ScoreWeights] = None,
        threshold: float = 60.0,
    ):
        self.baseline = baseline
        self.weights = weights or ScoreWeights()
        self.threshold = threshold

    # ------------------------------------------------------------------

    def compute(self, features: CombinedFeatures) -> float:
        """Return a Stuckness Score in [0, 100]."""
        w = self.weights
        b = self.baseline

        # --- Component 1: SCL slope (EDA tonic trend) ---
        # Normalise to [0, 1]: 0 = resting slope, 1 = stuck reference slope.
        # When slope range is very narrow (< 0.10 µS/min), fall back to
        # SCR amplitude as a proxy for tonic arousal level.
        slope_range = b.scl_slope_stuck - b.scl_slope_rest
        if abs(slope_range) < 0.10:
            # Use SCR amplitude as SCL-level proxy
            scr_amp_range = max(b.scr_count_stuck - b.scr_count_rest, 1.0)
            norm_scl = (features.eda.scr_amplitude_mean / max(features.eda.scr_count, 1)) / 0.4
        else:
            norm_scl = (features.eda.scl_slope - b.scl_slope_rest) / slope_range

        # --- Component 2: SCR count (phasic events) ---
        scr_range = b.scr_count_stuck - b.scr_count_rest
        if scr_range < 0.5:
            norm_scr = 0.0
        else:
            norm_scr = (features.eda.scr_count - b.scr_count_rest) / scr_range

        # --- Component 3: HRV drop (RMSSD ratio, inverted) ---
        # RMSSD_ratio = current / resting; lower ratio = more stress
        rmssd_ratio = features.ppg.rmssd_ms / max(b.rmssd_rest, 1.0)
        # Map ratio to [0, 1]: ratio=1 → 0, ratio≤(stuck/rest) → 1
        stuck_ratio = b.rmssd_stuck / max(b.rmssd_rest, 1.0)
        ratio_range = 1.0 - stuck_ratio
        if ratio_range < 0.01:
            norm_hrv = 0.0
        else:
            norm_hrv = (1.0 - rmssd_ratio) / ratio_range

        # --- Component 4: Movement pattern ---
        norm_movement = _movement_pattern_score(features.acc, b)

        # --- Weighted sum ---
        raw = (
            w.scl_slope * norm_scl
            + w.scr_count * norm_scr
            + w.hrv_drop  * norm_hrv
            + w.movement  * norm_movement
        )

        return float(np.clip(raw * 100.0, 0.0, 100.0))

    def is_stuck(self, score: float) -> bool:
        return score >= self.threshold

    def component_breakdown(self, features: CombinedFeatures) -> dict:
        """Return per-component raw normalised values for debugging / validation."""
        w = self.weights
        b = self.baseline

        slope_range = b.scl_slope_stuck - b.scl_slope_rest
        norm_scl = (features.eda.scl_slope - b.scl_slope_rest) / max(abs(slope_range), 0.01)

        scr_range = b.scr_count_stuck - b.scr_count_rest
        norm_scr = (features.eda.scr_count - b.scr_count_rest) / max(scr_range, 0.5)

        rmssd_ratio = features.ppg.rmssd_ms / max(b.rmssd_rest, 1.0)
        stuck_ratio = b.rmssd_stuck / max(b.rmssd_rest, 1.0)
        ratio_range = 1.0 - stuck_ratio
        norm_hrv = (1.0 - rmssd_ratio) / max(ratio_range, 0.01)

        norm_movement = _movement_pattern_score(features.acc, b)

        return {
            "scl_slope_norm": float(np.clip(norm_scl, 0, 1)),
            "scr_count_norm": float(np.clip(norm_scr, 0, 1)),
            "hrv_drop_norm":  float(np.clip(norm_hrv, 0, 1)),
            "movement_norm":  float(np.clip(norm_movement, 0, 1)),
            "weighted": {
                "scl": w.scl_slope * float(np.clip(norm_scl, 0, 1)),
                "scr": w.scr_count * float(np.clip(norm_scr, 0, 1)),
                "hrv": w.hrv_drop  * float(np.clip(norm_hrv, 0, 1)),
                "mov": w.movement  * float(np.clip(norm_movement, 0, 1)),
            },
        }


# ---------------------------------------------------------------------------
# Movement pattern helper
# ---------------------------------------------------------------------------

def _movement_pattern_score(acc: ACCFeatures, baseline: BaselineProfile) -> float:
    """
    Returns a [0, 1] score for the movement pattern:
        0.0 — normal movement
        0.5 — fidgeting (1–3 Hz oscillation detected)
        1.0 — freeze (near-zero movement with rising EDA context)

    Freeze is weighted higher than fidgeting because it is the strongest
    predictor of the "cognitive shutdown" subtype of stuckness.
    """
    if acc.freeze_flag:
        return 1.0
    if acc.fidget_score > baseline.fidget_threshold:
        return 0.5
    return 0.0
