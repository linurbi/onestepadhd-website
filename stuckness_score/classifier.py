"""
Stuckness type classifier and micro-action selector.

After the Stuckness Score crosses the trigger threshold, the classifier
distinguishes two subtypes (Freeze vs. Agitated) and picks the matching
micro-action. Different subtypes require different interruptions:

  Freeze    → child is cognitively overloaded, motor system locked.
              Best intervention: gentle motor cue ("move one finger").
  Agitated  → child is frustrated and fidgeting, energy needs redirect.
              Best intervention: brief breathing cue ("one slow breath").
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .features import CombinedFeatures
from .calibration import BaselineProfile


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StucknessType(Enum):
    NONE = "none"
    FREEZE = "freeze"
    AGITATED = "agitated"


class MicroAction(Enum):
    NONE = "none"
    MOTOR_CUE = "motor_cue"       # vibration + "move one finger"
    BREATHING_CUE = "breathing_cue"  # vibration + "one slow breath"


_MICRO_ACTION_TEXT = {
    MicroAction.NONE: "",
    MicroAction.MOTOR_CUE: "הזז אצבע אחת",          # "Move one finger"
    MicroAction.BREATHING_CUE: "שאיפה אחת, לאט",    # "One slow breath"
}

_MICRO_ACTION_VIBRATION_PATTERN = {
    MicroAction.NONE: [],
    MicroAction.MOTOR_CUE: [200, 100, 200],           # ms on/off/on
    MicroAction.BREATHING_CUE: [400, 200, 400],       # slower, calming
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    stuckness_type: StucknessType
    micro_action: MicroAction
    action_text: str
    vibration_pattern: list  # ms durations
    confidence: float        # 0–1, derived from signal clarity


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class StucknessClassifier:
    """
    Classifies a triggered stuckness event into Freeze or Agitated,
    then selects the appropriate micro-action.

    Classification rules (from the plan):
        Freeze    : freeze_flag == True  (low energy + rising EDA)
        Agitated  : fidget_score > threshold  (oscillatory wrist movement)
        Ambiguous : fall through to Freeze (more conservative intervention)
    """

    def __init__(self, baseline: Optional[BaselineProfile] = None):
        self.baseline = baseline

    def classify(
        self,
        score: float,
        features: CombinedFeatures,
        trigger_threshold: float = 60.0,
    ) -> ClassificationResult:
        """
        Classify stuckness type for a given score and features.

        Returns ClassificationResult with type=NONE and action=NONE when
        the score is below the trigger threshold.
        """
        if score < trigger_threshold:
            return ClassificationResult(
                stuckness_type=StucknessType.NONE,
                micro_action=MicroAction.NONE,
                action_text="",
                vibration_pattern=[],
                confidence=0.0,
            )

        fidget_threshold = (
            self.baseline.fidget_threshold if self.baseline else 0.25
        )

        freeze_flag = features.acc.freeze_flag
        fidgeting = features.acc.fidget_score > fidget_threshold

        if freeze_flag and not fidgeting:
            stype = StucknessType.FREEZE
            action = MicroAction.MOTOR_CUE
            confidence = self._freeze_confidence(features)
        elif fidgeting and not freeze_flag:
            stype = StucknessType.AGITATED
            action = MicroAction.BREATHING_CUE
            confidence = self._agitated_confidence(features)
        elif freeze_flag and fidgeting:
            # Contradictory signals — choose by dominant indicator
            if features.acc.movement_energy < (
                self.baseline.freeze_energy_threshold if self.baseline else 0.003
            ) * 2:
                stype = StucknessType.FREEZE
                action = MicroAction.MOTOR_CUE
            else:
                stype = StucknessType.AGITATED
                action = MicroAction.BREATHING_CUE
            confidence = 0.50
        else:
            # Neither clearly freeze nor fidgeting — use SCR count as tiebreaker.
            # High SCR rate + elevated HR suggests agitated; otherwise freeze.
            if (
                features.eda.scr_count >= 6
                and features.ppg.hr_bpm > (
                    self.baseline.hr_stuck if self.baseline else 82.0
                ) * 0.95
            ):
                stype = StucknessType.AGITATED
                action = MicroAction.BREATHING_CUE
            else:
                stype = StucknessType.FREEZE
                action = MicroAction.MOTOR_CUE
            confidence = 0.45

        return ClassificationResult(
            stuckness_type=stype,
            micro_action=action,
            action_text=_MICRO_ACTION_TEXT[action],
            vibration_pattern=_MICRO_ACTION_VIBRATION_PATTERN[action],
            confidence=confidence,
        )

    # ------------------------------------------------------------------

    def _freeze_confidence(self, features: CombinedFeatures) -> float:
        """
        Higher confidence when:
          - Movement energy is very low
          - EDA slope is clearly positive
        """
        energy_penalty = min(1.0, features.acc.movement_energy / 0.003)
        slope_boost = min(1.0, max(0.0, features.eda.scl_slope) / 1.0)
        return float(0.5 + 0.3 * (1 - energy_penalty) + 0.2 * slope_boost)

    def _agitated_confidence(self, features: CombinedFeatures) -> float:
        """
        Higher confidence when:
          - Fidget score is clearly above threshold
          - SCR count is elevated
        """
        threshold = self.baseline.fidget_threshold if self.baseline else 0.25
        fidget_clarity = min(1.0, (features.acc.fidget_score - threshold) / 0.2)
        scr_signal = min(1.0, features.eda.scr_count / 5.0)
        return float(0.5 + 0.3 * fidget_clarity + 0.2 * scr_signal)
