"""
False-positive protection guards.

Three mechanisms (from the plan):

  1. Physical Activity Veto  — suppress trigger when the child is simply
     moving (e.g. running). Detected via high movement energy.

  2. Cooldown               — enforce a minimum gap between consecutive
     triggers so the child doesn't become desensitised.

  3. Adaptive Threshold     — raise the trigger threshold when the child
     repeatedly dismisses triggers (gesture or button), indicating the
     current threshold is too sensitive for this child.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GuardConfig:
    # Physical activity veto
    # Fidgeting (stuck_agitated state) produces energy ~0.10–0.25.
    # Gross motor activity produces energy > 0.50.
    # Threshold is set above fidgeting but below running.
    activity_energy_threshold: float = 0.45   # m²/s⁴ — above = active movement

    # Cooldown
    cooldown_seconds: float = 180.0           # 3 minutes minimum between triggers

    # Adaptive threshold
    adaptive_step: float = 5.0               # score points per dismiss streak
    adaptive_max_raise: float = 20.0         # never raise threshold by more than 20pts
    dismiss_streak_required: int = 3         # consecutive dismissals before raising


# ---------------------------------------------------------------------------
# Guard state
# ---------------------------------------------------------------------------

@dataclass
class GuardState:
    last_trigger_time: float = 0.0          # Unix timestamp
    dismiss_streak: int = 0                 # consecutive dismiss count
    threshold_offset: float = 0.0          # accumulated offset added to base threshold
    total_triggers: int = 0
    total_vetoed: int = 0


# ---------------------------------------------------------------------------
# Main guard class
# ---------------------------------------------------------------------------

class StucknessGuard:
    """
    Wraps the score + classification result with three false-positive guards.

    Usage:
        guard = StucknessGuard(base_threshold=60.0)
        allowed = guard.should_trigger(score=72.0, movement_energy=0.05)
        if allowed:
            guard.record_trigger()
        # Later, if child dismisses:
        guard.record_dismiss()
    """

    def __init__(
        self,
        base_threshold: float = 60.0,
        config: Optional[GuardConfig] = None,
    ):
        self.base_threshold = base_threshold
        self.config = config or GuardConfig()
        self.state = GuardState()

    # ------------------------------------------------------------------

    @property
    def effective_threshold(self) -> float:
        """Current trigger threshold after adaptive adjustments."""
        return self.base_threshold + self.state.threshold_offset

    def should_trigger(
        self,
        score: float,
        movement_energy: float,
        current_time: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).

        All three guards must pass for the trigger to be allowed.
        """
        now = current_time or time.time()

        # Guard 1: Physical activity veto
        if movement_energy > self.config.activity_energy_threshold:
            self.state.total_vetoed += 1
            return False, "veto:activity"

        # Guard 2: Cooldown
        elapsed = now - self.state.last_trigger_time
        if elapsed < self.config.cooldown_seconds:
            remaining = int(self.config.cooldown_seconds - elapsed)
            return False, f"cooldown:{remaining}s"

        # Guard 3: Adaptive threshold
        if score < self.effective_threshold:
            return False, f"below_threshold:{score:.1f}<{self.effective_threshold:.1f}"

        return True, "ok"

    def record_trigger(self, current_time: Optional[float] = None) -> None:
        """Call after a trigger is delivered to the child."""
        self.state.last_trigger_time = current_time or time.time()
        self.state.total_triggers += 1
        # Successful trigger resets dismiss streak
        self.state.dismiss_streak = 0

    def record_dismiss(self) -> None:
        """
        Call when the child dismisses a trigger (swipe/button gesture).
        After `dismiss_streak_required` consecutive dismissals, threshold is raised.
        """
        self.state.dismiss_streak += 1
        if self.state.dismiss_streak >= self.config.dismiss_streak_required:
            raise_by = min(
                self.config.adaptive_step,
                self.config.adaptive_max_raise - self.state.threshold_offset,
            )
            if raise_by > 0:
                self.state.threshold_offset += raise_by
                self.state.dismiss_streak = 0  # reset after raising

    def record_accept(self) -> None:
        """Call when the child accepts / acts on a trigger (positive feedback)."""
        # Accepted trigger reduces threshold offset slightly (reward good sensitivity)
        self.state.threshold_offset = max(0.0, self.state.threshold_offset - 1.0)
        self.state.dismiss_streak = 0

    def status_summary(self) -> dict:
        return {
            "effective_threshold": self.effective_threshold,
            "base_threshold": self.base_threshold,
            "threshold_offset": self.state.threshold_offset,
            "cooldown_remaining_s": max(
                0.0, self.config.cooldown_seconds - (time.time() - self.state.last_trigger_time)
            ),
            "dismiss_streak": self.state.dismiss_streak,
            "total_triggers": self.state.total_triggers,
            "total_vetoed": self.state.total_vetoed,
        }
