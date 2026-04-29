"""
Physiological signal simulation for EDA, PPG, and Accelerometer.

Used during development and validation when actual wristband hardware is
unavailable. Generates realistic signals for five physiological states.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PhysiologicalState(Enum):
    REST = "rest"
    ENGAGED = "engaged"
    STUCK_FREEZE = "stuck_freeze"
    STUCK_AGITATED = "stuck_agitated"
    ACTIVE_MOVEMENT = "active_movement"


@dataclass
class SignalConfig:
    eda_fs: int = 4      # Hz — 4 Hz is sufficient for EDA (0–0.5 Hz bandwidth)
    ppg_fs: int = 64     # Hz — 64 Hz needed for reliable R-peak detection
    acc_fs: int = 50     # Hz — 50 Hz covers fidget band (1–3 Hz) with margin


@dataclass
class SignalWindow:
    eda: np.ndarray       # shape: (eda_fs * duration,)
    ppg: np.ndarray       # shape: (ppg_fs * duration,)
    acc: np.ndarray       # shape: (acc_fs * duration, 3)
    state: Optional[PhysiologicalState] = None
    timestamp: float = 0.0


# Physiological parameters per state
_STATE_PARAMS: dict = {
    PhysiologicalState.REST: {
        "scl_base": 2.0, "scl_noise": 0.10, "scr_rate": 0.5,
        "hr": 68,  "rmssd": 55,
        "acc_energy": 0.008, "fidget": False,
    },
    PhysiologicalState.ENGAGED: {
        "scl_base": 3.0, "scl_noise": 0.15, "scr_rate": 2.0,
        "hr": 75,  "rmssd": 40,
        "acc_energy": 0.04, "fidget": False,
    },
    PhysiologicalState.STUCK_FREEZE: {
        "scl_base": 4.5, "scl_noise": 0.20, "scr_rate": 4.5,
        "hr": 82,  "rmssd": 22,
        "acc_energy": 0.004, "fidget": False,   # near-zero movement
    },
    PhysiologicalState.STUCK_AGITATED: {
        "scl_base": 4.8, "scl_noise": 0.25, "scr_rate": 5.0,
        "hr": 85,  "rmssd": 20,
        "acc_energy": 0.08, "fidget": True,     # wrist fidgeting 1–3 Hz
    },
    PhysiologicalState.ACTIVE_MOVEMENT: {
        "scl_base": 3.5, "scl_noise": 0.30, "scr_rate": 1.5,
        "hr": 115, "rmssd": 28,
        "acc_energy": 1.8, "fidget": False,     # gross motor activity
    },
}


class SignalSimulator:
    """
    Generates synthetic physiological signals for a given state.

    Maintains SCL drift state across successive calls so that transitions
    between states are physiologically gradual rather than instantaneous.
    """

    def __init__(self, config: Optional[SignalConfig] = None, seed: int = 42):
        self.config = config or SignalConfig()
        self.rng = np.random.RandomState(seed)
        self._scl_current: float = 2.0

    def generate_window(
        self,
        state: PhysiologicalState,
        duration: float = 30.0,
        timestamp: float = 0.0,
    ) -> SignalWindow:
        params = _STATE_PARAMS[state]
        return SignalWindow(
            eda=self._generate_eda(params, duration),
            ppg=self._generate_ppg(params, duration),
            acc=self._generate_acc(params, duration),
            state=state,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_eda(self, params: dict, duration: float) -> np.ndarray:
        fs = self.config.eda_fs
        n = int(duration * fs)
        t = np.linspace(0, duration, n)

        # Tonic SCL drifts toward target (physiological time constant ~30s)
        target = params["scl_base"]
        self._scl_current += (target - self._scl_current) * 0.35
        tonic = self._scl_current + self.rng.normal(0, params["scl_noise"], n)
        # Slow undulation (0.02 Hz)
        tonic += 0.08 * np.sin(2 * np.pi * 0.02 * t)

        # Phasic SCRs via Poisson process
        scr_per_sec = params["scr_rate"] / 60.0
        n_scrs = self.rng.poisson(scr_per_sec * duration)
        onsets = self.rng.uniform(1.0, max(1.0, duration - 6.0), max(0, n_scrs))

        for onset in onsets:
            amp = self.rng.uniform(0.15, 0.55)
            oi = int(onset * fs)
            rise = int(1.5 * fs)
            decay = int(5.0 * fs)
            # Fast rise
            for i in range(rise):
                idx = oi + i
                if idx < n:
                    tonic[idx] += amp * (i / rise)
            # Exponential decay from peak
            peak_i = oi + rise
            for i in range(decay):
                idx = peak_i + i
                if idx < n:
                    tonic[idx] += amp * np.exp(-i / (2.0 * fs))

        return np.clip(tonic, 0.1, 25.0)

    def _generate_ppg(self, params: dict, duration: float) -> np.ndarray:
        fs = self.config.ppg_fs
        n = int(duration * fs)

        mean_rr_ms = 60000.0 / params["hr"]
        # RMSSD ≈ sqrt(mean(diff(RR)²)) → noise std ≈ RMSSD / sqrt(2)
        rr_noise_std = params["rmssd"] / np.sqrt(2.0)

        beat_times_ms: list = []
        t_ms = 0.0
        while t_ms < duration * 1000.0:
            rr = mean_rr_ms + self.rng.normal(0.0, rr_noise_std)
            rr = float(np.clip(rr, 350.0, 1500.0))
            beat_times_ms.append(t_ms)
            t_ms += rr

        ppg = self.rng.normal(0.0, 0.015, n)

        pulse_samples = int(0.38 * fs)
        for bt_ms in beat_times_ms:
            bi = int(bt_ms / 1000.0 * fs)
            if bi >= n:
                break
            for i in range(pulse_samples):
                idx = bi + i
                if idx >= n:
                    break
                p = i / pulse_samples
                if p < 0.20:
                    ppg[idx] += p / 0.20
                elif p < 0.35:
                    ppg[idx] += 1.0 - (p - 0.20) / 0.15 * 0.30
                elif p < 0.45:
                    ppg[idx] += 0.70 + (p - 0.35) / 0.10 * 0.10
                else:
                    ppg[idx] += 0.80 * np.exp(-(p - 0.45) / 0.15)

        return ppg

    def _generate_acc(self, params: dict, duration: float) -> np.ndarray:
        fs = self.config.acc_fs
        n = int(duration * fs)
        t = np.linspace(0, duration, n)

        acc = np.zeros((n, 3))
        # Static gravity on z-axis with sensor noise
        acc[:, 2] = 9.81 + self.rng.normal(0, 0.03, n)

        noise_std = float(np.sqrt(params["acc_energy"]))
        acc += self.rng.normal(0, noise_std, (n, 3))

        if params.get("fidget"):
            # Wrist fidgeting: 1.5–2.5 Hz oscillation on x and y axes
            freq = self.rng.uniform(1.5, 2.5)
            amp = 0.30
            acc[:, 0] += amp * np.sin(2 * np.pi * freq * t)
            acc[:, 1] += amp * 0.5 * np.sin(2 * np.pi * freq * t + np.pi / 4)

        if params["acc_energy"] > 1.0:
            # Gross motor activity: add large irregular bursts
            acc += self.rng.normal(0, 1.0, (n, 3))

        return acc
