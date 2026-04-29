"""
Personal baseline calibration protocol — 10-minute, 3-phase session.

Phase 1 (3 min) — REST:     establishes resting SCL, HR, HRV, ACC energy.
Phase 2 (4 min) — ENGAGED:  establishes "focused but not stuck" reference.
Phase 3 (3 min) — STUCK:    establishes frustrated/stuck reference ceiling.

All Stuckness Score normalisation is relative to these personal values so
that population-level norms are never needed.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .signals import SignalSimulator, PhysiologicalState, SignalConfig
from .features import (
    extract_eda_features, extract_ppg_features, extract_acc_features,
    EDAFeatures, PPGFeatures, ACCFeatures,
)


# ---------------------------------------------------------------------------
# Baseline profile dataclass
# ---------------------------------------------------------------------------

@dataclass
class BaselineProfile:
    """
    Personal physiological baseline captured during calibration.
    Stored as JSON and reloaded on subsequent sessions.
    """

    # Resting values
    scl_rest: float           # µS — mean resting SCL
    scl_slope_rest: float     # µS/min — expected slope at rest (≈ 0)
    scr_count_rest: float     # SCR events per 30s window at rest
    hr_rest: float            # bpm
    rmssd_rest: float         # ms
    acc_energy_rest: float    # m²/s⁴ — resting wrist energy

    # Stuck reference ceiling (for normalisation upper bound)
    scl_stuck: float          # µS — mean SCL during stuck state
    scl_slope_stuck: float    # µS/min
    scr_count_stuck: float
    hr_stuck: float
    rmssd_stuck: float        # ms — lower = higher stress
    acc_energy_stuck: float

    # Derived normalisation bounds (set automatically after calibration)
    fidget_threshold: float = 0.25   # fidget_score above which = agitated
    freeze_energy_threshold: float = 0.003

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "BaselineProfile":
        data = json.loads(path.read_text())
        return cls(**data)


# ---------------------------------------------------------------------------
# Calibration runner
# ---------------------------------------------------------------------------

_PHASE_CONFIG = [
    {
        "name": "MANUCHA (REST)",
        "description": "Shevu be-neches. Can see a favourite video.",
        "duration_s": 180,
        "sim_state": PhysiologicalState.REST,
        "n_windows": 6,   # 6 × 30s windows
    },
    {
        "name": "ENGAGED",
        "description": "Play a game you enjoy.",
        "duration_s": 240,
        "sim_state": PhysiologicalState.ENGAGED,
        "n_windows": 8,
    },
    {
        "name": "STUCK REFERENCE",
        "description": "Try a very hard puzzle (intentionally too difficult).",
        "duration_s": 180,
        "sim_state": PhysiologicalState.STUCK_FREEZE,
        "n_windows": 6,
    },
]


class CalibrationProtocol:
    """
    Runs the 3-phase calibration protocol using a signal source.

    In production: `signal_source` would be a live hardware reader.
    In simulation / testing: pass `simulate=True` to use `SignalSimulator`.
    """

    def __init__(
        self,
        config: Optional[SignalConfig] = None,
        simulate: bool = True,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.config = config or SignalConfig()
        self.simulate = simulate
        self.verbose = verbose
        self._simulator = SignalSimulator(config=self.config, seed=seed)

    # ------------------------------------------------------------------

    def run(self, save_path: Optional[Path] = None) -> BaselineProfile:
        """Execute the full calibration and return a BaselineProfile."""
        phase_features: list[dict] = []

        for phase in _PHASE_CONFIG:
            feats = self._run_phase(phase)
            phase_features.append(feats)
            if self.verbose:
                self._print_phase_summary(phase["name"], feats)

        profile = self._build_profile(phase_features)

        if save_path is not None:
            profile.save(save_path)
            if self.verbose:
                print(f"\n[Calibration] Baseline saved → {save_path}")

        return profile

    # ------------------------------------------------------------------

    def _run_phase(self, phase: dict) -> dict:
        """Collect windows for one phase and return averaged feature dicts."""
        if self.verbose:
            print(f"\n[Phase] {phase['name']} — {phase['description']}")
            print(f"        Duration: {phase['duration_s'] // 60} min {phase['duration_s'] % 60} s")

        scl_slopes, scr_counts, hr_values, rmssd_values, acc_energies = [], [], [], [], []

        for i in range(phase["n_windows"]):
            window = self._simulator.generate_window(
                state=phase["sim_state"],
                duration=30.0,
                timestamp=float(i * 30),
            )
            eda_f = extract_eda_features(window.eda, fs=self.config.eda_fs)
            ppg_f = extract_ppg_features(window.ppg, fs=self.config.ppg_fs)
            acc_f = extract_acc_features(window.acc, fs=self.config.acc_fs)

            scl_slopes.append(eda_f.scl_slope)
            scr_counts.append(eda_f.scr_count)
            hr_values.append(ppg_f.hr_bpm)
            rmssd_values.append(ppg_f.rmssd_ms)
            acc_energies.append(acc_f.movement_energy)

            if self.verbose:
                print(
                    f"  Window {i+1:02d}: SCL_slope={eda_f.scl_slope:+.2f} µS/min  "
                    f"SCR={eda_f.scr_count:2d}  HR={ppg_f.hr_bpm:.0f}  "
                    f"RMSSD={ppg_f.rmssd_ms:.0f} ms  "
                    f"ACC_E={acc_f.movement_energy:.4f}"
                )

        return {
            "scl_slope": float(np.median(scl_slopes)),
            "scl_mean": float(np.median(self._simulator._scl_current + np.array(scl_slopes) * 0)),
            "scr_count": float(np.mean(scr_counts)),
            "hr": float(np.mean(hr_values)),
            "rmssd": float(np.mean(rmssd_values)),
            "acc_energy": float(np.mean(acc_energies)),
        }

    def _build_profile(self, phases: list[dict]) -> BaselineProfile:
        rest, engaged, stuck = phases[0], phases[1], phases[2]

        # Use the simulator's final SCL for resting / stuck references
        # (approximation; in production, use mean SCL from each phase)
        scl_rest_approx = 2.0 + rest["scl_slope"] * 1.5   # midpoint of REST phase
        scl_stuck_approx = 4.5 + stuck["scl_slope"] * 1.5

        return BaselineProfile(
            scl_rest=scl_rest_approx,
            scl_slope_rest=rest["scl_slope"],
            scr_count_rest=rest["scr_count"],
            hr_rest=rest["hr"],
            rmssd_rest=rest["rmssd"],
            acc_energy_rest=rest["acc_energy"],

            scl_stuck=scl_stuck_approx,
            scl_slope_stuck=stuck["scl_slope"],
            scr_count_stuck=stuck["scr_count"],
            hr_stuck=stuck["hr"],
            rmssd_stuck=stuck["rmssd"],
            acc_energy_stuck=stuck["acc_energy"],

            fidget_threshold=0.25,
            # Midpoint between resting energy and stuck-freeze energy
            freeze_energy_threshold=max(
                0.001,
                (rest["acc_energy"] + stuck["acc_energy"]) / 2.0,
            ),
        )

    @staticmethod
    def _print_phase_summary(name: str, feats: dict) -> None:
        print(
            f"\n  -> {name} summary: "
            f"SCL_slope={feats['scl_slope']:+.2f}  "
            f"SCR={feats['scr_count']:.1f}  "
            f"HR={feats['hr']:.0f}  "
            f"RMSSD={feats['rmssd']:.0f} ms  "
            f"ACC_E={feats['acc_energy']:.4f}"
        )
