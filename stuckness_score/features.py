"""
Feature extraction from raw EDA, PPG, and Accelerometer windows.

All functions operate on a single time window (default 30 seconds) and
return typed dataclasses. No baseline normalization is applied here —
that is handled by the scorer using the personal BaselineProfile.
"""

import numpy as np
from scipy import signal as sp_signal
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EDAFeatures:
    scl_slope: float          # µS/min — positive = rising stress
    scr_count: int            # phasic events in window
    scr_amplitude_mean: float # µS — mean SCR peak amplitude


@dataclass
class PPGFeatures:
    hr_bpm: float             # beats per minute
    rmssd_ms: float           # HRV metric in milliseconds


@dataclass
class ACCFeatures:
    movement_energy: float    # mean squared magnitude of filtered ACC
    fidget_score: float       # 0–1, spectral power fraction in 1–3 Hz band
    freeze_flag: bool         # True when energy is below freeze threshold


@dataclass
class CombinedFeatures:
    eda: EDAFeatures
    ppg: PPGFeatures
    acc: ACCFeatures


# ---------------------------------------------------------------------------
# EDA feature extraction
# ---------------------------------------------------------------------------

_MIN_SCR_HEIGHT = 0.20       # µS — minimum phasic peak to count as SCR
_MIN_SCR_DISTANCE_S = 2.0    # seconds — refractory period between SCRs


def extract_eda_features(eda: np.ndarray, fs: int = 4) -> EDAFeatures:
    """
    Decompose EDA signal into tonic (SCL) and phasic (SCR) components.

    Tonic trend is estimated via a 0.05 Hz low-pass Butterworth filter.
    Phasic component is eda - tonic; peaks above 0.2 µS are counted as SCRs.
    """
    n = len(eda)
    t_min = np.arange(n) / fs / 60.0  # time axis in minutes

    # SCL slope via least-squares linear fit
    coeffs = np.polyfit(t_min, eda, 1)
    scl_slope = float(coeffs[0])

    # Separate tonic baseline with low-pass filter (requires > 8 samples)
    if n > int(fs * 4):
        b, a = sp_signal.butter(2, 0.05 / (fs / 2.0), btype="low")
        try:
            tonic = sp_signal.filtfilt(b, a, eda)
        except Exception:
            tonic = np.full(n, float(np.mean(eda)))
    else:
        tonic = np.full(n, float(np.mean(eda)))

    phasic = eda - tonic

    min_dist = max(1, int(_MIN_SCR_DISTANCE_S * fs))
    peaks, props = sp_signal.find_peaks(
        phasic,
        height=_MIN_SCR_HEIGHT,
        distance=min_dist,
    )

    scr_count = len(peaks)
    scr_amplitude_mean = float(np.mean(phasic[peaks])) if scr_count > 0 else 0.0

    return EDAFeatures(
        scl_slope=scl_slope,
        scr_count=scr_count,
        scr_amplitude_mean=scr_amplitude_mean,
    )


# ---------------------------------------------------------------------------
# PPG feature extraction
# ---------------------------------------------------------------------------

_MIN_RR_MS = 350.0    # 170 bpm ceiling
_MAX_RR_MS = 1500.0   # 40 bpm floor


def extract_ppg_features(ppg: np.ndarray, fs: int = 64) -> PPGFeatures:
    """
    Detect heartbeats in PPG and compute HR and RMSSD.

    Falls back to physiologically plausible defaults when fewer than
    3 reliable beats are detected (e.g. very short or noisy windows).
    """
    _FALLBACK = PPGFeatures(hr_bpm=72.0, rmssd_ms=40.0)

    if len(ppg) < fs * 3:
        return _FALLBACK

    # Normalise to [0, 1] to make peak detection threshold signal-independent
    ptp = float(np.max(ppg) - np.min(ppg))
    if ptp < 1e-6:
        return _FALLBACK
    ppg_norm = (ppg - np.min(ppg)) / ptp

    min_dist = max(1, int(0.40 * fs))  # refractory: 400 ms → max 150 bpm
    peaks, _ = sp_signal.find_peaks(ppg_norm, height=0.50, distance=min_dist)

    if len(peaks) < 3:
        return _FALLBACK

    rr_ms = np.diff(peaks).astype(float) / fs * 1000.0
    rr_ms = rr_ms[(rr_ms > _MIN_RR_MS) & (rr_ms < _MAX_RR_MS)]

    if len(rr_ms) < 2:
        return _FALLBACK

    hr_bpm = float(60000.0 / np.mean(rr_ms))
    successive_diffs = np.diff(rr_ms)
    rmssd_ms = float(np.sqrt(np.mean(successive_diffs ** 2)))

    return PPGFeatures(hr_bpm=hr_bpm, rmssd_ms=rmssd_ms)


# ---------------------------------------------------------------------------
# Accelerometer feature extraction
# ---------------------------------------------------------------------------

_FIDGET_LOW_HZ = 1.0
_FIDGET_HIGH_HZ = 3.0
_DC_CUTOFF_HZ = 0.5         # high-pass to remove gravity
# Default freeze threshold — overridden at runtime by personal baseline.
# Real wrist energy at rest is ~0.020–0.030 m²/s⁴; stuck_freeze ~0.010–0.015.
_FREEZE_ENERGY_THRESHOLD = 0.018


def extract_acc_features(
    acc: np.ndarray,
    fs: int = 50,
    freeze_threshold: Optional[float] = None,
) -> ACCFeatures:
    """
    Compute movement energy, fidget spectral score, and freeze flag.

    Gravity is removed with a 0.5 Hz high-pass filter applied per axis.
    Fidget score is the fraction of spectral power in the 1–3 Hz band.
    """
    n = acc.shape[0] if acc.ndim == 2 else len(acc)

    # High-pass filter to remove gravity (needs at least ~padlen samples)
    min_filter_len = 20
    if n > min_filter_len and acc.ndim == 2:
        nyq = fs / 2.0
        b, a = sp_signal.butter(2, _DC_CUTOFF_HZ / nyq, btype="high")
        try:
            acc_filt = sp_signal.filtfilt(b, a, acc, axis=0)
        except Exception:
            acc_filt = acc.copy()
        magnitude = np.sqrt(np.sum(acc_filt ** 2, axis=1))
    elif acc.ndim == 2:
        magnitude = np.sqrt(np.sum(acc ** 2, axis=1))
    else:
        magnitude = np.abs(acc)

    movement_energy = float(np.mean(magnitude ** 2))

    # Fidget score: spectral power fraction
    if n >= fs:
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        power = np.abs(np.fft.rfft(magnitude)) ** 2
        above_dc = freqs > 0.1
        total_power = float(np.sum(power[above_dc])) + 1e-10
        fidget_band = (freqs >= _FIDGET_LOW_HZ) & (freqs <= _FIDGET_HIGH_HZ)
        fidget_score = float(np.sum(power[fidget_band]) / total_power)
    else:
        fidget_score = 0.0

    threshold = freeze_threshold if freeze_threshold is not None else _FREEZE_ENERGY_THRESHOLD
    freeze_flag = movement_energy < threshold

    return ACCFeatures(
        movement_energy=movement_energy,
        fidget_score=fidget_score,
        freeze_flag=freeze_flag,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def extract_all_features(
    eda: np.ndarray,
    ppg: np.ndarray,
    acc: np.ndarray,
    eda_fs: int = 4,
    ppg_fs: int = 64,
    acc_fs: int = 50,
    freeze_threshold: Optional[float] = None,
) -> CombinedFeatures:
    return CombinedFeatures(
        eda=extract_eda_features(eda, fs=eda_fs),
        ppg=extract_ppg_features(ppg, fs=ppg_fs),
        acc=extract_acc_features(acc, fs=acc_fs, freeze_threshold=freeze_threshold),
    )
