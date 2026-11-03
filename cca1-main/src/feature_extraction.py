"""Hand-crafted feature sets for Approach 1 (RR/HRV + SVM) and Approach 2
(DWT + Random Forest). Both operate on the same segmented beats produced by
preprocessing.segment_beats.
"""
import numpy as np
import pywt


# ---------------------------------------------------------------------------
# Approach 1: RR interval / heart-rate-variability features
# ---------------------------------------------------------------------------
def rr_hrv_features(rr_prev: np.ndarray, rr_next: np.ndarray) -> np.ndarray:
    """Per-beat RR/HRV feature vector.

    Missing neighbour RR (first/last beat of a record) is imputed with the
    other neighbour so a beat is never dropped just for being at a record
    boundary.
    """
    prev = np.where(np.isnan(rr_prev), rr_next, rr_prev)
    nxt = np.where(np.isnan(rr_next), rr_prev, rr_next)
    prev = np.nan_to_num(prev, nan=np.nanmedian(prev))
    nxt = np.nan_to_num(nxt, nan=np.nanmedian(nxt))

    mean_rr = (prev + nxt) / 2.0
    rr_ratio = prev / np.where(nxt == 0, 1e-6, nxt)
    rmssd_local = np.abs(nxt - prev)  # local successive-difference magnitude
    heart_rate = 60.0 / np.where(mean_rr == 0, 1e-6, mean_rr)

    feats = np.stack([prev, nxt, mean_rr, rr_ratio, rmssd_local, heart_rate], axis=1)
    return feats


RR_FEATURE_NAMES = ["rr_prev", "rr_next", "mean_rr", "rr_ratio", "rmssd_local", "heart_rate"]


def record_level_hrv(rr_prev_all: np.ndarray) -> dict:
    """Whole-record HRV summary (mean RR, SDNN, RMSSD) -- reported for EDA,
    not fed per-beat (those statistics need a window of beats, not one)."""
    rr = rr_prev_all[~np.isnan(rr_prev_all)]
    if len(rr) < 2:
        return {"mean_rr": np.nan, "sdnn": np.nan, "rmssd": np.nan}
    diffs = np.diff(rr)
    return {
        "mean_rr": float(np.mean(rr)),
        "sdnn": float(np.std(rr)),
        "rmssd": float(np.sqrt(np.mean(diffs ** 2))),
    }


# ---------------------------------------------------------------------------
# Approach 2: Discrete Wavelet Transform features
# ---------------------------------------------------------------------------
DWT_WAVELET = "db4"
DWT_LEVEL = 4


def _band_stats(coeffs: np.ndarray) -> list:
    energy = float(np.sum(coeffs ** 2))
    prob = coeffs ** 2 / (energy + 1e-12)
    prob = prob[prob > 0]
    entropy = float(-np.sum(prob * np.log(prob))) if len(prob) else 0.0
    return [
        float(np.mean(coeffs)), float(np.std(coeffs)), float(np.min(coeffs)),
        float(np.max(coeffs)), energy, entropy,
    ]


def dwt_features(beat: np.ndarray, wavelet: str = DWT_WAVELET, level: int = DWT_LEVEL) -> np.ndarray:
    coeffs = pywt.wavedec(beat, wavelet, level=level)
    feats = []
    for band in coeffs:
        feats.extend(_band_stats(band))
    return np.array(feats, dtype=np.float32)


def dwt_feature_matrix(beats: np.ndarray) -> np.ndarray:
    return np.array([dwt_features(b) for b in beats], dtype=np.float32)


DWT_FEATURE_NAMES = [
    f"{band}_{stat}"
    for band in (["cA4"] + [f"cD{l}" for l in range(DWT_LEVEL, 0, -1)])
    for stat in ["mean", "std", "min", "max", "energy", "entropy"]
]
