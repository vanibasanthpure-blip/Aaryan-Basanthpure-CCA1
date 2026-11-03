"""ECG signal preprocessing and beat segmentation.

Pipeline (applied identically for every record, before any model sees data):
  1. Baseline wander removal via a two-stage median filter (200ms / 600ms
     windows) -- removes slow drift without distorting the QRS complex the
     way a naive high-pass filter can.
  2. Bandpass Butterworth filter (0.5-40 Hz) -- removes residual low-frequency
     drift and high-frequency muscle/powerline noise. 40 Hz keeps QRS and
     T-wave morphology intact while attenuating 60 Hz mains interference
     (MIT-BIH was recorded in the US), so a separate notch filter is not
     needed.
  3. Beat segmentation: a fixed window around each annotated R-peak
     (0.28 s before, 0.40 s after -> 245 samples at 360 Hz) covering the
     QRS complex and T-wave for every beat.
  4. Per-beat z-score normalisation (zero mean, unit variance) so amplitude
     differences across patients/leads don't dominate learned features.
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

from utils import BEAT_LEN, BEAT_WINDOW_SEC, FS, SYMBOL_TO_AAMI


def remove_baseline_wander(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    win1 = int(0.2 * fs) | 1  # odd window, ~200ms
    win2 = int(0.6 * fs) | 1  # ~600ms
    baseline = median_filter(signal, size=win2)
    baseline = median_filter(baseline, size=win1)
    return signal - baseline


def bandpass_filter(signal: np.ndarray, fs: int = FS, low=0.5, high=40.0, order=3) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def clean_signal(raw_signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """raw_signal: 1-D array (single lead, e.g. MLII)."""
    x = remove_baseline_wander(raw_signal, fs)
    x = bandpass_filter(x, fs)
    return x


def segment_beats(clean_sig: np.ndarray, ann_samples, ann_symbols, fs: int = FS):
    """Extract fixed-length, normalised beat windows for valid AAMI-mapped beats.

    Returns (beats [n, BEAT_LEN], labels [n] as AAMI class strings, rr_info).
    rr_info holds (prev_rr, next_rr) in seconds per kept beat for HRV features.
    """
    before = int(BEAT_WINDOW_SEC[0] * fs)
    after = int(BEAT_WINDOW_SEC[1] * fs)
    n = len(clean_sig)

    # Keep only genuine beat annotations (drop rhythm/signal-quality markers
    # such as '+', '~', '|') *before* computing RR intervals, so adjacent-beat
    # timing isn't polluted by non-beat markers.
    beat_mask = np.array([s in SYMBOL_TO_AAMI for s in ann_symbols])
    beat_samples = np.asarray(ann_samples)[beat_mask]
    beat_symbols = np.asarray(ann_symbols)[beat_mask]

    beats, labels, rr_prev, rr_next = [], [], [], []
    for i, (sample, symbol) in enumerate(zip(beat_samples, beat_symbols)):
        start, end = sample - before, sample + after
        if start < 0 or end > n:
            continue  # drop incomplete edge beats rather than zero-pad artefacts

        window = clean_sig[start:end]
        if len(window) != BEAT_LEN or not np.isfinite(window).all():
            continue
        std = window.std()
        if std < 1e-8:
            continue  # flatline / invalid segment
        window = (window - window.mean()) / std

        prev_rr = (sample - beat_samples[i - 1]) / fs if i > 0 else np.nan
        next_rr = (beat_samples[i + 1] - sample) / fs if i < len(beat_samples) - 1 else np.nan

        beats.append(window)
        labels.append(SYMBOL_TO_AAMI[symbol])
        rr_prev.append(prev_rr)
        rr_next.append(next_rr)

    return (np.array(beats, dtype=np.float32), np.array(labels),
            np.array(rr_prev), np.array(rr_next))
