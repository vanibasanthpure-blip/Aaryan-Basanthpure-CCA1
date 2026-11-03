"""Shared constants and small helpers used across the pipeline."""
import os
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Inter-patient split (de Chazal et al., 2004 / AAMI-recommended DS1-DS2 split)
# ---------------------------------------------------------------------------
# Four records are excluded because they contain paced beats that dominate
# the recording and are not representative of the five AAMI classes studied
# here (102, 104, 107, 217). This exclusion is standard practice in the
# inter-patient ECG classification literature and is documented in the README.
DS1_RECORDS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
               122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_RECORDS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
               210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
EXCLUDED_RECORDS = [102, 104, 107, 217]
ALL_RECORDS = sorted(DS1_RECORDS + DS2_RECORDS)

# ---------------------------------------------------------------------------
# AAMI EC57 5-class mapping from MIT-BIH beat annotation symbols
# ---------------------------------------------------------------------------
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

SYMBOL_TO_AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

# Beat-level annotation symbols; everything else (rhythm/signal-quality
# markers such as '+', '~', '|', etc.) is not a beat and is ignored.
VALID_SYMBOLS = set(SYMBOL_TO_AAMI.keys())

FS = 360  # MIT-BIH Arrhythmia Database sampling frequency (Hz)
BEAT_WINDOW_SEC = (0.28, 0.40)  # seconds before / after R-peak kept per beat
BEAT_LEN = int((BEAT_WINDOW_SEC[0] + BEAT_WINDOW_SEC[1]) * FS)  # fixed-length samples

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mitdb")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
