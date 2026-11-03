"""Run the full data pipeline (download -> clean -> segment) over every
record in the DS1/DS2 split and cache the result as a single .npz so the
three training scripts don't each re-run signal processing from scratch.
"""
import os

import numpy as np

from data_loader import load_record
from preprocessing import clean_signal, segment_beats
from utils import ALL_RECORDS, RAW_DIR, RESULTS_DIR

CACHE_PATH = os.path.join(os.path.dirname(RAW_DIR), "beats_cache.npz")
MLII_CHANNEL_FALLBACK = 0  # used only if 'MLII' isn't the first channel


def build(record_ids=ALL_RECORDS, local_dir=RAW_DIR, cache_path=CACHE_PATH, force=False):
    if os.path.exists(cache_path) and not force:
        print(f"Using cached dataset at {cache_path}")
        return dict(np.load(cache_path, allow_pickle=True))

    all_beats, all_labels, all_groups = [], [], []
    all_rr_prev, all_rr_next = [], []
    per_record_summary = []

    for rid in record_ids:
        sig, fields, ann = load_record(rid, local_dir)
        sig_names = fields["sig_name"]
        channel = sig_names.index("MLII") if "MLII" in sig_names else MLII_CHANNEL_FALLBACK
        raw = sig[:, channel]

        cleaned = clean_signal(raw, fs=fields["fs"])
        beats, labels, rr_prev, rr_next = segment_beats(cleaned, ann.sample, ann.symbol, fs=fields["fs"])

        n_kept = len(beats)
        n_total_ann = len(ann.sample)
        per_record_summary.append({"record": rid, "n_beats_kept": n_kept, "n_annotations": n_total_ann})
        print(f"Record {rid}: kept {n_kept}/{n_total_ann} annotations as beats")

        all_beats.append(beats)
        all_labels.append(labels)
        all_groups.append(np.full(n_kept, rid, dtype=np.int32))
        all_rr_prev.append(rr_prev)
        all_rr_next.append(rr_next)

    data = {
        "beats": np.concatenate(all_beats, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "groups": np.concatenate(all_groups, axis=0),
        "rr_prev": np.concatenate(all_rr_prev, axis=0),
        "rr_next": np.concatenate(all_rr_next, axis=0),
    }
    np.savez_compressed(cache_path, **data)
    print(f"Cached {len(data['beats'])} beats from {len(record_ids)} records -> {cache_path}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    import csv
    with open(os.path.join(RESULTS_DIR, "per_record_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["record", "n_beats_kept", "n_annotations"])
        writer.writeheader()
        writer.writerows(per_record_summary)

    return data


if __name__ == "__main__":
    d = build()
    print({k: v.shape for k, v in d.items()})
