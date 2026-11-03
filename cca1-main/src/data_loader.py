"""Download and load MIT-BIH Arrhythmia Database records via WFDB/PhysioNet.

Records are cached under data/mitdb/ after the first download so repeated
runs do not re-hit PhysioNet.
"""
import os

import wfdb

from utils import ALL_RECORDS, RAW_DIR

PHYSIONET_DB = "mitdb"


def ensure_record(record_id: int, local_dir: str = RAW_DIR) -> str:
    """Download one record's .dat/.hea/.atr files if not already cached."""
    os.makedirs(local_dir, exist_ok=True)
    record_name = str(record_id)
    header_path = os.path.join(local_dir, record_name + ".hea")
    if not os.path.exists(header_path):
        wfdb.dl_database(PHYSIONET_DB, local_dir, records=[record_name],
                          annotators=["atr"])
    return os.path.join(local_dir, record_name)


def load_record(record_id: int, local_dir: str = RAW_DIR):
    """Return (signal [n_samples, n_channels], fields, annotation) for a record."""
    path = ensure_record(record_id, local_dir)
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, "atr")
    return record.p_signal, record.__dict__, annotation


def download_all(record_ids=ALL_RECORDS, local_dir: str = RAW_DIR) -> None:
    for rid in record_ids:
        print(f"Fetching record {rid} ...")
        ensure_record(rid, local_dir)
    print(f"All {len(record_ids)} records cached in {local_dir}")


if __name__ == "__main__":
    download_all()
