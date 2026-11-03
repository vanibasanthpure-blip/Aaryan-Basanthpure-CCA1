# Data

This folder is intentionally empty in version control. Raw signals are
~100 MB and are downloaded on demand from PhysioNet instead of being
committed.

## Download

```bash
python src/data_loader.py
```

This fetches the 44 MIT-BIH Arrhythmia Database records used by this project
(see `src/utils.py` for the exact list and `docs/methodology.md` for why four
records are excluded) into `data/mitdb/` via the `wfdb` package, directly from
<https://physionet.org/content/mitdb/1.0.0/>.

## Cache

`python src/build_dataset.py` (also run automatically by `train.py`) then
preprocesses every record once and caches the segmented, labelled beats as
`data/beats_cache.npz` so later runs skip re-filtering and re-segmenting.
Delete this file to force a rebuild after changing `preprocessing.py`.
