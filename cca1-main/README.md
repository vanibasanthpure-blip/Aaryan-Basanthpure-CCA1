# ECG Arrhythmia Classification Using Advanced Temporal Pattern Recognition

A university Pattern Recognition course project comparing three independent
approaches to heartbeat arrhythmia classification on the MIT-BIH Arrhythmia
Database, under an identical patient-independent evaluation protocol.

> **Scope note:** this is an offline academic project trained and evaluated
> on recorded, retrospective ECG data. It is not a clinically validated
> diagnostic or real-time monitoring system.

## 1. Overview

Three independent pipelines are trained on the **same patient-independent
train/test split** of the MIT-BIH Arrhythmia Database and compared on the
AAMI EC57 five-class scheme (N, S, V, F, Q):

| # | Approach | Features | Classifier |
|---|---|---|---|
| 1 | RR interval / HRV | 6 hand-crafted timing features | SVM (RBF kernel) |
| 2 | Discrete Wavelet Transform | 30 `db4` coefficient-band statistics | Random Forest |
| 3 | Raw beat waveform | learned end-to-end | 1D-CNN + Bidirectional LSTM |

## 2. Problem statement

Automated arrhythmia detection from ECG is a classic pattern recognition
problem: distinguish normal heartbeats from several clinically distinct
abnormal rhythm types using a short waveform segment around each heartbeat.
The central methodological difficulty -- and the reason this project treats
it carefully -- is that a model can trivially achieve high accuracy by
memorising patient-specific morphology if beats from the same patient are
allowed into both the training and test set. A clinically meaningful
evaluation must instead test on **patients the model has never seen**.

## 3. Objectives

- Implement and fairly compare a classical feature-engineering pipeline
  (RR/HRV + SVM), a signal-decomposition pipeline (DWT + Random Forest), and
  a temporal deep-learning pipeline (1D-CNN + BiLSTM) on identical data.
- Use a genuine patient-independent (inter-patient) split, not a random
  beat-level split, to produce a realistic estimate of generalisation.
- Report honest, imbalance-aware metrics (macro-F1 and per-class
  recall/precision, not just accuracy) with no fabricated numbers.

## 4. Dataset

[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
(PhysioNet) -- 48 half-hour two-lead ambulatory ECG recordings at **360 Hz**,
11-bit resolution, with cardiologist beat annotations. This project uses
**44 of the 48 records** (4 paced-beat-dominated records -- 102, 104, 107,
217 -- are excluded; see `docs/methodology.md` for the justification) and
the standard **de Chazal DS1/DS2 inter-patient split**:

- **DS1 (train, 22 records):** 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
- **DS2 (test, 22 records):** 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

No patient's beats ever appear in both DS1 and DS2, and DS1 is further split
(by record, never by beat) for cross-validation/early-stopping -- DS2 is
touched exactly once, for final evaluation. Full annotation-to-AAMI mapping
and realised class counts are documented in `docs/methodology.md`.

### Realised class distribution (from the actual pipeline run)

| Class | DS1 (train) | DS2 (test) | Meaning |
|---|---|---|---|
| N | 45,847 | 44,239 | Normal / bundle-branch-block beat |
| S | 944 | 1,837 | Supraventricular ectopic beat |
| V | 3,788 | 3,220 | Ventricular ectopic beat |
| F | 414 | 388 | Fusion beat |
| Q | 8 | 7 | Unknown / paced beat |

`Q` has single-digit support after excluding the paced records -- this is a
known, documented limitation of the protocol, not a data-handling bug (see
`docs/methodology.md`, section 1).

## 5. Methodology

See **[`docs/methodology.md`](docs/methodology.md)** for the full write-up:
preprocessing pipeline and rationale, exact AAMI annotation mapping,
feature definitions, and the one documented compute trade-off (the SVM is
trained on a class-balanced subsample of DS1 since its RBF kernel does not
scale to the full ~51k beats; Random Forest and CNN-BiLSTM use the full
DS1 set).

### Workflow

```
PhysioNet (wfdb) -> raw ECG + annotations
      |
      v
baseline removal -> 0.5-40Hz bandpass -> per-beat segmentation (R-peak +/- window) -> z-score normalise
      |
      +--- DS1 (train, 22 records) ----------- DS2 (test, 22 records) ---+
      |                                                                   |
      v                                                                  v
  [Approach 1] RR/HRV features  --------> SVM (RBF)         -----> predict -----> metrics
  [Approach 2] DWT (db4) features ------> Random Forest     -----> predict -----> metrics
  [Approach 3] raw beat waveform -------> 1D-CNN + BiLSTM   -----> predict -----> metrics
```

### Why these three approaches are comparable

All three see the *same* segmented, cleaned beats and the *same* DS1/DS2
split; they differ only in how much signal structure is hand-engineered
away before the classifier sees it (full timing-only summary -> frequency-
band statistics -> nothing, raw waveform). That makes the comparison a
genuine test of feature engineering vs. representation learning, not an
artifact of different data handling.

## 6. Technology stack

Python 3.12, [`wfdb`](https://pypi.org/project/wfdb/) (PhysioNet access),
NumPy, SciPy (filtering), scikit-learn (SVM, Random Forest, metrics),
PyWavelets (DWT), PyTorch (CNN-BiLSTM), pandas/Matplotlib/Seaborn
(analysis & plots), Streamlit (demo UI).

## 7. Installation

```bash
git clone <this-repository-url>
cd ecg-arrhythmia-pattern-recognition
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Python 3.12 is recommended; at the time of writing several dependencies
here do not yet ship wheels for the very latest Python releases.)

## 8. Dataset download

```bash
python src/data_loader.py
```

Downloads the 44 required records from PhysioNet into `data/mitdb/`
(~100 MB, not committed to git -- see `data/README.md`).

## 9. Train the models

```bash
cd src
python train.py --approach all     # all three, or:
python train.py --approach svm     # RR/HRV + SVM only
python train.py --approach rf      # DWT + Random Forest only
python train.py --approach cnn     # 1D-CNN + BiLSTM only
```

The first run also builds and caches the segmented-beat dataset
(`data/beats_cache.npz`, ~2-3 minutes); later runs reuse the cache. Each
`--approach` call re-evaluates on DS2 and overwrites `results/metrics.csv`
with just that run's model(s) -- after training pieces separately, run
`python src/generate_comparison.py` once to re-evaluate every model
currently saved under `models/` together and regenerate the combined
comparison table/plots without re-training anything.

## 10. Evaluate / regenerate the comparison report

```bash
cd src
python generate_comparison.py
```

Writes `results/metrics.csv`, per-model classification reports and
confusion matrices (`results/classification_reports/`), and comparison
figures (`results/figures/`).

## 11. Run the prediction interface

Command-line, on one MIT-BIH record:

```bash
cd src
python predict.py --record 100 --n-beats 10
```

Streamlit demo (loads whichever trained models are present under `models/`):

```bash
streamlit run app/streamlit_app.py
```

## 12. Actual experimental results (DS2 test set, 49,691 beats)

All numbers below come from the trained models in this repository,
evaluated once on the untouched DS2 test set -- see
`results/metrics.csv` and `results/classification_reports/`.

| Model | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC (macro, OvR) |
|---|---|---|---|---|
| SVM (RR/HRV) | 0.789 | **0.362** | 0.839 | **0.812** |
| Random Forest (DWT) | **0.884** | 0.334 | **0.882** | 0.730 |
| CNN-BiLSTM | 0.449 | 0.183 | 0.570 | 0.563 |

Because classes are severely imbalanced (N is ~89% of DS2), **accuracy
alone is misleading** -- macro-F1 and per-class recall matter more here.

### Per-class F1-score

| Class | SVM (RR/HRV) | Random Forest (DWT) | CNN-BiLSTM |
|---|---|---|---|
| N | 0.884 | **0.937** | 0.621 |
| S | **0.065** | 0.011 | 0.057 |
| V | **0.757** | 0.724 | 0.236 |
| F | **0.101** | 0.000 | 0.001 |
| Q (n=7) | 0.002 | 0.000 | 0.000 |

See `results/figures/model_comparison.png`,
`results/figures/per_class_f1_comparison.png`, and each model's confusion
matrix (`results/figures/confusion_matrix_*.png`) for the full picture, and
`results/figures/cnn_bilstm_training_curves.png` for the CNN-BiLSTM
train/validation loss and validation macro-F1 curves.

### Reading these results honestly

- **Random Forest (DWT)** gets the best raw accuracy and N/weighted-F1 by
  being very good at the majority class, but its recall on S and F
  collapses to near zero on unseen patients -- morphology-only wavelet
  statistics apparently don't generalise well across patients for these
  rarer classes, even with `class_weight="balanced"`.
- **SVM (RR/HRV)** has the best macro-F1 and ROC-AUC and by far the best F
  recall (0.94, not shown as F1 above because precision is low) -- RR-interval
  timing is a strong, patient-invariant signal for premature/fusion beats,
  even though it says almost nothing about S beats (recall 0.046).
- **CNN-BiLSTM** underperforms both classical approaches in this project's
  configuration. Two likely, documented reasons: (1) the network is
  intentionally compact and trained for only a handful of epochs, within
  this project's compute budget; (2) with no explicit timing input, raw
  single-lead waveform morphology appears to generalise across unseen
  patients worse than the hand-crafted features do, for this amount of
  training data. An uncapped inverse-frequency class-weighting scheme was
  also found, during development, to destabilise training given the
  single-digit `Q` class support -- fixed here by capping class weights at
  10x and selecting the checkpoint by validation macro-F1 rather than
  validation loss (see `src/model_cnn_bilstm.py`); this is disclosed rather
  than hidden because it materially affects the result.
- These are **real, imperfect results**, not polished-for-effect numbers.
  Nothing here was tuned against the DS2 test set -- hyperparameter search
  (`GridSearchCV`, early stopping) only ever sees DS1.

## 13. Limitations

- `Q` has only 7-8 beats once paced-dominated records are excluded; its
  metrics are not statistically meaningful and should be read as
  illustrative, not evidence of real paced-beat detection capability.
- The SVM is trained on a class-balanced subsample of DS1 (compute
  trade-off, documented in `docs/methodology.md`), not the full training
  set, unlike the other two models.
- CNN-BiLSTM is intentionally small and briefly trained; a deeper network,
  longer training, multi-lead input, or an explicit RR-interval input
  channel would likely close (or reverse) the gap with the classical
  approaches -- this is exactly the kind of extension a follow-on paper
  would explore.
- Single-lead (MLII) input only; the database's second lead is unused.
- This is an offline, retrospective evaluation on a 1970s research
  database recorded from 47 subjects -- it says nothing about performance
  on modern recording hardware or a different patient population.

## 14. Future scope

- Multi-lead input and/or an explicit RR-interval channel fed into the
  CNN-BiLSTM alongside the raw waveform.
- Larger/longer-trained deep model, and a proper learning-rate schedule.
- Attention-based temporal models (e.g. a small Transformer encoder) as a
  fourth approach.
- External validation on a second database (e.g. INCART, PTB-XL) to test
  cross-database generalisation, not just cross-patient.
- A short IEEE-format paper summarising this methodology and these results
  (see `paper/README.md` -- not yet written).

## 15. References and dataset attribution

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.*
  IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
- Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet:
  Components of a New Research Resource for Complex Physiologic Signals.*
  Circulation 101(23):e215-e220 (2000).
  <https://physionet.org/content/mitdb/1.0.0/>
- de Chazal P, O'Dwyer M, Reilly RB. *Automatic classification of
  heartbeats using ECG morphology and heartbeat interval features.*
  IEEE Trans Biomed Eng 51(7):1196-1206 (2004). (Source of the DS1/DS2
  inter-patient split and the AAMI-mapping convention used here.)
- AAMI EC57:1998/(R)2008, *Testing and reporting performance results of
  cardiac rhythm and ST segment measurement algorithms.*

## Repository structure

```
ecg-arrhythmia-pattern-recognition/
├── README.md                    <- this file
├── requirements.txt
├── .gitignore
├── data/                        <- download target (not committed)
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── src/
│   ├── utils.py                 constants: split, AAMI mapping, seed
│   ├── data_loader.py           PhysioNet/wfdb download + load
│   ├── preprocessing.py         filtering, beat segmentation
│   ├── build_dataset.py         full pipeline -> cached beats_cache.npz
│   ├── feature_extraction.py    RR/HRV features, DWT features
│   ├── model_svm.py
│   ├── model_random_forest.py
│   ├── model_cnn_bilstm.py
│   ├── train.py                 main experiment driver
│   ├── evaluate.py              metrics, plots, report export
│   ├── generate_comparison.py   re-evaluate all saved models together
│   └── predict.py               CLI inference demo
├── app/
│   └── streamlit_app.py
├── results/
│   ├── metrics.csv
│   ├── per_record_summary.csv
│   ├── classification_reports/
│   └── figures/
├── models/                      <- trained artifacts (not committed)
├── paper/                       <- placeholder for a future IEEE paper
└── docs/
    └── methodology.md
```
