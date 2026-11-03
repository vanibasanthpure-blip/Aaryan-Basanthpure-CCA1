# Methodology

## 1. Dataset

[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/),
48 half-hour two-lead ambulatory ECG recordings at 360 Hz, 11-bit resolution
over a 10 mV range, from 47 subjects (BIH Arrhythmia Laboratory, 1975-1979).
Each recording has independent beat-by-beat annotations (symbol + sample
index) produced by cardiologists and curated by PhysioNet.

### Records used and excluded

This project uses **44 of the 48 records**. Four are excluded:
**102, 104, 107, 217**. These four contain paced beats that dominate (or
entirely replace) the normal-sinus-rhythm content of the recording; with the
annotation mapping below they would contribute almost nothing to classes
N/S/V/F and would make the already tiny `Q` class behave inconsistently
across records. Excluding them is standard practice in the inter-patient
MIT-BIH classification literature (de Chazal et al., 2004) and is applied
identically to every model in this project -- no record is dropped for one
approach and kept for another.

### Patient-independent split (DS1 / DS2)

Beats are **never** split randomly across train/test: all beats from a given
patient's record go entirely into one side. This project uses the
de Chazal / AAMI-recommended **DS1/DS2 split**, applied identically to all
three models and to hyperparameter tuning:

- **DS1 (train, 22 records):** 101, 106, 108, 109, 112, 114, 115, 116, 118,
  119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
- **DS2 (test, 22 records):** 100, 103, 105, 111, 113, 117, 121, 123, 200,
  202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

DS2 is **only** used for the final evaluation reported in `results/`.
Hyperparameters for the SVM and Random Forest are selected by grid search
with `GroupKFold` cross-validation **within DS1**, grouped by record, so no
validation fold ever contains beats from a patient also present in a training
fold. The CNN-BiLSTM's early-stopping validation set is likewise carved out
of DS1 by record (`GroupShuffleSplit`), never from DS2.

### AAMI EC57 five-class mapping

| AAMI class | Meaning | MIT-BIH symbols mapped |
|---|---|---|
| N | Normal / bundle-branch-block beat | `N`, `L`, `R`, `e`, `j` |
| S | Supraventricular ectopic beat | `A`, `a`, `J`, `S` |
| V | Ventricular ectopic beat | `V`, `E` |
| F | Fusion beat | `F` |
| Q | Unknown / paced beat | `/`, `f`, `Q` |

All other annotation symbols (e.g. `+`, `~`, `|`, rhythm-change and
signal-quality markers) are not beats and are discarded before beat
segmentation; they never reach the labelled dataset.

With the four paced-dominated records excluded, class `Q` becomes extremely
rare in the remaining 44 records (around 15 beats total, split ~8/7 between
DS1/DS2). This is expected and is reported plainly in the results rather
than hidden -- per-class metrics for `Q` should be read as low-support and
not statistically reliable, and this is a known limitation of the AAMI
5-class protocol on this particular record subset.

### Realised class counts (from the actual cached pipeline run)

| Class | DS1 (train) | DS2 (test) |
|---|---|---|
| N | 45,847 | 44,239 |
| S | 944 | 1,837 |
| V | 3,788 | 3,220 |
| F | 414 | 388 |
| Q | 8 | 7 |
| **Total** | **51,001** | **49,691** |

These numbers match the widely-cited de Chazal DS1/DS2 counts closely,
confirming the segmentation pipeline follows the standard protocol.

## 2. Preprocessing pipeline

For every record (single lead, `MLII`):

1. **Baseline wander removal** -- two-stage median filter (≈200 ms then
   ≈600 ms windows). Preferred over a naive high-pass filter because it
   removes slow drift without distorting the QRS complex or ST segment.
2. **Bandpass filtering** -- 3rd-order Butterworth, 0.5-40 Hz, applied via
   zero-phase `filtfilt`. 40 Hz retains QRS/T-wave morphology while
   attenuating high-frequency muscle noise and 60 Hz mains interference
   (MIT-BIH was recorded in the US), so a separate notch filter is
   unnecessary.
3. **Beat segmentation** -- fixed window of 0.28 s before and 0.40 s after
   each annotated R-peak (244 samples at 360 Hz), covering the QRS complex
   and T-wave. Windows that fall off the edge of the recording, contain
   non-finite values, or are flat (std ≈ 0, a recording artefact) are
   dropped rather than zero-padded, to avoid feeding the models synthetic
   segments.
4. **Per-beat normalisation** -- zero mean, unit variance, so amplitude
   differences across leads/patients/gain settings don't dominate learned
   features.
5. **RR interval bookkeeping** -- previous/next RR interval (seconds) is
   computed from the *filtered beat-annotation sequence only* (rhythm and
   quality markers excluded first), so HRV features are never computed
   against a non-beat marker.

## 3. Feature sets per approach

- **Approach 1 (SVM):** `rr_prev`, `rr_next`, `mean_rr`, `rr_ratio`,
  local RR-difference magnitude (a single-beat proxy for RMSSD), and
  instantaneous heart rate -- 6 features, standardised before the SVM.
- **Approach 2 (Random Forest):** Daubechies-4 (`db4`) discrete wavelet
  decomposition to level 4 of the normalised beat waveform; mean, std, min,
  max, energy and Shannon entropy of each of the 5 resulting coefficient
  bands -- 30 features.
- **Approach 3 (CNN-BiLSTM):** the normalised 244-sample raw beat waveform
  directly; the network learns its own morphological/temporal features.

## 4. Models

See `src/model_svm.py`, `src/model_random_forest.py`,
`src/model_cnn_bilstm.py` for exact architectures/hyperparameter grids.
Only `GroupKFold`/`GroupShuffleSplit` cross-validation **within DS1** is used
for model/hyperparameter selection; DS2 is touched exactly once, at the end,
for the reported metrics.

One documented compute trade-off: the SVM (RBF kernel) does not scale to
tens of thousands of training points in reasonable time, so it is trained on
a class-balanced subsample of DS1 capped at 4,000 beats/class
(`SVM_MAX_PER_CLASS` in `src/train.py`). The Random Forest and CNN-BiLSTM are
both trained on the **full** DS1 set, since both scale to this size without
difficulty. All three are evaluated on the complete, unmodified DS2 test set.

## 5. Evaluation

Accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, confusion
matrices, and one-vs-rest macro ROC-AUC, all computed from actual model
predictions on DS2 (see `src/evaluate.py`). Given the severe class imbalance
(N is ~90% of beats), **macro-F1 and per-class recall for S/V/F/Q are the
primary comparison metrics** -- accuracy alone is not meaningful here.
