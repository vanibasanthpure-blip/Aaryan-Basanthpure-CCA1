"""Main experiment driver: builds the dataset once, trains all three models
on the identical patient-independent DS1 (train)/DS2 (test) split, and saves
metrics, figures, and model artifacts under results/ and models/.

Usage:
    python train.py --approach all
    python train.py --approach svm
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from build_dataset import build
from evaluate import (compute_metrics, plot_class_distribution, plot_confusion_matrix,
                       plot_model_comparison, plot_per_class_f1, plot_training_curves,
                       results_to_dataframe, save_classification_report, save_confusion_matrix_csv)
from feature_extraction import dwt_feature_matrix, rr_hrv_features
from model_cnn_bilstm import predict_cnn_bilstm, train_cnn_bilstm
from model_random_forest import train_rf
from model_svm import train_svm
from utils import AAMI_CLASSES, DS1_RECORDS, DS2_RECORDS, MODELS_DIR, RESULTS_DIR, SEED, set_seed

# SVC's RBF kernel does not scale to the tens of thousands of beats in DS1
# (training time grows faster than linearly with n). We cap beats-per-class
# for SVM training only; RF and CNN-BiLSTM below are trained on the full
# DS1 set since both scale to this size without issue. Documented in README.
SVM_MAX_PER_CLASS = 4000
N_CV_FOLDS = 5
VAL_FRACTION_OF_DS1 = 0.15  # held out from DS1, by record, for CNN-BiLSTM early stopping


def class_balanced_subsample(X, y, groups, max_per_class, seed=SEED):
    rng = np.random.RandomState(seed)
    keep_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx], groups[keep_idx]


def main(approach: str):
    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    data = build()
    beats, labels, groups = data["beats"], data["labels"], data["groups"]
    rr_prev, rr_next = data["rr_prev"], data["rr_next"]
    print(f"Total beats: {len(beats)} | class counts: {dict(zip(*np.unique(labels, return_counts=True)))}")

    plot_class_distribution(labels, "Class distribution - full dataset", "class_distribution_full.png")

    train_mask = np.isin(groups, DS1_RECORDS)
    test_mask = np.isin(groups, DS2_RECORDS)
    assert not (set(groups[train_mask]) & set(groups[test_mask])), "Patient leakage between DS1 and DS2!"
    print(f"DS1 (train) beats: {train_mask.sum()} | DS2 (test) beats: {test_mask.sum()}")
    plot_class_distribution(labels[train_mask], "Class distribution - DS1 (train)", "class_distribution_ds1.png")
    plot_class_distribution(labels[test_mask], "Class distribution - DS2 (test)", "class_distribution_ds2.png")

    y_test = labels[test_mask]
    results = {}

    # ---------------- Approach 1: RR/HRV + SVM ----------------
    if approach in ("all", "svm"):
        t0 = time.time()
        rr_feats_all = rr_hrv_features(rr_prev, rr_next)
        X_svm_train_full = rr_feats_all[train_mask]
        y_svm_train_full = labels[train_mask]
        groups_svm_train_full = groups[train_mask]

        X_svm_train, y_svm_train, g_svm_train = class_balanced_subsample(
            X_svm_train_full, y_svm_train_full, groups_svm_train_full, SVM_MAX_PER_CLASS)
        cv = list(GroupKFold(N_CV_FOLDS).split(X_svm_train, y_svm_train, groups=g_svm_train))

        svm_model = train_svm(X_svm_train, y_svm_train, cv)
        X_svm_test = rr_feats_all[test_mask]
        y_svm_pred = svm_model.predict(X_svm_test)
        y_svm_proba = svm_model.predict_proba(X_svm_test)
        # reorder proba columns to AAMI_CLASSES order
        y_svm_proba = y_svm_proba[:, [list(svm_model.classes_).index(c) for c in AAMI_CLASSES]]

        results["SVM_RR_HRV"] = compute_metrics(y_test, y_svm_pred, y_svm_proba)
        save_classification_report(y_test, y_svm_pred, "SVM_RR_HRV")
        plot_confusion_matrix(results["SVM_RR_HRV"]["confusion_matrix"], "SVM_RR_HRV")
        save_confusion_matrix_csv(results["SVM_RR_HRV"]["confusion_matrix"], "SVM_RR_HRV")
        joblib.dump(svm_model, os.path.join(MODELS_DIR, "svm_rr_hrv.joblib"))
        print(f"[SVM] done in {time.time() - t0:.1f}s")

    # ---------------- Approach 2: DWT + Random Forest ----------------
    if approach in ("all", "rf"):
        t0 = time.time()
        X_dwt_train = dwt_feature_matrix(beats[train_mask])
        y_rf_train = labels[train_mask]
        g_rf_train = groups[train_mask]
        cv = list(GroupKFold(N_CV_FOLDS).split(X_dwt_train, y_rf_train, groups=g_rf_train))

        rf_model = train_rf(X_dwt_train, y_rf_train, cv)
        X_dwt_test = dwt_feature_matrix(beats[test_mask])
        y_rf_pred = rf_model.predict(X_dwt_test)
        y_rf_proba = rf_model.predict_proba(X_dwt_test)
        y_rf_proba = y_rf_proba[:, [list(rf_model.classes_).index(c) for c in AAMI_CLASSES]]

        results["RF_DWT"] = compute_metrics(y_test, y_rf_pred, y_rf_proba)
        save_classification_report(y_test, y_rf_pred, "RF_DWT")
        plot_confusion_matrix(results["RF_DWT"]["confusion_matrix"], "RF_DWT")
        save_confusion_matrix_csv(results["RF_DWT"]["confusion_matrix"], "RF_DWT")
        joblib.dump(rf_model, os.path.join(MODELS_DIR, "rf_dwt.joblib"))
        print(f"[RF] done in {time.time() - t0:.1f}s")

    # ---------------- Approach 3: 1D-CNN + BiLSTM ----------------
    if approach in ("all", "cnn"):
        t0 = time.time()
        import torch

        class_to_idx = {c: i for i, c in enumerate(AAMI_CLASSES)}
        X_all_beats = beats
        y_idx = np.array([class_to_idx[l] for l in labels])

        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION_OF_DS1, random_state=SEED)
        ds1_idx = np.where(train_mask)[0]
        tr_sub, val_sub = next(gss.split(ds1_idx, groups=groups[ds1_idx]))
        tr_idx, val_idx = ds1_idx[tr_sub], ds1_idx[val_sub]
        assert not (set(groups[tr_idx]) & set(groups[val_idx])), "Patient leakage between CNN train/val!"

        X_cnn_train, y_cnn_train = X_all_beats[tr_idx], y_idx[tr_idx]
        X_cnn_val, y_cnn_val = X_all_beats[val_idx], y_idx[val_idx]
        X_cnn_test, y_cnn_test_idx = X_all_beats[test_mask], y_idx[test_mask]

        class_counts = np.bincount(y_cnn_train, minlength=len(AAMI_CLASSES))
        class_weights = (class_counts.sum() / (len(AAMI_CLASSES) * np.maximum(class_counts, 1)))
        # Q has single-digit support in DS1; raw inverse-frequency weighting
        # gives it a huge weight that destabilizes training (verified: an
        # uncapped run collapsed N recall to 0.38). Cap weights at 10x so no
        # one class can dominate the loss.
        class_weights = np.clip(class_weights, None, 10.0)

        cnn_model, history = train_cnn_bilstm(X_cnn_train, y_cnn_train, X_cnn_val, y_cnn_val,
                                               n_classes=len(AAMI_CLASSES), class_weights=class_weights)
        plot_training_curves(history)

        y_cnn_pred_idx, y_cnn_proba = predict_cnn_bilstm(cnn_model, X_cnn_test)
        y_cnn_pred = np.array([AAMI_CLASSES[i] for i in y_cnn_pred_idx])

        results["CNN_BiLSTM"] = compute_metrics(y_test, y_cnn_pred, y_cnn_proba)
        save_classification_report(y_test, y_cnn_pred, "CNN_BiLSTM")
        plot_confusion_matrix(results["CNN_BiLSTM"]["confusion_matrix"], "CNN_BiLSTM")
        save_confusion_matrix_csv(results["CNN_BiLSTM"]["confusion_matrix"], "CNN_BiLSTM")
        torch.save(cnn_model.state_dict(), os.path.join(MODELS_DIR, "cnn_bilstm.pt"))
        with open(os.path.join(RESULTS_DIR, "cnn_bilstm_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"[CNN-BiLSTM] done in {time.time() - t0:.1f}s")

    if results:
        df = results_to_dataframe(results)
        df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
        print(df[["model", "accuracy", "macro_f1", "weighted_f1", "roc_auc_macro_ovr"]])

        if len(results) > 1:
            plot_model_comparison(results)
            plot_per_class_f1(results)

    print("Done. See results/ for metrics, figures, and classification reports.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=["all", "svm", "rf", "cnn"], default="all")
    args = parser.parse_args()
    main(args.approach)
