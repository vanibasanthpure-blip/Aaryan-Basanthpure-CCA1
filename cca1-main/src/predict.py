"""Offline inference demo: load a MIT-BIH record, segment its beats, and run
all three trained models on it, printing predicted class + confidence.

This is an offline demonstration on recorded ECG signals, not a clinically
validated real-time monitoring system.
"""
import argparse
import os

import joblib
import numpy as np
import torch

from data_loader import load_record
from feature_extraction import dwt_feature_matrix, rr_hrv_features
from model_cnn_bilstm import CNNBiLSTM, predict_cnn_bilstm
from preprocessing import clean_signal, segment_beats
from utils import AAMI_CLASSES, MODELS_DIR

AAMI_DESCRIPTIONS = {
    "N": "Normal / bundle branch block beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Unknown / paced beat",
}


def load_models(models_dir: str = MODELS_DIR):
    models = {}
    svm_path = os.path.join(models_dir, "svm_rr_hrv.joblib")
    rf_path = os.path.join(models_dir, "rf_dwt.joblib")
    cnn_path = os.path.join(models_dir, "cnn_bilstm.pt")
    if os.path.exists(svm_path):
        models["SVM_RR_HRV"] = joblib.load(svm_path)
    if os.path.exists(rf_path):
        models["RF_DWT"] = joblib.load(rf_path)
    if os.path.exists(cnn_path):
        cnn = CNNBiLSTM(len(AAMI_CLASSES))
        cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))
        cnn.eval()
        models["CNN_BiLSTM"] = cnn
    return models


def prepare_record_beats(record_id: int):
    sig, fields, ann = load_record(record_id)
    sig_names = fields["sig_name"]
    channel = sig_names.index("MLII") if "MLII" in sig_names else 0
    cleaned = clean_signal(sig[:, channel], fs=fields["fs"])
    beats, labels, rr_prev, rr_next = segment_beats(cleaned, ann.sample, ann.symbol, fs=fields["fs"])
    return cleaned, beats, labels, rr_prev, rr_next


def predict_all(models: dict, beats: np.ndarray, rr_prev: np.ndarray, rr_next: np.ndarray):
    """Returns {model_name: (pred_labels[n], confidence[n])}."""
    preds = {}
    if "SVM_RR_HRV" in models:
        model = models["SVM_RR_HRV"]
        X = rr_hrv_features(rr_prev, rr_next)
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        pred = [classes[i] for i in proba.argmax(axis=1)]
        preds["SVM_RR_HRV"] = (np.array(pred), proba.max(axis=1))

    if "RF_DWT" in models:
        model = models["RF_DWT"]
        X = dwt_feature_matrix(beats)
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        pred = [classes[i] for i in proba.argmax(axis=1)]
        preds["RF_DWT"] = (np.array(pred), proba.max(axis=1))

    if "CNN_BiLSTM" in models:
        pred_idx, proba = predict_cnn_bilstm(models["CNN_BiLSTM"], beats)
        pred = np.array([AAMI_CLASSES[i] for i in pred_idx])
        preds["CNN_BiLSTM"] = (pred, proba.max(axis=1))

    return preds


def main():
    parser = argparse.ArgumentParser(description="Run trained models on one MIT-BIH record.")
    parser.add_argument("--record", type=int, default=100, help="MIT-BIH record number")
    parser.add_argument("--n-beats", type=int, default=10, help="number of beats to display")
    args = parser.parse_args()

    models = load_models()
    if not models:
        raise SystemExit("No trained models found in models/. Run train.py first.")

    _, beats, true_labels, rr_prev, rr_next = prepare_record_beats(args.record)
    preds = predict_all(models, beats, rr_prev, rr_next)

    n = min(args.n_beats, len(beats))
    print(f"Record {args.record}: showing predictions for first {n} beats\n")
    header = f"{'#':>3} {'true':>5}" + "".join(f"{m:>18}" for m in preds)
    print(header)
    for i in range(n):
        row = f"{i:>3} {true_labels[i]:>5}"
        for m, (p, conf) in preds.items():
            row += f"{p[i] + f' ({conf[i]:.2f})':>18}"
        print(row)


if __name__ == "__main__":
    main()
