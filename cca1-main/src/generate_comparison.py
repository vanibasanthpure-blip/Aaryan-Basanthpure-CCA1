"""Load the three already-trained models (saved under models/ by train.py)
and re-evaluate all of them together on the same DS2 test set, so the
combined metrics.csv / comparison plots reflect all three consistently --
without re-running training (which is expensive for CNN-BiLSTM).
"""
import numpy as np

from build_dataset import build
from evaluate import (compute_metrics, plot_confusion_matrix, plot_model_comparison,
                       plot_per_class_f1, results_to_dataframe, save_classification_report,
                       save_confusion_matrix_csv)
from feature_extraction import dwt_feature_matrix, rr_hrv_features
from model_cnn_bilstm import predict_cnn_bilstm
from predict import load_models
from utils import AAMI_CLASSES, DS2_RECORDS, RESULTS_DIR


def main():
    data = build()
    beats, labels, groups = data["beats"], data["labels"], data["groups"]
    rr_prev, rr_next = data["rr_prev"], data["rr_next"]
    test_mask = np.isin(groups, DS2_RECORDS)
    y_test = labels[test_mask]

    models = load_models()
    results = {}
    predictions = {}

    # NOTE: predicted labels always come from .predict(), not
    # .predict_proba().argmax() -- for SVC with probability=True, the Platt-
    # scaled probabilities are a separately calibrated model and can
    # disagree with the decision-function-based .predict() output (a
    # documented sklearn caveat). predict_proba is used only for ROC-AUC.
    if "SVM_RR_HRV" in models:
        model = models["SVM_RR_HRV"]
        X = rr_hrv_features(rr_prev, rr_next)[test_mask]
        pred = model.predict(X)
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        proba_ordered = proba[:, [classes.index(c) for c in AAMI_CLASSES]]
        predictions["SVM_RR_HRV"] = pred
        results["SVM_RR_HRV"] = compute_metrics(y_test, pred, proba_ordered)

    if "RF_DWT" in models:
        model = models["RF_DWT"]
        X = dwt_feature_matrix(beats[test_mask])
        pred = model.predict(X)
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        proba_ordered = proba[:, [classes.index(c) for c in AAMI_CLASSES]]
        predictions["RF_DWT"] = pred
        results["RF_DWT"] = compute_metrics(y_test, pred, proba_ordered)

    if "CNN_BiLSTM" in models:
        pred_idx, proba = predict_cnn_bilstm(models["CNN_BiLSTM"], beats[test_mask])
        pred = np.array([AAMI_CLASSES[i] for i in pred_idx])
        predictions["CNN_BiLSTM"] = pred
        results["CNN_BiLSTM"] = compute_metrics(y_test, pred, proba)

    for name, pred in predictions.items():
        save_classification_report(y_test, pred, name)
        plot_confusion_matrix(results[name]["confusion_matrix"], name)
        save_confusion_matrix_csv(results[name]["confusion_matrix"], name)

    df = results_to_dataframe(results)
    df.to_csv(f"{RESULTS_DIR}/metrics.csv", index=False)
    print(df[["model", "accuracy", "macro_f1", "weighted_f1", "roc_auc_macro_ovr"]])

    plot_model_comparison(results)
    plot_per_class_f1(results)
    print("Combined comparison written to results/")


if __name__ == "__main__":
    main()
