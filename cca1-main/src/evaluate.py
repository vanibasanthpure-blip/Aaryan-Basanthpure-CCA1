"""Shared evaluation metrics, CSV export, and plotting for all three models.

All reported numbers come from the confusion matrix / sklearn metrics
computed on actual model predictions on the DS2 held-out test set -- nothing
here fabricates a number.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                              precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import label_binarize

from utils import AAMI_CLASSES, RESULTS_DIR

sns.set_theme(style="whitegrid")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
REPORT_DIR = os.path.join(RESULTS_DIR, "classification_reports")


def compute_metrics(y_true, y_pred, y_proba=None, classes=AAMI_CLASSES) -> dict:
    acc = (np.asarray(y_true) == np.asarray(y_pred)).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)

    roc_auc = np.nan
    if y_proba is not None:
        present = sorted(set(y_true) | set(y_pred))
        if len(present) > 1:
            y_bin = label_binarize(y_true, classes=classes)
            try:
                roc_auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
            except ValueError:
                roc_auc = np.nan

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc_macro_ovr": roc_auc,
        "per_class": {c: {"precision": precision[i], "recall": recall[i],
                           "f1": f1[i], "support": int(support[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes),
    }


def save_classification_report(y_true, y_pred, model_name, classes=AAMI_CLASSES):
    os.makedirs(REPORT_DIR, exist_ok=True)
    report = classification_report(y_true, y_pred, labels=classes, zero_division=0)
    path = os.path.join(REPORT_DIR, f"{model_name}_classification_report.txt")
    with open(path, "w") as f:
        f.write(report)
    return path


def save_confusion_matrix_csv(cm, model_name, classes=AAMI_CLASSES):
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"{model_name}_confusion_matrix.csv")
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(path)
    return path


def plot_confusion_matrix(cm, model_name, classes=AAMI_CLASSES):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_class_distribution(labels, title, filename):
    os.makedirs(FIG_DIR, exist_ok=True)
    vals, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(vals, counts, color=sns.color_palette("Blues_d", len(vals)))
    ax.set_xlabel("AAMI class")
    ax.set_ylabel("Beat count")
    ax.set_yscale("log")
    ax.set_title(title)
    for v, c in zip(vals, counts):
        ax.text(v, c, str(c), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_model_comparison(results: dict, filename="model_comparison.png"):
    os.makedirs(FIG_DIR, exist_ok=True)
    models = list(results.keys())
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(metrics):
        vals = [results[model][m] for model in models]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_title("Model performance comparison (DS2 test set)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_class_f1(results: dict, classes=AAMI_CLASSES, filename="per_class_f1_comparison.png"):
    os.makedirs(FIG_DIR, exist_ok=True)
    models = list(results.keys())
    x = np.arange(len(classes))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, model in enumerate(models):
        vals = [results[model]["per_class"][c]["f1"] for c in classes]
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1-score comparison (DS2 test set)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_curves(history: dict, filename="cnn_bilstm_training_curves.png"):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("CNN-BiLSTM loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["val_macro_f1"], color="green")
    axes[1].set_title("Validation macro-F1")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def results_to_dataframe(results: dict, classes=AAMI_CLASSES) -> pd.DataFrame:
    rows = []
    for model, r in results.items():
        row = {"model": model, "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
               "weighted_f1": r["weighted_f1"], "roc_auc_macro_ovr": r["roc_auc_macro_ovr"]}
        for c in classes:
            row[f"precision_{c}"] = r["per_class"][c]["precision"]
            row[f"recall_{c}"] = r["per_class"][c]["recall"]
            row[f"f1_{c}"] = r["per_class"][c]["f1"]
            row[f"support_{c}"] = r["per_class"][c]["support"]
        rows.append(row)
    return pd.DataFrame(rows)
