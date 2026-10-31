"""
Test Set Evaluation Script for the Selected Best Model.

Strictly adheres to evaluation protocol:
  1. Loads the selected model from `models/best_model.joblib`.
  2. Evaluates ONCE on the untouched test partition (data/processed/X_test.csv, y_test.csv).
  3. Accepts raw feature inputs and processes them end-to-end through the embedded ColumnTransformer.
  4. Generates:
     - Comprehensive classification report.
     - Formatted confusion matrix with raw counts and normalized percentages.
     - High-resolution confusion matrix heatmap at reports/figures/confusion_matrix.png.
  5. Serializes test performance metrics to reports/test_evaluation.json.
"""

import json
import logging
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.models_def import StringTargetXGBClassifier  # Required for XGBoost unpickling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_LABELS = ["Safe", "At-Risk", "High-Risk"]


def load_test_data(
    X_path: str = "data/processed/X_test.csv",
    y_path: str = "data/processed/y_test.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the untouched raw test partition."""
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Test data files missing ({X_path}, {y_path}). Run src/preprocessing.py first."
        )
    X_test = pd.read_csv(X_path)
    y_test = pd.read_csv(y_path).squeeze()
    logger.info(f"Loaded test data: X_test shape={X_test.shape}, y_test shape={y_test.shape}")
    return X_test, y_test


def load_best_model(model_path: str = "models/best_model.joblib") -> object:
    """Load the serialized end-to-end best model pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Selected model not found at {model_path}. Run src/train_models.py first.")
    logger.info(f"Loading selected model from {model_path}")
    return joblib.load(model_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list = CLASS_LABELS,
    output_path: str = "reports/figures/confusion_matrix.png",
):
    """Plot and save publication-grade confusion matrix heatmap."""
    logger.info(f"Generating confusion matrix plot: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=300)

    # Annotate with count and row percentage
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_normalized[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_normalized,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Normalized Recall Rate (Row %)", "shrink": 0.8},
        linewidths=1.2,
        linecolor="#ffffff",
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
    )

    ax.set_title("Test Set Confusion Matrix: Selected Model (N = 20,000)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Risk Category", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Risk Category (Ground Truth)", fontsize=11, fontweight="bold", labelpad=10)

    # Footnote
    ax.text(
        0.5,
        -0.18,
        "Evaluated once on untouched test split. Cell values show absolute count and class-recall percentage.",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
        style="italic",
        color="#555555",
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def evaluate_on_test(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Execute single-pass test set evaluation and return full performance dictionary."""
    logger.info("Executing test set inference...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # High-Risk specific recall
    hr_recall = recall_score(y_test == "High-Risk", y_pred == "High-Risk", zero_division=0)

    clf_report_dict = classification_report(
        y_test, y_pred, labels=CLASS_LABELS, target_names=CLASS_LABELS, output_dict=True
    )
    clf_report_text = classification_report(
        y_test, y_pred, labels=CLASS_LABELS, target_names=CLASS_LABELS, digits=4
    )

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)

    # Verify directional critical errors: High-Risk misclassified as Safe
    hr_idx = CLASS_LABELS.index("High-Risk")
    safe_idx = CLASS_LABELS.index("Safe")
    catastrophic_errors = int(cm[hr_idx, safe_idx])

    results = {
        "test_sample_count": len(y_test),
        "overall_metrics": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_precision": float(prec_macro),
            "macro_recall": float(rec_macro),
            "macro_f1": float(f1_macro),
            "high_risk_recall": float(hr_recall),
        },
        "confusion_matrix": {
            "labels": CLASS_LABELS,
            "matrix": cm.tolist(),
            "critical_errors_high_risk_as_safe": catastrophic_errors,
        },
        "classification_report": clf_report_dict,
        "classification_report_text": clf_report_text,
    }
    return results, cm


def main():
    X_test, y_test = load_test_data()
    model = load_best_model()

    results, cm = evaluate_on_test(model, X_test, y_test)

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION REPORT (UNTOUCHED TEST SPLIT)")
    print("=" * 70)
    print(f"Total Test Samples: {results['test_sample_count']:,}")
    print(f"Overall Accuracy:   {results['overall_metrics']['accuracy']:.4f}")
    print(f"Balanced Accuracy:  {results['overall_metrics']['balanced_accuracy']:.4f}")
    print(f"Macro Precision:    {results['overall_metrics']['macro_precision']:.4f}")
    print(f"Macro Recall:       {results['overall_metrics']['macro_recall']:.4f}")
    print(f"Macro F1-Score:     {results['overall_metrics']['macro_f1']:.4f}")
    print(f"High-Risk Recall:   {results['overall_metrics']['high_risk_recall']:.4f}")
    print(f"Critical Errors (High-Risk predicted as Safe): {results['confusion_matrix']['critical_errors_high_risk_as_safe']}")
    print("-" * 70)
    print("DETAILED PER-CLASS CLASSIFICATION REPORT:")
    print(results["classification_report_text"])
    print("=" * 70 + "\n")

    # Plot confusion matrix
    plot_confusion_matrix(cm, classes=CLASS_LABELS, output_path="reports/figures/confusion_matrix.png")

    # Save evaluation results to JSON
    with open("reports/test_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved test evaluation results to reports/test_evaluation.json")


if __name__ == "__main__":
    main()
