"""Comprehensive model evaluation and comparative benchmark script.

Evaluates the trained gesture classifier using Stratified K-Fold Cross Validation,
computes precision, recall, F1-scores, and benchmarks Random Forest against SVM and KNN.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification.gesture_classifier import GestureClassifier
from src.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    MODELS_DIR,
    gesture_config,
)
from src.features.dataset_manager import DatasetManager
from src.features.landmark_extractor import LandmarkFeatureExtractor


def run_comprehensive_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    n_splits: int = 5,
    benchmark_models: bool = True,
) -> Dict[str, Any]:
    """Run cross-validation, per-class metrics, and optional multi-model benchmark."""
    print("=" * 70)
    print(" HAND GESTURE CLASSIFIER EVALUATION & BENCHMARK")
    print("=" * 70)

    # 1. Load dataset
    dataset_manager = DatasetManager(dataset_path=dataset_path)
    if not dataset_path.exists():
        print(f"Dataset not found at: {dataset_path}. Generating synthetic dataset...")
        dataset_manager.create_synthetic_bootstrap_dataset(samples_per_class=100)

    df = dataset_manager.load_dataset()
    extractor = LandmarkFeatureExtractor()
    feature_cols = extractor.feature_names

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["gesture_id"].to_numpy(dtype=np.int32)
    classes = np.unique(y)
    class_names = [gesture_config.GESTURES[c] for c in classes]

    print(f"Loaded dataset: {len(df)} samples across {len(classes)} gesture classes.")

    # 2. Stratified K-Fold Cross Validation on Random Forest
    print(f"\n[1/3] Running {n_splits}-Fold Stratified Cross Validation on Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_scores = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")
    mean_cv_acc = float(np.mean(cv_scores))
    std_cv_acc = float(np.std(cv_scores))

    print(f"  -> Fold Accuracies: {[f'{s * 100:.2f}%' for s in cv_scores]}")
    print(f"  -> Mean CV Accuracy: {mean_cv_acc * 100:.2f}% (± {std_cv_acc * 100:.2f}%)")

    # 3. Out-of-fold predictions for full classification report
    print("\n[2/3] Generating Out-Of-Fold Classification Metrics...")
    oof_predictions = np.zeros(len(y), dtype=np.int32)
    for train_idx, val_idx in skf.split(X, y):
        rf_fold = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        rf_fold.fit(X[train_idx], y[train_idx])
        oof_predictions[val_idx] = rf_fold.predict(X[val_idx])

    report_dict = classification_report(
        y,
        oof_predictions,
        labels=classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y,
        oof_predictions,
        labels=classes,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report_text)

    results: Dict[str, Any] = {
        "mean_cv_accuracy": mean_cv_acc,
        "std_cv_accuracy": std_cv_acc,
        "fold_accuracies": [float(s) for s in cv_scores],
        "classification_report": report_dict,
    }

    # 4. Multi-model comparative benchmark (Random Forest vs SVM vs KNN)
    if benchmark_models:
        print("\n[3/3] Benchmarking Random Forest vs SVM vs KNN...")
        benchmark_results = _benchmark_classifiers(X, y, skf)
        results["benchmark"] = benchmark_results
        _display_benchmark_table(benchmark_results)
        _plot_benchmark(benchmark_results, MODELS_DIR / "model_comparison.png")

    # 5. Export JSON report
    report_path = MODELS_DIR / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation report saved to: {report_path}")
    print("=" * 70)

    return results


def _benchmark_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    cv_splitter: StratifiedKFold,
) -> Dict[str, Dict[str, float]]:
    """Compare Random Forest, SVM (RBF), and KNN on Accuracy, Train Time, and Inference Latency."""
    models: Dict[str, Any] = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", C=10.0, probability=True, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    }

    results: Dict[str, Dict[str, float]] = {}
    sample_feature = X[0].reshape(1, -1)

    for name, clf in models.items():
        # Accuracy via cross validation
        acc_scores = cross_val_score(clf, X, y, cv=cv_splitter, scoring="accuracy")
        mean_acc = float(np.mean(acc_scores))

        # Training latency measurement
        t0 = time.perf_counter()
        clf.fit(X, y)
        train_time_ms = (time.perf_counter() - t0) * 1000.0

        # Inference latency (average over 500 single-sample predictions)
        latencies = []
        for _ in range(500):
            t_inf_start = time.perf_counter()
            clf.predict(sample_feature)
            latencies.append((time.perf_counter() - t_inf_start) * 1000.0)
        avg_inf_latency_ms = float(np.mean(latencies))

        results[name] = {
            "cv_accuracy": mean_acc,
            "train_time_ms": train_time_ms,
            "inference_latency_ms": avg_inf_latency_ms,
        }

    return results


def _display_benchmark_table(benchmark_results: Dict[str, Dict[str, float]]) -> None:
    """Print formatted comparative benchmark results."""
    print("\n" + "-" * 70)
    print(f"{'Classifier':<18} | {'5-Fold CV Acc':<14} | {'Train Time (ms)':<16} | {'Inference Latency':<16}")
    print("-" * 70)
    for name, metrics in benchmark_results.items():
        acc_str = f"{metrics['cv_accuracy'] * 100:.2f}%"
        train_str = f"{metrics['train_time_ms']:.1f} ms"
        inf_str = f"{metrics['inference_latency_ms']:.3f} ms / frame"
        print(f"{name:<18} | {acc_str:<14} | {train_str:<16} | {inf_str:<16}")
    print("-" * 70)


def _plot_benchmark(
    benchmark_results: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Generate side-by-side bar plots comparing Accuracy and Latency."""
    names = list(benchmark_results.keys())
    accuracies = [benchmark_results[m]["cv_accuracy"] * 100.0 for m in names]
    latencies = [benchmark_results[m]["inference_latency_ms"] for m in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # 1. Accuracy comparison
    bars1 = ax1.bar(names, accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55)
    ax1.set_ylabel("5-Fold CV Accuracy (%)", fontsize=11)
    ax1.set_title("Cross-Validation Accuracy", fontsize=12, weight="bold")
    ax1.set_ylim([max(0, min(accuracies) - 10), 102])
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10, weight="bold")

    # 2. Latency comparison
    bars2 = ax2.bar(names, latencies, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55)
    ax2.set_ylabel("Inference Latency (ms / frame)", fontsize=11)
    ax2.set_title("Single-Sample Inference Latency", fontsize=12, weight="bold")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.2f}ms", ha="center", va="bottom", fontsize=10, weight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Benchmark comparison plot saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hand Gesture Recognition Classifier")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to CSV dataset")
    parser.add_argument("--splits", type=int, default=5, help="Number of folds for Stratified CV")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip comparison with SVM and KNN")
    args = parser.parse_args()

    run_comprehensive_evaluation(
        dataset_path=args.dataset,
        n_splits=args.splits,
        benchmark_models=not args.no_benchmark,
    )
