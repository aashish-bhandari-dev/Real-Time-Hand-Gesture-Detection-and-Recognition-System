"""Train Random Forest classifier on the collected hand gesture feature dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # Headless backend for plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification.gesture_classifier import GestureClassifier
from src.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    gesture_config,
    training_config,
)
from src.features.dataset_manager import DatasetManager
from src.features.landmark_extractor import LandmarkFeatureExtractor


def train_gesture_classifier(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_output_path: Path = DEFAULT_MODEL_PATH,
    metadata_output_path: Path = MODEL_METADATA_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int | None = 20,
) -> GestureClassifier:
    """Load dataset, train Random Forest, evaluate performance, and persist model."""
    print("=" * 65)
    print(" HAND GESTURE MODEL TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load and validate dataset
    dataset_manager = DatasetManager(dataset_path=dataset_path)
    if not dataset_path.exists():
        print(f"\n[WARNING] Dataset file not found at: {dataset_path}")
        print("Auto-generating a synthetic baseline dataset for training...")
        dataset_manager.create_synthetic_bootstrap_dataset(samples_per_class=100)

    df = dataset_manager.load_dataset()
    if df.empty or len(df) < 20:
        raise ValueError(
            f"Dataset at {dataset_path} contains insufficient samples ({len(df)}). "
            "Record more samples using 'python scripts/collect_data.py'."
        )

    print(f"\nLoaded dataset with {len(df)} total samples.")

    # 2. Check class distribution
    class_counts = df["gesture_name"].value_counts()
    print("\nClass distribution:")
    for name, cnt in class_counts.items():
        print(f"  - {name:<18} : {cnt:>4} samples")

    num_classes = df["gesture_id"].nunique()
    if num_classes < 2:
        raise ValueError("Dataset must contain at least 2 distinct classes to train a classifier.")

    # 3. Extract features X and targets y
    extractor = LandmarkFeatureExtractor()
    feature_cols = extractor.feature_names

    # Verify no missing values
    null_counts = df[feature_cols].isnull().sum().sum()
    if null_counts > 0:
        print(f"[WARNING] Found {null_counts} NaN values. Dropping invalid rows...")
        df = df.dropna(subset=feature_cols)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["gesture_id"].to_numpy(dtype=np.int32)

    # 4. Stratified Train-Test Split
    print(f"\nSplitting data: {100 * (1 - test_size):.0f}% Train / {100 * test_size:.0f}% Test (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Training samples: {len(X_train)}  |  Testing samples: {len(X_test)}")

    # 5. Fit Random Forest Classifier
    print(f"\nFitting RandomForestClassifier (n_estimators={n_estimators}, max_depth={max_depth})...")
    classifier = GestureClassifier()
    classifier.fit(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    # 6. Evaluate Model on Test Set
    y_pred = classifier.model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    print("\n" + "=" * 65)
    print(f" MODEL EVALUATION RESULTS (Accuracy: {accuracy * 100:.2f}%)")
    print("=" * 65)

    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    target_names = [gesture_config.GESTURES[cid] for cid in unique_classes]

    report = classification_report(
        y_test,
        y_pred,
        labels=unique_classes,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print("\nClassification Report:\n")
    print(report)

    # 7. Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cm_plot_path = MODELS_DIR / "confusion_matrix.png"
    _plot_and_save_confusion_matrix(cm, target_names, cm_plot_path)
    print(f"Confusion matrix plot saved to: {cm_plot_path}")

    # 8. Save model checkpoint and metadata
    evaluation_metadata = {
        "dataset_path": str(dataset_path),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": accuracy,
        "classes_evaluated": [int(c) for c in unique_classes],
    }
    classifier.save(
        model_path=model_output_path,
        metadata_path=metadata_output_path,
        extra_metadata=evaluation_metadata,
    )

    print(f"\nModel checkpoint serialized to: {model_output_path}")
    print(f"Metadata exported to: {metadata_output_path}")
    print("=" * 65)
    return classifier


def _plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Render a publication-quality confusion matrix heatmap using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10, weight="bold")

    fig.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Gesture", fontsize=11, labelpad=10)
    ax.set_ylabel("True Gesture", fontsize=11)
    ax.set_title("Hand Gesture Recognition - Confusion Matrix", fontsize=13, weight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hand Gesture Recognition Random Forest Model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to input CSV dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Output path for .joblib model")
    parser.add_argument("--trees", type=int, default=100, help="Number of trees in Random Forest")
    args = parser.parse_args()

    train_gesture_classifier(
        dataset_path=args.dataset,
        model_output_path=args.output,
        n_estimators=args.trees,
    )
