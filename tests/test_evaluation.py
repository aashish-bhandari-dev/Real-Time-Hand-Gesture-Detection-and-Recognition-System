"""Unit tests for model evaluation and multi-model benchmarking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from scripts.evaluate_model import _benchmark_classifiers


@pytest.fixture
def synthetic_eval_data():
    """Create reproducible synthetic dataset with 3 classes for fast testing."""
    np.random.seed(42)
    n_samples_per_class = 20
    n_features = 88

    X_list = []
    y_list = []
    for class_id in range(3):
        X_class = np.random.normal(loc=float(class_id) * 3.0, scale=0.5, size=(n_samples_per_class, n_features)).astype(
            np.float32
        )
        y_class = np.full(n_samples_per_class, class_id, dtype=np.int32)
        X_list.append(X_class)
        y_list.append(y_class)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def test_benchmark_classifiers(synthetic_eval_data):
    """Verify that multi-model benchmark executes and returns metrics for RF, SVM, and KNN."""
    X, y = synthetic_eval_data
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = _benchmark_classifiers(X, y, skf)

    assert "Random Forest" in results
    assert "SVM (RBF)" in results
    assert "KNN (k=5)" in results

    for model_name, metrics in results.items():
        assert "cv_accuracy" in metrics
        assert 0.0 <= metrics["cv_accuracy"] <= 1.0
        assert "train_time_ms" in metrics
        assert metrics["train_time_ms"] > 0.0
        assert "inference_latency_ms" in metrics
        assert metrics["inference_latency_ms"] > 0.0
