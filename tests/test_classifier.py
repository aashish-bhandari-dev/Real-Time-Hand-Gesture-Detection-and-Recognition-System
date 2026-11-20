"""Unit tests for the GestureClassifier model wrapper, inference, and serialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.classification.gesture_classifier import GestureClassifier, ModelNotLoadedError
from src.config import ClassifierConfig, gesture_config


@pytest.fixture
def synthetic_training_data():
    """Create reproducible synthetic features and labels for 3 classes."""
    np.random.seed(42)
    n_samples_per_class = 25
    n_features = 88

    X_list = []
    y_list = []
    for class_id in range(3):
        # Center each class around a distinct cluster
        mean_val = float(class_id) * 2.0
        X_class = np.random.normal(loc=mean_val, scale=0.5, size=(n_samples_per_class, n_features)).astype(
            np.float32
        )
        y_class = np.full(n_samples_per_class, class_id, dtype=np.int32)
        X_list.append(X_class)
        y_list.append(y_class)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def test_uninitialized_model_raises_error():
    """Verify inference on uninitialized classifier raises ModelNotLoadedError."""
    classifier = GestureClassifier()
    assert not classifier.is_loaded

    with pytest.raises(ModelNotLoadedError, match="model is not loaded"):
        classifier.predict(np.zeros(88, dtype=np.float32))

    with pytest.raises(ModelNotLoadedError, match="model is not loaded"):
        classifier.predict_proba(np.zeros(88, dtype=np.float32))


def test_fit_and_prediction(synthetic_training_data):
    """Verify fitting on training data and running predictions."""
    X, y = synthetic_training_data
    classifier = GestureClassifier()
    classifier.fit(X, y, n_estimators=20, random_state=42)

    assert classifier.is_loaded
    assert len(classifier.model.classes_) == 3

    # Predict single sample from class 0
    sample_c0 = X[0]
    label, confidence = classifier.predict(sample_c0)

    assert isinstance(label, str)
    assert 0.0 <= confidence <= 1.0
    assert label == gesture_config.GESTURES[0]


def test_confidence_threshold_gating(synthetic_training_data):
    """Verify low confidence returns unknown gesture label."""
    X, y = synthetic_training_data
    # Configure high confidence cutoff (0.999)
    cfg = ClassifierConfig(confidence_threshold=0.999, unknown_label="Custom Unknown")
    classifier = GestureClassifier(config=cfg)
    classifier.fit(X, y, n_estimators=10, random_state=42)

    # Point right between clusters to trigger uncertainty
    uncertain_sample = np.ones(88, dtype=np.float32) * 1.0
    label, conf = classifier.predict(uncertain_sample)

    assert label == "Custom Unknown"
    assert conf < 0.999


def test_model_serialization_roundtrip(synthetic_training_data, tmp_path: Path):
    """Verify saving to joblib and loading back preserves identical predictions."""
    X, y = synthetic_training_data
    model_path = tmp_path / "test_model.joblib"
    meta_path = tmp_path / "test_meta.json"

    classifier = GestureClassifier()
    classifier.fit(X, y, n_estimators=15, random_state=42)

    orig_label, orig_conf = classifier.predict(X[10])

    # Save
    classifier.save(model_path=model_path, metadata_path=meta_path)
    assert model_path.exists()
    assert meta_path.exists()

    # Load back
    loaded_classifier = GestureClassifier.load(model_path=model_path, metadata_path=meta_path)
    assert loaded_classifier.is_loaded

    loaded_label, loaded_conf = loaded_classifier.predict(X[10])
    assert orig_label == loaded_label
    assert np.isclose(orig_conf, loaded_conf)


def test_malformed_input_rejection(synthetic_training_data):
    """Verify dimension errors are caught."""
    X, y = synthetic_training_data
    classifier = GestureClassifier()
    classifier.fit(X, y, n_estimators=10, random_state=42)

    # 3D array should raise ValueError
    with pytest.raises(ValueError, match="Expected 1D or 2D feature array"):
        classifier.predict(np.zeros((2, 2, 88)))
