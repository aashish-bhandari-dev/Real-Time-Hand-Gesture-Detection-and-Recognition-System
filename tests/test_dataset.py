"""Unit tests for dataset recording, schema validation, and storage operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import gesture_config
from src.features.dataset_manager import DatasetManager
from src.features.landmark_extractor import LandmarkFeatureExtractor


@pytest.fixture
def temp_dataset_path(tmp_path: Path) -> Path:
    return tmp_path / "test_gestures.csv"


def test_create_record_schema(temp_dataset_path):
    """Verify that create_record constructs a row dictionary with 88 features + 4 metadata keys."""
    manager = DatasetManager(dataset_path=temp_dataset_path)
    features = np.ones(88, dtype=np.float32) * 0.5

    record = manager.create_record(
        features=features,
        gesture_name="Peace Sign",
        gesture_id=3,
        handedness="Right",
        timestamp=1700000000.0,
    )

    assert len(record) == 88 + 4
    assert record["gesture_name"] == "Peace Sign"
    assert record["gesture_id"] == 3
    assert record["handedness"] == "Right"
    assert "lm_0_x" in record
    assert "ratio_pinky_ext" in record


def test_save_and_load_roundtrip(temp_dataset_path):
    """Verify writing records to CSV and reading back preserves data types and row counts."""
    manager = DatasetManager(dataset_path=temp_dataset_path)

    records = []
    for i in range(10):
        feat = np.full(88, i * 0.1, dtype=np.float32)
        records.append(
            manager.create_record(
                features=feat,
                gesture_name="Open Palm",
                gesture_id=0,
                handedness="Right",
                timestamp=float(i),
            )
        )

    saved_count = manager.append_samples(records)
    assert saved_count == 10
    assert temp_dataset_path.exists()

    df = manager.load_dataset()
    assert len(df) == 10
    assert "gesture_name" in df.columns
    assert df["gesture_name"].iloc[0] == "Open Palm"
    assert np.isclose(df["lm_0_x"].iloc[5], 0.5)


def test_class_distribution_counting(temp_dataset_path):
    """Verify get_class_distribution correctly tallies sample counts per class."""
    manager = DatasetManager(dataset_path=temp_dataset_path)

    # Initially empty
    dist = manager.get_class_distribution()
    assert dist["Open Palm"] == 0
    assert dist["Closed Fist"] == 0

    # Add 5 Open Palm and 3 Closed Fist
    records = []
    for _ in range(5):
        records.append(
            manager.create_record(
                features=np.zeros(88, dtype=np.float32),
                gesture_name="Open Palm",
                gesture_id=0,
                handedness="Left",
                timestamp=0.0,
            )
        )
    for _ in range(3):
        records.append(
            manager.create_record(
                features=np.zeros(88, dtype=np.float32),
                gesture_name="Closed Fist",
                gesture_id=1,
                handedness="Right",
                timestamp=0.0,
            )
        )

    manager.append_samples(records)
    dist = manager.get_class_distribution()

    assert dist["Open Palm"] == 5
    assert dist["Closed Fist"] == 3
    assert dist["Peace Sign"] == 0


def test_schema_validation_rejection(temp_dataset_path):
    """Verify validate_schema rejects dataframes missing required columns."""
    manager = DatasetManager(dataset_path=temp_dataset_path)

    # Incomplete dataframe
    bad_df = pd.DataFrame({"col_a": [1, 2], "gesture_name": ["Open Palm", "Closed Fist"]})

    with pytest.raises(ValueError, match="missing .* expected feature columns"):
        manager.validate_schema(bad_df)


def test_synthetic_bootstrap_dataset(temp_dataset_path):
    """Verify synthetic dataset generator creates balanced samples for all 10 gestures."""
    manager = DatasetManager(dataset_path=temp_dataset_path)
    df = manager.create_synthetic_bootstrap_dataset(samples_per_class=5)

    assert len(df) == 5 * len(gesture_config.GESTURES)
    counts = df["gesture_name"].value_counts()
    for g_name in gesture_config.GESTURES:
        assert counts[g_name] == 5
