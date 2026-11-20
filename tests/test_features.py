"""Unit tests for hand landmark feature extraction and invariant normalization."""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.hand_detector import DetectedHand, LandmarkPoint
from src.features.landmark_extractor import LandmarkFeatureExtractor


@pytest.fixture
def sample_hand_landmarks() -> np.ndarray:
    """Generate realistic synthetic hand landmark coordinates of shape (21, 3)."""
    np.random.seed(42)
    # Start with wrist at (0.5, 0.8, 0.0)
    coords = np.zeros((21, 3), dtype=np.float32)
    coords[0] = [0.5, 0.8, 0.0]

    # Generate finger rays extending upward
    for finger_idx in range(5):
        base_x = 0.35 + finger_idx * 0.075
        base_y = 0.55
        for joint_idx in range(4):
            lm_idx = 1 + finger_idx * 4 + joint_idx
            coords[lm_idx] = [
                base_x + np.random.uniform(-0.01, 0.01),
                base_y - joint_idx * 0.09,
                np.random.uniform(-0.02, 0.02),
            ]
    return coords


def test_feature_vector_dimension_and_dtype(sample_hand_landmarks):
    """Verify that feature extraction yields exactly 88 float32 values."""
    extractor = LandmarkFeatureExtractor()
    features = extractor.extract_features(sample_hand_landmarks)

    assert isinstance(features, np.ndarray)
    assert features.shape == (88,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_feature_names_integrity():
    """Verify feature names list matches the fixed feature dimension."""
    extractor = LandmarkFeatureExtractor()
    names = extractor.feature_names

    assert len(names) == 88
    assert len(names) == len(set(names)), "Feature names must be unique"
    assert names[0] == "lm_0_x"
    assert "wrist_to_index_tip_dist" in names
    assert "cos_index_pip_angle" in names


def test_translation_invariance(sample_hand_landmarks):
    """Features must remain identical when the hand moves anywhere across the frame."""
    extractor = LandmarkFeatureExtractor()
    features_orig = extractor.extract_features(sample_hand_landmarks)

    # Apply large translation offset
    offset = np.array([125.4, -89.2, 45.1], dtype=np.float32)
    translated_landmarks = sample_hand_landmarks + offset
    features_trans = extractor.extract_features(translated_landmarks)

    np.testing.assert_allclose(
        features_orig,
        features_trans,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Extracted features are not translation invariant!",
    )


def test_scale_invariance(sample_hand_landmarks):
    """Features must remain identical when hand is scaled (e.g. closer or farther from camera)."""
    extractor = LandmarkFeatureExtractor()
    features_orig = extractor.extract_features(sample_hand_landmarks)

    # Center first to avoid translating, then scale by 3.5x
    wrist = sample_hand_landmarks[0]
    scaled_landmarks = (sample_hand_landmarks - wrist) * 3.5 + wrist
    features_scaled = extractor.extract_features(scaled_landmarks)

    np.testing.assert_allclose(
        features_orig,
        features_scaled,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Extracted features are not scale invariant!",
    )


def test_input_type_flexibility(sample_hand_landmarks):
    """Verify extractor accepts DetectedHand, List[LandmarkPoint], and np.ndarray."""
    extractor = LandmarkFeatureExtractor()

    # 1. np.ndarray
    f_arr = extractor.extract_features(sample_hand_landmarks)

    # 2. List of LandmarkPoint
    lm_list = [LandmarkPoint(x=row[0], y=row[1], z=row[2]) for row in sample_hand_landmarks]
    f_list = extractor.extract_features(lm_list)

    # 3. DetectedHand
    hand_obj = DetectedHand(handedness="Right", confidence=0.99, landmarks=lm_list)
    f_obj = extractor.extract_features(hand_obj)

    np.testing.assert_allclose(f_arr, f_list, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(f_arr, f_obj, rtol=1e-6, atol=1e-6)


def test_invalid_shape_rejection():
    """Verify ValueError is raised when landmark shape is not (21, 3)."""
    extractor = LandmarkFeatureExtractor()

    with pytest.raises(ValueError, match="Expected landmark numpy array of shape"):
        extractor.extract_features(np.zeros((20, 3)))

    with pytest.raises(ValueError, match="Expected landmark numpy array of shape"):
        extractor.extract_features(np.zeros((21, 2)))


def test_nan_and_inf_handling():
    """Verify coordinates with NaN or Inf are rejected safely."""
    extractor = LandmarkFeatureExtractor()
    corrupt_coords = np.zeros((21, 3), dtype=np.float32)
    corrupt_coords[5, 1] = np.nan

    with pytest.raises(ValueError, match="contain NaN or Inf"):
        extractor.extract_features(corrupt_coords)


def test_degenerate_zero_input():
    """Verify extractor does not crash on all-zero coordinates."""
    extractor = LandmarkFeatureExtractor()
    zero_coords = np.zeros((21, 3), dtype=np.float32)

    features = extractor.extract_features(zero_coords)
    assert features.shape == (88,)
    assert np.all(np.isfinite(features))
