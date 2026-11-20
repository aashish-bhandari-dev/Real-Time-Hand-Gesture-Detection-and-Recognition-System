"""Shared test fixtures for the Hand Gesture Detection and Recognition System."""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.hand_detector import DetectedHand, LandmarkPoint


@pytest.fixture
def canonical_hand_landmarks() -> np.ndarray:
    """Generate reproducible 21-point canonical hand coordinates."""
    rng = np.random.default_rng(seed=42)
    coords = np.zeros((21, 3), dtype=np.float32)
    coords[0] = [0.5, 0.8, 0.0]

    for finger_idx in range(5):
        base_x = 0.36 + finger_idx * 0.07
        base_y = 0.58
        for joint_idx in range(4):
            lm_idx = 1 + finger_idx * 4 + joint_idx
            coords[lm_idx] = [
                base_x + rng.normal(0, 0.005),
                base_y - joint_idx * 0.08,
                rng.normal(0, 0.01),
            ]
    return coords


@pytest.fixture
def canonical_detected_hand(canonical_hand_landmarks) -> DetectedHand:
    """Provide a pre-built DetectedHand dataclass instance."""
    lms = [
        LandmarkPoint(x=float(r[0]), y=float(r[1]), z=float(r[2]))
        for r in canonical_hand_landmarks
    ]
    return DetectedHand(handedness="Right", confidence=0.97, landmarks=lms)
