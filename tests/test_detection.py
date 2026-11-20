"""Unit tests for hand detection data structures and visualization routines."""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.hand_detector import DetectedHand, HandDetectionResult, LandmarkPoint
from src.visualization.drawer import FINGERTIP_INDICES, HAND_CONNECTIONS, Visualizer


def test_landmark_point_to_numpy():
    """Verify conversion of 21 landmarks to a (21, 3) float32 numpy array."""
    landmarks = [LandmarkPoint(x=float(i) * 0.04, y=float(i) * 0.03, z=0.0) for i in range(21)]
    hand = DetectedHand(handedness="Right", confidence=0.98, landmarks=landmarks)

    arr = hand.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (21, 3)
    assert arr.dtype == np.float32
    assert np.isclose(arr[0, 0], 0.0)
    assert np.isclose(arr[1, 0], 0.04)


def test_visualizer_rendering():
    """Verify drawing functions operate on frames without raising errors or altering shape."""
    visualizer = Visualizer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Synthetic hand landmarks forming a simple grid
    landmarks = [
        LandmarkPoint(x=0.3 + (i % 5) * 0.08, y=0.3 + (i // 5) * 0.1, z=0.0)
        for i in range(21)
    ]
    hand = DetectedHand(handedness="Left", confidence=0.95, landmarks=landmarks)
    result = HandDetectionResult(hands=[hand], timestamp_ms=1000)

    # Test drawing landmarks
    frame_annotated = visualizer.draw_landmarks(frame, hand)
    assert frame_annotated.shape == (480, 640, 3)

    # Test drawing bounding box
    frame_boxed = visualizer.draw_bounding_box(frame, hand)
    assert frame_boxed.shape == (480, 640, 3)

    # Test drawing detection result
    frame_res = visualizer.draw_detection_result(frame, result)
    assert frame_res.shape == (480, 640, 3)

    # Test HUD overlay
    frame_hud = visualizer.draw_hud(frame, fps=30.0, num_hands=1)
    assert frame_hud.shape == (480, 640, 3)


def test_skeleton_connections_validity():
    """Verify all 21 landmark indices referenced in HAND_CONNECTIONS are within [0, 20]."""
    for start_idx, end_idx in HAND_CONNECTIONS:
        assert 0 <= start_idx < 21
        assert 0 <= end_idx < 21
        assert start_idx != end_idx

    for tip_idx in FINGERTIP_INDICES:
        assert 0 <= tip_idx < 21
