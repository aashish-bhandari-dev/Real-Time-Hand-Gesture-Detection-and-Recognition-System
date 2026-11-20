"""Unit tests for the PredictionSmoother rolling-window majority voter."""

from __future__ import annotations

import pytest

from src.classification.prediction_smoother import PredictionSmoother


def test_invalid_window_size():
    """Verify window size < 1 raises ValueError."""
    with pytest.raises(ValueError, match="Window size must be >= 1"):
        PredictionSmoother(window_size=0)


def test_majority_voting():
    """Verify majority vote selects dominant gesture over rolling window."""
    smoother = PredictionSmoother(window_size=5)

    # 4 frames of "Peace Sign", 1 frame of "Open Palm"
    smoother.update("Right", "Peace Sign", 0.90)
    smoother.update("Right", "Peace Sign", 0.92)
    smoother.update("Right", "Open Palm", 0.70)
    smoother.update("Right", "Peace Sign", 0.88)
    res = smoother.update("Right", "Peace Sign", 0.94)

    assert res.label == "Peace Sign"
    assert res.is_stable is True
    assert 0.85 < res.confidence < 0.95


def test_confidence_weighted_voting():
    """Verify high-confidence predictions outvote low-confidence noise."""
    smoother = PredictionSmoother(window_size=3)

    # Two weak votes for "Open Palm" (total confidence = 0.52 + 0.51 = 1.03)
    smoother.update("Left", "Open Palm", 0.52)
    smoother.update("Left", "Open Palm", 0.51)

    # One very strong vote for "Thumbs Up" (total confidence = 0.98)
    res1 = smoother.update("Left", "Thumbs Up", 0.98)
    assert res1.label == "Open Palm"  # 1.03 > 0.98

    # Next strong vote for "Thumbs Up" (0.99) pushes Thumbs Up ahead
    res2 = smoother.update("Left", "Thumbs Up", 0.99)
    assert res2.label == "Thumbs Up"


def test_independent_multi_hand_tracking():
    """Verify Left and Right hands maintain independent temporal buffers."""
    smoother = PredictionSmoother(window_size=3)

    smoother.update("Left", "Closed Fist", 0.95)
    smoother.update("Left", "Closed Fist", 0.95)
    left_res = smoother.update("Left", "Closed Fist", 0.95)

    smoother.update("Right", "Thumbs Up", 0.90)
    smoother.update("Right", "Thumbs Up", 0.90)
    right_res = smoother.update("Right", "Thumbs Up", 0.90)

    assert left_res.label == "Closed Fist"
    assert right_res.label == "Thumbs Up"


def test_reset_functionality():
    """Verify reset clears buffer."""
    smoother = PredictionSmoother(window_size=3)
    smoother.update("Right", "OK Sign", 0.99)
    smoother.update("Right", "OK Sign", 0.99)

    smoother.reset_hand("Right")
    # Fresh prediction after reset
    res = smoother.update("Right", "Stop Gesture", 0.80)
    assert res.label == "Stop Gesture"
