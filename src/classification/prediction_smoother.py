"""Prediction smoothing module using confidence-weighted rolling majority voting."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SmoothedPrediction:
    """Smoothed prediction result for a hand."""

    label: str
    confidence: float
    is_stable: bool  # True if majority vote exceeds 50% of the window


class PredictionSmoother:
    """Maintains a per-hand temporal rolling window to eliminate inter-frame label flickering."""

    def __init__(
        self,
        window_size: int = 7,
        unknown_label: str = "Unknown Gesture",
    ) -> None:
        if window_size < 1:
            raise ValueError(f"Window size must be >= 1, got {window_size}")

        self.window_size = window_size
        self.unknown_label = unknown_label
        # Map handedness ("Left", "Right", or "Default") to deque of (label, confidence)
        self._buffers: Dict[str, deque[Tuple[str, float]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def update(
        self,
        hand_id: str,
        label: str,
        confidence: float,
    ) -> SmoothedPrediction:
        """Add a new frame prediction and return the confidence-weighted majority label.

        Args:
            hand_id: Identifier for the hand (e.g. "Left", "Right", "0", "1").
            label: Raw predicted gesture label for the current frame.
            confidence: Associated prediction confidence in [0.0, 1.0].

        Returns:
            SmoothedPrediction containing consensus label and smoothed confidence.
        """
        buf = self._buffers[hand_id]
        buf.append((label, confidence))

        # Tally confidence-weighted scores per class
        weighted_scores: Dict[str, float] = defaultdict(float)
        class_confidences: Dict[str, List[float]] = defaultdict(list)

        for hist_label, hist_conf in buf:
            weighted_scores[hist_label] += hist_conf
            class_confidences[hist_label].append(hist_conf)

        # Winning class with highest weighted score
        winning_label = max(weighted_scores.keys(), key=lambda k: weighted_scores[k])
        # Average confidence of winning class within the window
        avg_confidence = float(
            sum(class_confidences[winning_label]) / len(class_confidences[winning_label])
        )

        # Stability: winning class appears in >= half of the window frames
        count_winning = sum(1 for l, _ in buf if l == winning_label)
        is_stable = count_winning >= (len(buf) // 2 + 1)

        return SmoothedPrediction(
            label=winning_label,
            confidence=avg_confidence,
            is_stable=is_stable,
        )

    def reset_hand(self, hand_id: str) -> None:
        """Clear historical prediction buffer for a specific hand."""
        if hand_id in self._buffers:
            self._buffers[hand_id].clear()

    def reset_all(self) -> None:
        """Clear all historical buffers."""
        self._buffers.clear()
