"""Geometric rule-based gesture classifier.

Analyzes anatomical landmark geometry, joint flexions, fingertip distances,
and finger extension states to accurately recognize all 10 gestures in real time.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.detection.hand_detector import DetectedHand, LandmarkPoint


class RuleBasedGestureClassifier:
    """Classifies gestures using anatomical hand geometry and finger extension rules."""

    def predict_hand(self, hand: DetectedHand) -> Tuple[str, float]:
        """Classify gesture directly from 21 hand landmarks."""
        coords = hand.to_numpy()  # (21, 3)
        return self.predict_coordinates(coords)

    def predict_coordinates(self, coords: np.ndarray) -> Tuple[str, float]:
        """Classify gesture from (21, 3) landmark coordinates.

        Returns:
            Tuple of (gesture_name, confidence).
        """
        # 1. Translation invariant: center on wrist (0)
        wrist = coords[0]
        centered = coords - wrist

        # 2. Scale invariant: normalize by palm size (wrist to middle MCP joint 9)
        palm_size = float(np.linalg.norm(centered[9]))
        if palm_size < 1e-6:
            palm_size = 1.0
        norm_coords = centered / palm_size

        # 3. Determine extension state for each of the 4 fingers
        # Index (5, 6, 7, 8), Middle (9, 10, 11, 12), Ring (13, 14, 15, 16), Pinky (17, 18, 19, 20)
        is_index_extended = self._is_finger_extended(norm_coords, tip_idx=8, pip_idx=6, mcp_idx=5)
        is_middle_extended = self._is_finger_extended(norm_coords, tip_idx=12, pip_idx=10, mcp_idx=9)
        is_ring_extended = self._is_finger_extended(norm_coords, tip_idx=16, pip_idx=14, mcp_idx=13)
        is_pinky_extended = self._is_finger_extended(norm_coords, tip_idx=20, pip_idx=18, mcp_idx=17)

        # 4. Robust Thumb extension detection
        thumb_tip = norm_coords[4]
        thumb_ip = norm_coords[3]
        thumb_mcp = norm_coords[2]
        index_mcp = norm_coords[5]
        middle_mcp = norm_coords[9]

        dist_thumb_tip_index_mcp = float(np.linalg.norm(thumb_tip - index_mcp))
        dist_thumb_tip_middle_mcp = float(np.linalg.norm(thumb_tip - middle_mcp))
        dist_thumb_tip_wrist = float(np.linalg.norm(thumb_tip))
        dist_thumb_mcp_wrist = float(np.linalg.norm(thumb_mcp))

        # Thumb is folded across palm if tip is tucked close to index/middle MCP
        is_thumb_folded = (dist_thumb_tip_middle_mcp < 0.55) or (dist_thumb_tip_index_mcp < 0.38)
        # Thumb is extended if it reaches outward away from index base and wrist
        is_thumb_extended = (not is_thumb_folded) and (dist_thumb_tip_wrist > dist_thumb_mcp_wrist * 1.08)

        # OK Sign check: thumb tip (4) touches index tip (8)
        dist_thumb_index_tips = float(np.linalg.norm(norm_coords[4] - norm_coords[8]))
        is_ok_circle = dist_thumb_index_tips < 0.35

        # Count how many of the 4 main fingers are extended
        extended_fingers_count = sum(
            [is_index_extended, is_middle_extended, is_ring_extended, is_pinky_extended]
        )

        # Check if thumb points upward for Thumbs Up (negative Y is UP in image space)
        thumb_pointing_up = (thumb_tip[1] < thumb_mcp[1] - 0.25) and (abs(thumb_tip[0] - thumb_mcp[0]) < 0.85)

        # 5. Gesture Rule Matching Logic

        # OK Sign: thumb and index tips touching, while other fingers are mostly extended
        if is_ok_circle and (is_middle_extended or is_ring_extended):
            return "OK Sign", 0.95

        # Closed Fist: all 4 fingers curled, thumb folded
        if extended_fingers_count == 0 and not is_thumb_extended:
            return "Closed Fist", 0.96

        # Thumbs Up: only thumb extended upward, 4 fingers folded
        if thumb_pointing_up and extended_fingers_count == 0:
            return "Thumbs Up", 0.95

        # One Finger: only index finger extended
        if is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
            return "One Finger", 0.94

        # Peace Sign: index and middle extended in V-shape, ring & pinky curled
        if is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended:
            return "Peace Sign", 0.95

        # Three Fingers:
        # Case A: Index + Middle + Ring extended, pinky curled, thumb folded
        # Case B: Thumb + Index + Middle extended, ring & pinky curled
        if is_index_extended and is_middle_extended and is_ring_extended and not is_pinky_extended and not is_thumb_extended:
            return "Three Fingers", 0.93
        if is_thumb_extended and is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended:
            return "Three Fingers", 0.93

        # Four Fingers: all 4 fingers extended, thumb EXPLICITLY folded/tucked inward
        if extended_fingers_count == 4 and is_thumb_folded:
            return "Four Fingers", 0.94

        # All 5 fingers extended (or 4 fingers + thumb extended)
        if extended_fingers_count == 4 and is_thumb_extended:
            # Measure spread between fingertips
            adjacent_tip_spread = float(
                np.mean(
                    [
                        np.linalg.norm(norm_coords[8] - norm_coords[12]),
                        np.linalg.norm(norm_coords[12] - norm_coords[16]),
                        np.linalg.norm(norm_coords[16] - norm_coords[20]),
                    ]
                )
            )
            thumb_to_pinky_spread = float(np.linalg.norm(norm_coords[4] - norm_coords[20]))

            # Fingers held together pointing up = Stop Gesture
            if adjacent_tip_spread < 0.26:
                return "Stop Gesture", 0.93

            # Wide spread = Five Fingers
            if thumb_to_pinky_spread > 1.35 or adjacent_tip_spread > 0.32:
                return "Five Fingers", 0.95

            # Default open hand = Open Palm
            return "Open Palm", 0.92

        # Robust fallbacks for minor landmark tracking variations
        if extended_fingers_count >= 3 and is_thumb_extended:
            return "Five Fingers", 0.90
        if extended_fingers_count == 4:
            # If thumb is somewhat ambiguous, default to Five Fingers unless thumb is tightly tucked
            if is_thumb_folded:
                return "Four Fingers", 0.90
            return "Five Fingers", 0.88
        if is_index_extended and is_middle_extended:
            return "Peace Sign", 0.86
        if extended_fingers_count == 0:
            return "Closed Fist", 0.86

        return "Unknown Gesture", 0.45

    @staticmethod
    def _is_finger_extended(
        norm_coords: np.ndarray,
        tip_idx: int,
        pip_idx: int,
        mcp_idx: int,
    ) -> bool:
        """A finger is extended if its tip distance from wrist exceeds both PIP and MCP distances."""
        tip_dist = np.linalg.norm(norm_coords[tip_idx])
        pip_dist = np.linalg.norm(norm_coords[pip_idx])
        mcp_dist = np.linalg.norm(norm_coords[mcp_idx])

        return bool((tip_dist > pip_dist * 1.10) and (tip_dist > mcp_dist * 1.25))
