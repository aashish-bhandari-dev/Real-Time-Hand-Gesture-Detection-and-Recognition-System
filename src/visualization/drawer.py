"""Visual overlay and rendering utilities for hand landmarks, skeleton connections, and HUD."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection.hand_detector import DetectedHand, HandDetectionResult

# 21 Landmark standard skeleton connections (pairs of landmark indices)
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Palm base & Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky finger
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

# Landmark indices for fingertips
FINGERTIP_INDICES = [4, 8, 12, 16, 20]


class Visualizer:
    """Renders hand landmarks, connections, bounding boxes, and HUD metrics onto OpenCV frames."""

    def __init__(
        self,
        connection_color: Tuple[int, int, int] = (0, 200, 255),  # Gold / Amber
        joint_color: Tuple[int, int, int] = (255, 255, 255),       # White
        tip_color: Tuple[int, int, int] = (0, 255, 0),            # Bright Green
        line_thickness: int = 2,
        joint_radius: int = 4,
        tip_radius: int = 6,
    ) -> None:
        self.connection_color = connection_color
        self.joint_color = joint_color
        self.tip_color = tip_color
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius
        self.tip_radius = tip_radius

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand: DetectedHand,
        custom_color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Draw 21 landmarks and skeletal connection lines for a single hand."""
        h, w = frame.shape[:2]

        # Vectorized coordinate scaling
        coords = hand.to_numpy()
        xs = np.clip(coords[:, 0] * w, 0, w - 1).astype(np.int32)
        ys = np.clip(coords[:, 1] * h, 0, h - 1).astype(np.int32)
        pixel_points = list(zip(xs, ys))

        line_col = custom_color or self.connection_color

        # 1. Draw connection lines
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = pixel_points[start_idx]
            pt2 = pixel_points[end_idx]
            cv2.line(frame, pt1, pt2, line_col, self.line_thickness, cv2.LINE_AA)

        # 2. Draw joints
        for idx, pt in enumerate(pixel_points):
            if idx in FINGERTIP_INDICES:
                cv2.circle(frame, pt, self.tip_radius, self.tip_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, self.tip_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, self.joint_radius, self.joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, self.joint_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        hand: DetectedHand,
        label: Optional[str] = None,
        confidence: Optional[float] = None,
        is_stable: bool = True,
        box_color: Tuple[int, int, int] = (50, 205, 50),
        margin: int = 20,
    ) -> np.ndarray:
        """Draw an informative bounding box with gesture badge, confidence bar, and stability."""
        h, w = frame.shape[:2]
        coords = hand.to_numpy()
        xs = coords[:, 0] * w
        ys = coords[:, 1] * h

        x_min = max(0, int(np.min(xs)) - margin)
        y_min = max(0, int(np.min(ys)) - margin)
        x_max = min(w - 1, int(np.max(xs)) + margin)
        y_max = min(h - 1, int(np.max(ys)) + margin)

        # Draw main bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2, cv2.LINE_AA)

        # Build label text
        display_text = label if label else f"{hand.handedness} Hand"
        conf_text = f" {confidence * 100:.0f}%" if confidence is not None else ""
        badge_text = f"{display_text}{conf_text}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.58
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)

        badge_width = max(text_w + 14, 110)
        badge_height = text_h + 14
        badge_y = max(y_min - badge_height - 6, 4)

        # Optimized slice-based alpha blending for badge background
        x2 = min(w, x_min + badge_width)
        y2 = min(h, badge_y + badge_height + 8)
        if x2 > x_min and y2 > badge_y:
            sub_rect = frame[badge_y:y2, x_min:x2]
            dark_overlay = np.zeros_like(sub_rect)
            dark_overlay[:] = (20, 20, 20)
            cv2.addWeighted(dark_overlay, 0.85, sub_rect, 0.15, 0, sub_rect)

        # Text label
        cv2.putText(
            frame,
            badge_text,
            (x_min + 6, badge_y + text_h + 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        # Mini horizontal confidence bar
        if confidence is not None:
            bar_y = badge_y + badge_height + 4
            bar_w = badge_width - 12
            cv2.rectangle(
                frame,
                (x_min + 6, bar_y),
                (x_min + 6 + bar_w, bar_y + 3),
                (60, 60, 60),
                -1,
            )
            filled_w = int(bar_w * np.clip(confidence, 0.0, 1.0))
            cv2.rectangle(
                frame,
                (x_min + 6, bar_y),
                (x_min + 6 + filled_w, bar_y + 3),
                box_color,
                -1,
            )

        return frame

    def draw_detection_result(
        self,
        frame: np.ndarray,
        result: HandDetectionResult,
    ) -> np.ndarray:
        """Draw landmarks and handedness for all detected hands."""
        for hand in result.hands:
            self.draw_landmarks(frame, hand)
            self.draw_bounding_box(
                frame,
                hand,
                label=f"{hand.handedness}",
                confidence=hand.confidence,
            )
        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        num_hands: int,
        title: str = "Hand Gesture Detection & Recognition",
    ) -> np.ndarray:
        """Render a memory-optimized HUD banner across the top of the frame."""
        h, w = frame.shape[:2]
        banner_h = 52

        # Sub-region slice alpha blending (zero full-frame memory copies)
        top_slice = frame[0:banner_h, 0:w]
        dark_banner = np.zeros_like(top_slice)
        dark_banner[:] = (18, 18, 18)
        cv2.addWeighted(dark_banner, 0.85, top_slice, 0.15, 0, top_slice)

        # Title
        cv2.putText(
            frame,
            title,
            (16, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Right-aligned metrics
        metrics_str = f"FPS: {fps:4.1f}  |  Hands: {num_hands}"
        (mw, _), _ = cv2.getTextSize(metrics_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(
            frame,
            metrics_str,
            (w - mw - 16, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )

        return frame

    def draw_diagnostics_panel(
        self,
        frame: np.ndarray,
        stage_latencies: Dict[str, float],
    ) -> np.ndarray:
        """Render real-time diagnostic latency panel when debug mode is enabled."""
        h, w = frame.shape[:2]
        panel_w = 280
        panel_h = len(stage_latencies) * 22 + 40
        x1 = w - panel_w - 14
        y1 = 65

        # Sub-region blending
        sub_panel = frame[y1 : y1 + panel_h, x1 : x1 + panel_w]
        overlay = np.zeros_like(sub_panel)
        overlay[:] = (20, 20, 20)
        cv2.addWeighted(overlay, 0.85, sub_panel, 0.15, 0, sub_panel)

        cv2.putText(
            frame,
            "PIPELINE LATENCY PROFILER",
            (x1 + 10, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for i, (stage, ms) in enumerate(stage_latencies.items()):
            row_y = y1 + 46 + i * 22
            text = f"{stage:<16} : {ms:>5.2f} ms"
            cv2.putText(
                frame,
                text,
                (x1 + 10, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return frame
