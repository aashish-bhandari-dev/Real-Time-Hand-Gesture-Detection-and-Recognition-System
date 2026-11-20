"""MediaPipe Tasks Hand Landmarker detection module."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import HandDetectorConfig, detector_config


@dataclass
class LandmarkPoint:
    """Normalized 3D landmark point (x, y in [0, 1], z represents depth)."""

    x: float
    y: float
    z: float


@dataclass
class DetectedHand:
    """Information and landmarks for a single detected hand."""

    handedness: str  # "Left" or "Right"
    confidence: float
    landmarks: List[LandmarkPoint]
    world_landmarks: Optional[List[LandmarkPoint]] = None

    def to_numpy(self) -> np.ndarray:
        """Convert 21 landmarks to a (21, 3) numpy array."""
        return np.array([[lm.x, lm.y, lm.z] for lm in self.landmarks], dtype=np.float32)


@dataclass
class HandDetectionResult:
    """Detection results across all hands in a frame."""

    hands: List[DetectedHand]
    timestamp_ms: int


class HandDetector:
    """Detects hands and extracts 21 3D landmarks using MediaPipe Hand Landmarker Tasks API."""

    def __init__(self, config: Optional[HandDetectorConfig] = None) -> None:
        self.config = config or detector_config
        self._validate_model_path(self.config.model_path)

        self._last_timestamp_ms = 0
        self._landmarker: Optional[vision.HandLandmarker] = None
        self._init_landmarker()

    def _validate_model_path(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"MediaPipe Hand Landmarker model file not found at: {path}\n"
                f"Run 'python scripts/download_model.py' to download the required asset."
            )

    def _init_landmarker(self) -> None:
        """Initialize the MediaPipe Hand Landmarker in VIDEO running mode."""
        base_options = python.BaseOptions(model_asset_path=str(self.config.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_hand_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(
        self,
        bgr_frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> HandDetectionResult:
        """Detect hands and landmarks in an OpenCV BGR frame.

        Args:
            bgr_frame: OpenCV BGR image array.
            timestamp_ms: Monotonically increasing frame timestamp in milliseconds.
                          If None, wall-clock time is used with monotonicity guarantee.

        Returns:
            HandDetectionResult containing detected hands and handedness.
        """
        if self._landmarker is None:
            raise RuntimeError("HandLandmarker is closed or not initialized.")

        # Ensure strictly increasing timestamp for VIDEO mode
        if timestamp_ms is None:
            current_ms = int(time.time() * 1000)
            if current_ms <= self._last_timestamp_ms:
                current_ms = self._last_timestamp_ms + 1
            timestamp_ms = current_ms
        elif timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1

        self._last_timestamp_ms = timestamp_ms

        # MediaPipe expects RGB format
        rgb_frame = cv2_to_rgb(bgr_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        raw_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        detected_hands: List[DetectedHand] = []
        if raw_result.hand_landmarks:
            num_detected = len(raw_result.hand_landmarks)
            for idx in range(num_detected):
                landmarks = [
                    LandmarkPoint(x=lm.x, y=lm.y, z=lm.z)
                    for lm in raw_result.hand_landmarks[idx]
                ]

                # Parse handedness metadata if present
                handedness_label = "Unknown"
                handedness_score = 1.0
                if raw_result.handedness and idx < len(raw_result.handedness):
                    category = raw_result.handedness[idx][0]
                    handedness_label = category.category_name
                    handedness_score = float(category.score)

                # Parse optional real-world 3D coordinates in meters
                world_lms: Optional[List[LandmarkPoint]] = None
                if raw_result.hand_world_landmarks and idx < len(raw_result.hand_world_landmarks):
                    world_lms = [
                        LandmarkPoint(x=wlm.x, y=wlm.y, z=wlm.z)
                        for wlm in raw_result.hand_world_landmarks[idx]
                    ]

                detected_hands.append(
                    DetectedHand(
                        handedness=handedness_label,
                        confidence=handedness_score,
                        landmarks=landmarks,
                        world_landmarks=world_lms,
                    )
                )

        return HandDetectionResult(hands=detected_hands, timestamp_ms=timestamp_ms)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def cv2_to_rgb(bgr_image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image array to RGB without re-allocating if contiguous."""
    import cv2

    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
