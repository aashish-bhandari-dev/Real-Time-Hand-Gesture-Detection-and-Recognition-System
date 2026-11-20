"""OpenCV webcam video capture wrapper with robust error handling and mirroring."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import CameraConfig, camera_config

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Raised when the camera fails to open or cannot deliver frames."""

    pass


class CameraManager:
    """Manages OpenCV VideoCapture lifecycle, mirroring, and resolution configuration."""

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self.config = config or camera_config
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False

    def open(self) -> None:
        """Open the camera and apply resolution and capture properties."""
        if self._is_opened:
            return

        logger.info(f"Opening camera at index {self.config.camera_index}...")
        self._cap = cv2.VideoCapture(self.config.camera_index)

        if not self._cap.isOpened():
            available = self.find_available_cameras()
            raise CameraError(
                f"Could not open camera at index {self.config.camera_index}.\n"
                f"Available detected camera indices: {available if available else 'None found'}.\n"
                f"On macOS, verify that Camera permissions are granted in System Settings > Privacy & Security > Camera."
            )

        # Configure camera resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        self._is_opened = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera, applying horizontal flip if mirroring is enabled.

        Returns:
            Tuple of (success_flag, bgr_frame).
        """
        if not self._is_opened or self._cap is None:
            raise CameraError("Camera is not opened. Call open() first or use context manager.")

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None

        if self.config.mirror:
            frame = cv2.flip(frame, 1)

        return True, frame

    def get_actual_dimensions(self) -> Tuple[int, int]:
        """Return the actual width and height reported by the capture device."""
        if not self._is_opened or self._cap is None:
            return (self.config.frame_width, self.config.frame_height)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def release(self) -> None:
        """Release the capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    def __enter__(self) -> CameraManager:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @staticmethod
    def find_available_cameras(max_tested: int = 5) -> list[int]:
        """Probe potential camera indices to check which devices are available."""
        available = []
        for idx in range(max_tested):
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                available.append(idx)
                test_cap.release()
        return available
