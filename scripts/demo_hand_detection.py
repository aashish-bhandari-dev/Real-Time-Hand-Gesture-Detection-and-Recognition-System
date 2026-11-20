"""Live demonstration script for MediaPipe Hand Detection and tracking via webcam."""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import camera_config, detector_config
from src.detection.camera import CameraError, CameraManager
from src.detection.hand_detector import HandDetector
from src.visualization.drawer import Visualizer


def run_detection_demo() -> None:
    """Run real-time webcam hand detection using MediaPipe Tasks API."""
    print("Initializing MediaPipe Hand Landmarker...")
    try:
        detector = HandDetector(detector_config)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print("Opening webcam...")
    camera = CameraManager(camera_config)
    try:
        camera.open()
    except CameraError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        detector.close()
        sys.exit(1)

    visualizer = Visualizer()

    fps_buffer: deque[float] = deque(maxlen=30)
    prev_time = time.perf_counter()

    window_name = "Hand Gesture System - Live Detection (Phase 2)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    actual_w, actual_h = camera.get_actual_dimensions()
    print(f"Camera stream active: {actual_w}x{actual_h}")
    print("Controls: Press 'q' or 'ESC' to exit.")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("Failed to grab camera frame. Exiting loop.")
                break

            # Calculate FPS
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_buffer.append(1.0 / dt)
            avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0.0

            # Run hand detection
            detection_result = detector.detect(frame)

            # Draw landmarks and HUD
            visualizer.draw_detection_result(frame, detection_result)
            visualizer.draw_hud(frame, fps=avg_fps, num_hands=len(detection_result.hands))

            # Display frame
            cv2.imshow(window_name, frame)

            # Key controls: ESC (27) or 'q' (ord('q'))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                print("Exit requested by user.")
                break

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        print("Camera and detector resources successfully released.")


if __name__ == "__main__":
    run_detection_demo()
