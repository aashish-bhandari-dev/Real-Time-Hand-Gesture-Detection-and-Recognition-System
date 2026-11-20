"""Interactive custom dataset collection script for Hand Gesture Recognition.

Allows recording hand landmark feature samples directly from the webcam for all 10 gestures,
with rate-limiting to prevent near-duplicate adjacent frames and real-time class distribution counters.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import camera_config, detector_config, gesture_config
from src.detection.camera import CameraError, CameraManager
from src.detection.hand_detector import HandDetector
from src.features.dataset_manager import DatasetManager
from src.features.landmark_extractor import LandmarkFeatureExtractor
from src.visualization.drawer import Visualizer


class DatasetCollectorApp:
    """Manages the interactive webcam session for gesture data collection."""

    # Minimum seconds between auto-collected samples to prevent near-duplicate frames
    MIN_SAMPLE_INTERVAL_SEC = 0.12  # ~8 samples/second maximum
    # Minimum feature vector Euclidean distance to consider consecutive frames distinct
    MIN_FEATURE_DIFFERENCE = 0.05

    def __init__(self) -> None:
        self.camera = CameraManager(camera_config)
        self.detector = HandDetector(detector_config)
        self.extractor = LandmarkFeatureExtractor()
        self.dataset_manager = DatasetManager()
        self.visualizer = Visualizer()

        # State tracking
        self.current_gesture_idx = 0
        self.is_recording = False
        self.last_sample_time = 0.0
        self.last_feature_vector: Optional[np.ndarray] = None

        # Session buffer: samples collected in the current run before writing to disk
        self.session_records: List[Dict[str, object]] = []
        self.saved_counts = self.dataset_manager.get_class_distribution()

    @property
    def current_gesture_name(self) -> str:
        return gesture_config.GESTURES[self.current_gesture_idx]

    def run(self) -> None:
        """Run the main interactive collection loop."""
        try:
            self.camera.open()
        except CameraError as e:
            print(f"\n[ERROR] {e}", file=sys.stderr)
            self.detector.close()
            sys.exit(1)

        window_name = "Hand Gesture Dataset Collector (Phase 4)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        fps_buffer: deque[float] = deque(maxlen=20)
        prev_time = time.perf_counter()

        print("\n" + "=" * 60)
        print(" HAND GESTURE DATASET COLLECTOR")
        print("=" * 60)
        print("Controls:")
        print("  SPACE       : Toggle recording ON / OFF")
        print("  0 - 9       : Select gesture 0 to 9")
        print("  n / p       : Next / Previous gesture")
        print("  r           : Single-shot capture 1 sample")
        print("  s           : Flush session buffer to CSV")
        print("  q / ESC     : Save and Exit")
        print("=" * 60 + "\n")

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    print("Failed to grab camera frame. Exiting.")
                    break

                # FPS calculation
                curr_time = time.perf_counter()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0:
                    fps_buffer.append(1.0 / dt)
                avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0.0

                # Run Hand Detection
                detection_result = self.detector.detect(frame)

                # Process hand features & recording
                hand_detected = len(detection_result.hands) > 0
                sample_captured = False

                if hand_detected:
                    primary_hand = detection_result.hands[0]
                    # Draw landmarks on frame
                    self.visualizer.draw_landmarks(frame, primary_hand)
                    self.visualizer.draw_bounding_box(
                        frame,
                        primary_hand,
                        label=f"{self.current_gesture_name} ({primary_hand.handedness})",
                    )

                    try:
                        features = self.extractor.extract_features(primary_hand)
                        if self.is_recording:
                            now = time.time()
                            if self._should_capture_sample(features, now):
                                self._record_sample(features, primary_hand.handedness, now)
                                sample_captured = True
                                self.last_sample_time = now
                                self.last_feature_vector = features.copy()
                    except Exception as e:
                        print(f"Feature extraction error: {e}", file=sys.stderr)

                # Render Collector HUD
                self._render_collector_hud(
                    frame=frame,
                    fps=avg_fps,
                    hand_detected=hand_detected,
                    sample_captured=sample_captured,
                )

                cv2.imshow(window_name, frame)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                elif key == ord(" "):
                    self.is_recording = not self.is_recording
                elif ord("0") <= key <= ord("9"):
                    target_idx = key - ord("0")
                    if target_idx < len(gesture_config.GESTURES):
                        self.current_gesture_idx = target_idx
                elif key in (ord("n"), ord("N")):
                    self.current_gesture_idx = (self.current_gesture_idx + 1) % len(
                        gesture_config.GESTURES
                    )
                elif key in (ord("p"), ord("P")):
                    self.current_gesture_idx = (
                        self.current_gesture_idx - 1 + len(gesture_config.GESTURES)
                    ) % len(gesture_config.GESTURES)
                elif key in (ord("r"), ord("R")):
                    # Single frame record
                    if hand_detected:
                        features = self.extractor.extract_features(detection_result.hands[0])
                        self._record_sample(
                            features, detection_result.hands[0].handedness, time.time()
                        )
                elif key in (ord("s"), ord("S")):
                    self._flush_buffer_to_disk()

        finally:
            self._flush_buffer_to_disk()
            self.camera.release()
            self.detector.close()
            cv2.destroyAllWindows()
            print("Collector exited cleanly. Dataset saved.")

    def _should_capture_sample(self, features: np.ndarray, current_time: float) -> bool:
        """Check time interval and movement distance to prevent duplicate frames."""
        if current_time - self.last_sample_time < self.MIN_SAMPLE_INTERVAL_SEC:
            return False

        if self.last_feature_vector is not None:
            # Measure Euclidean distance between consecutive feature vectors
            dist = float(np.linalg.norm(features - self.last_feature_vector))
            if dist < self.MIN_FEATURE_DIFFERENCE:
                return False  # Hand is completely stationary

        return True

    def _record_sample(self, features: np.ndarray, handedness: str, timestamp: float) -> None:
        """Add record to the active session buffer."""
        record = self.dataset_manager.create_record(
            features=features,
            gesture_name=self.current_gesture_name,
            gesture_id=self.current_gesture_idx,
            handedness=handedness,
            timestamp=timestamp,
        )
        self.session_records.append(record)
        self.saved_counts[self.current_gesture_name] = (
            self.saved_counts.get(self.current_gesture_name, 0) + 1
        )

        # Auto-flush to disk every 50 samples
        if len(self.session_records) >= 50:
            self._flush_buffer_to_disk()

    def _flush_buffer_to_disk(self) -> None:
        """Flush in-memory session records to the CSV file."""
        if not self.session_records:
            return
        count = self.dataset_manager.append_samples(self.session_records)
        print(f"Flushed {count} samples to {self.dataset_manager.dataset_path}")
        self.session_records.clear()

    def _render_collector_hud(
        self,
        frame: np.ndarray,
        fps: float,
        hand_detected: bool,
        sample_captured: bool,
    ) -> None:
        """Render the collection dashboard including recording status and class counts."""
        h, w = frame.shape[:2]

        # Top banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Status & Recording Pill
        if self.is_recording:
            status_text = "● RECORDING"
            status_color = (0, 0, 255)  # Red
        else:
            status_text = "[PAUSED] Press SPACE"
            status_color = (0, 200, 255)  # Orange

        cv2.putText(
            frame,
            f"Active: [{self.current_gesture_idx}] {self.current_gesture_name}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            status_text,
            (w - 240, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # Green flash indicator on frame border when sample is captured
        if sample_captured:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 4)

        # Left panel: Class distribution counter
        panel_w = 260
        panel_h = len(gesture_config.GESTURES) * 24 + 40
        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (10, 65), (10 + panel_w, 65 + panel_h), (25, 25, 25), -1)
        cv2.addWeighted(panel_overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(
            frame,
            "DATASET SAMPLES",
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        for idx, g_name in enumerate(gesture_config.GESTURES):
            count = self.saved_counts.get(g_name, 0)
            is_selected = idx == self.current_gesture_idx

            row_y = 112 + idx * 24
            bullet = ">" if is_selected else " "
            row_color = (0, 255, 255) if is_selected else (200, 200, 200)

            # Abbreviate name if needed
            display_name = (g_name[:12] + "..") if len(g_name) > 14 else g_name
            text = f"{bullet} [{idx}] {display_name:<14} : {count:>4}"
            cv2.putText(
                frame,
                text,
                (20, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                row_color,
                1,
                cv2.LINE_AA,
            )

        # Bottom help bar
        help_bar = frame.copy()
        cv2.rectangle(help_bar, (0, h - 28), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(help_bar, 0.8, frame, 0.2, 0, frame)
        cv2.putText(
            frame,
            "Keys: [SPACE] Record/Pause  |  [0-9] Select Gesture  |  [S] Save  |  [Q/ESC] Exit",
            (16, h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    app = DatasetCollectorApp()
    app.run()
