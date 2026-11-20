"""Main entry point for Real-Time Hand Gesture Detection and Recognition System."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from src.classification.gesture_classifier import GestureClassifier
from src.classification.prediction_smoother import PredictionSmoother
from src.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    MODEL_METADATA_PATH,
    PROJECT_ROOT,
    camera_config,
    classifier_config,
    detector_config,
)
from src.detection.camera import CameraError, CameraManager
from src.detection.hand_detector import HandDetector
from src.features.dataset_manager import DatasetManager
from src.features.landmark_extractor import LandmarkFeatureExtractor
from src.visualization.drawer import Visualizer
from src.visualization.profiler import PipelineProfiler


class GestureRecognitionApp:
    """Orchestrates webcam capture, hand landmark tracking, feature extraction, and real-time inference."""

    def __init__(self) -> None:
        self.camera = CameraManager(camera_config)
        self.detector = HandDetector(detector_config)
        self.extractor = LandmarkFeatureExtractor()
        self.visualizer = Visualizer()
        self.profiler = PipelineProfiler(window_size=30)
        self.smoother = PredictionSmoother(
            window_size=classifier_config.smoothing_window_size,
            unknown_label=classifier_config.unknown_label,
        )

        self.classifier = self._init_classifier()
        self.draw_skeleton = True
        self.show_diagnostics = False

    def _init_classifier(self) -> GestureClassifier:
        """Load trained model or auto-train baseline checkpoint if missing."""
        if not DEFAULT_MODEL_PATH.exists():
            print("\n[INFO] Trained classifier checkpoint not found.")
            print("Auto-training baseline model using synthetic bootstrap dataset...")
            dm = DatasetManager(dataset_path=DEFAULT_DATASET_PATH)
            if not DEFAULT_DATASET_PATH.exists():
                dm.create_synthetic_bootstrap_dataset(samples_per_class=100)

            from scripts.train_model import train_gesture_classifier

            train_gesture_classifier(
                dataset_path=DEFAULT_DATASET_PATH,
                model_output_path=DEFAULT_MODEL_PATH,
                metadata_output_path=MODEL_METADATA_PATH,
            )

        print(f"Loading trained classifier from {DEFAULT_MODEL_PATH}...")
        return GestureClassifier.load(
            model_path=DEFAULT_MODEL_PATH,
            metadata_path=MODEL_METADATA_PATH,
            config=classifier_config,
        )

    def run(self) -> None:
        """Run the real-time webcam inference loop."""
        try:
            self.camera.open()
        except CameraError as e:
            print(f"\n[ERROR] {e}", file=sys.stderr)
            self.detector.close()
            sys.exit(1)

        window_name = "Real-Time Hand Gesture Recognition System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "=" * 65)
        print(" REAL-TIME HAND GESTURE RECOGNITION ACTIVE")
        print("=" * 65)
        print("Controls:")
        print("  q / ESC   : Exit application")
        print("  s         : Save screenshot")
        print("  t         : Toggle landmark skeleton overlay")
        print("  d         : Toggle latency diagnostics HUD")
        print("=" * 65 + "\n")

        try:
            while True:
                self.profiler.tick_frame()

                # 1. Camera Frame Read
                self.profiler.start_stage("camera_read")
                ret, frame = self.camera.read()
                self.profiler.end_stage()

                if not ret or frame is None:
                    print("Failed to grab camera frame. Exiting.")
                    break

                # 2. Hand Landmark Detection
                self.profiler.start_stage("hand_detect")
                detection_result = self.detector.detect(frame)
                self.profiler.end_stage()

                num_hands = len(detection_result.hands)
                recognized_gestures = []

                if num_hands == 0:
                    self.smoother.reset_all()
                else:
                    for i, hand in enumerate(detection_result.hands):
                        # 3. Feature Extraction
                        self.profiler.start_stage("feature_extract")
                        try:
                            features = self.extractor.extract_features(hand)
                        except Exception as e:
                            print(f"Feature error: {e}", file=sys.stderr)
                            self.profiler.end_stage()
                            continue
                        self.profiler.end_stage()

                        # 4. ML Inference & Smoothing
                        self.profiler.start_stage("ml_inference")
                        raw_label, raw_confidence = self.classifier.predict(features, hand=hand)
                        hand_key = f"{hand.handedness}_{i}"
                        smoothed = self.smoother.update(
                            hand_id=hand_key,
                            label=raw_label,
                            confidence=raw_confidence,
                        )
                        self.profiler.end_stage()

                        recognized_gestures.append(f"{hand.handedness}: {smoothed.label}")

                        # 5. Overlay Rendering
                        self.profiler.start_stage("rendering")
                        if self.draw_skeleton:
                            self.visualizer.draw_landmarks(frame, hand)

                        box_color = (
                            (50, 205, 50)
                            if smoothed.label != classifier_config.unknown_label
                            else (0, 165, 255)
                        )
                        self.visualizer.draw_bounding_box(
                            frame=frame,
                            hand=hand,
                            label=f"{smoothed.label}",
                            confidence=smoothed.confidence,
                            is_stable=smoothed.is_stable,
                            box_color=box_color,
                        )
                        self.profiler.end_stage()

                # Render top HUD banner
                self.profiler.start_stage("rendering")
                self._render_main_hud(
                    frame=frame,
                    fps=self.profiler.fps,
                    num_hands=num_hands,
                    recognized_gestures=recognized_gestures,
                )

                if self.show_diagnostics:
                    self.visualizer.draw_diagnostics_panel(
                        frame=frame,
                        stage_latencies={k: v.average_ms for k, v in self.profiler.stages.items()},
                    )
                self.profiler.end_stage()

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                elif key in (ord("s"), ord("S")):
                    self._save_screenshot(frame)
                elif key in (ord("t"), ord("T")):
                    self.draw_skeleton = not self.draw_skeleton
                elif key in (ord("d"), ord("D")):
                    self.show_diagnostics = not self.show_diagnostics

        finally:
            self.camera.release()
            self.detector.close()
            cv2.destroyAllWindows()
            print("Application resources released cleanly.")

    def _render_main_hud(
        self,
        frame: np.ndarray,
        fps: float,
        num_hands: int,
        recognized_gestures: list[str],
    ) -> None:
        """Render informative top HUD overlay."""
        h, w = frame.shape[:2]
        self.visualizer.draw_hud(frame, fps=fps, num_hands=num_hands)

        # Active gestures display in HUD
        if num_hands == 0:
            status_text = "No Hand Detected"
            status_color = (160, 160, 160)
        else:
            status_text = ", ".join(recognized_gestures)
            status_color = (0, 255, 200)

        cv2.putText(
            frame,
            f"Status: {status_text}",
            (16, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            1,
            cv2.LINE_AA,
        )

    def _save_screenshot(self, frame: np.ndarray) -> None:
        """Save high-resolution screenshot to disk."""
        screenshots_dir = PROJECT_ROOT / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = screenshots_dir / f"gesture_capture_{timestamp}.png"
        cv2.imwrite(str(filename), frame)
        print(f"Screenshot saved to: {filename}")


if __name__ == "__main__":
    app = GestureRecognitionApp()
    app.run()
