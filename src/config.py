"""Central configuration module for the Hand Gesture Detection and Recognition System.

Maintains camera, MediaPipe Task, dataset, feature extraction, and model inference settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Base directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# MediaPipe Hand Landmarker model asset
MODEL_TASK_NAME = "hand_landmarker.task"
MODEL_TASK_PATH = ASSETS_DIR / MODEL_TASK_NAME
MODEL_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Default ML Model and Dataset paths
DEFAULT_DATASET_PATH = PROCESSED_DATA_DIR / "gestures_dataset.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "gesture_classifier.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"


@dataclass(frozen=True)
class GestureConfig:
    """Predefined gesture labels and mappings."""

    GESTURES: List[str] = field(
        default_factory=lambda: [
            "Open Palm",
            "Closed Fist",
            "Thumbs Up",
            "Peace Sign",
            "One Finger",
            "Three Fingers",
            "Four Fingers",
            "Five Fingers",
            "OK Sign",
            "Stop Gesture",
        ]
    )

    @property
    def label_to_id(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.GESTURES)}

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {idx: name for idx, name in enumerate(self.GESTURES)}


@dataclass
class CameraConfig:
    """Webcam video capture configuration."""

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    mirror: bool = True


@dataclass
class HandDetectorConfig:
    """MediaPipe Tasks Hand Landmarker configuration."""

    model_path: Path = MODEL_TASK_PATH
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class ClassifierConfig:
    """Classification and inference post-processing configuration."""

    model_path: Path = DEFAULT_MODEL_PATH
    confidence_threshold: float = 0.55
    smoothing_window_size: int = 5
    unknown_label: str = "Unknown Gesture"
    no_hand_label: str = "No Hand Detected"


@dataclass
class TrainingConfig:
    """Random Forest training and evaluation settings."""

    dataset_path: Path = DEFAULT_DATASET_PATH
    output_model_path: Path = DEFAULT_MODEL_PATH
    metadata_path: Path = MODEL_METADATA_PATH
    test_size: float = 0.20
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int | None = 20
    min_samples_split: int = 2


# Shared default instances
gesture_config = GestureConfig()
camera_config = CameraConfig()
detector_config = HandDetectorConfig()
classifier_config = ClassifierConfig()
training_config = TrainingConfig()
