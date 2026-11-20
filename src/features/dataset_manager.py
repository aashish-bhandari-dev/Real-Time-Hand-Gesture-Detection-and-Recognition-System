"""Dataset management utility for saving, loading, and validating hand gesture datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DEFAULT_DATASET_PATH, gesture_config
from src.features.landmark_extractor import LandmarkFeatureExtractor

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages reading, writing, updating, and validating the hand gesture feature dataset."""

    METADATA_COLUMNS = ["gesture_name", "gesture_id", "handedness", "timestamp"]

    def __init__(
        self,
        dataset_path: Path = DEFAULT_DATASET_PATH,
        feature_extractor: Optional[LandmarkFeatureExtractor] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.extractor = feature_extractor or LandmarkFeatureExtractor()
        self.feature_columns = self.extractor.feature_names
        self.all_columns = self.feature_columns + self.METADATA_COLUMNS

        # Ensure parent directory exists
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from CSV. If not existing, returns an empty DataFrame with proper schema."""
        if not self.dataset_path.exists():
            return pd.DataFrame(columns=self.all_columns)

        try:
            df = pd.read_csv(self.dataset_path)
            self.validate_schema(df)
            return df
        except Exception as e:
            logger.error(f"Error loading dataset from {self.dataset_path}: {e}")
            raise

    def get_class_distribution(self, df: Optional[pd.DataFrame] = None) -> Dict[str, int]:
        """Return a mapping of gesture names to the count of existing samples."""
        if df is None:
            df = self.load_dataset()

        distribution = {name: 0 for name in gesture_config.GESTURES}
        if df.empty or "gesture_name" not in df.columns:
            return distribution

        counts = df["gesture_name"].value_counts().to_dict()
        for name, count in counts.items():
            distribution[name] = int(count)

        return distribution

    def append_samples(self, new_records: List[Dict[str, object]]) -> int:
        """Append new sample dictionaries to the CSV file.

        Args:
            new_records: List of dicts, each containing 88 feature values plus metadata keys.

        Returns:
            Total count of samples added.
        """
        if not new_records:
            return 0

        new_df = pd.DataFrame(new_records)
        self.validate_schema(new_df)

        if not self.dataset_path.exists():
            new_df.to_csv(self.dataset_path, index=False)
        else:
            new_df.to_csv(self.dataset_path, mode="a", header=False, index=False)

        logger.info(f"Saved {len(new_records)} samples to {self.dataset_path}")
        return len(new_records)

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Verify that the dataframe matches the expected 88 features and metadata columns."""
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Dataset is missing {len(missing_features)} expected feature columns, "
                f"e.g.: {list(missing_features)[:5]}"
            )

        missing_meta = set(self.METADATA_COLUMNS) - set(df.columns)
        if missing_meta:
            raise ValueError(f"Dataset is missing metadata columns: {missing_meta}")

        return True

    def create_record(
        self,
        features: np.ndarray,
        gesture_name: str,
        gesture_id: int,
        handedness: str,
        timestamp: float,
    ) -> Dict[str, object]:
        """Construct a standardized single-row dictionary ready for DataFrame / CSV insertion."""
        if features.shape != (self.extractor.FEATURE_DIMENSION,):
            raise ValueError(
                f"Feature vector shape {features.shape} != ({self.extractor.FEATURE_DIMENSION},)"
            )

        record: Dict[str, object] = {
            col: float(val) for col, val in zip(self.feature_columns, features)
        }
        record["gesture_name"] = gesture_name
        record["gesture_id"] = int(gesture_id)
        record["handedness"] = handedness
        record["timestamp"] = timestamp

        return record

    def create_synthetic_bootstrap_dataset(self, samples_per_class: int = 50) -> pd.DataFrame:
        """Generate a physiologically realistic synthetic dataset across all 10 gestures.

        Useful for bootstrapping testing and running training pipelines without requiring
        manual webcam collection.
        """
        import time

        records = []
        rng = np.random.default_rng(seed=42)

        for g_idx, g_name in enumerate(gesture_config.GESTURES):
            for i in range(samples_per_class):
                # Base canonical hand landmarks (21, 3)
                coords = np.zeros((21, 3), dtype=np.float32)
                coords[0] = [0.5, 0.8, 0.0]  # Wrist

                # Base angles for each gesture
                # g_idx: 0: Open Palm, 1: Closed Fist, 2: Thumbs Up, 3: Peace, 4: One Finger,
                # 5: Three, 6: Four, 7: Five, 8: OK, 9: Stop
                for f_idx in range(5):
                    base_x = 0.38 + f_idx * 0.06
                    base_y = 0.60
                    # Determine whether finger is extended or curled
                    if g_idx == 0 or g_idx == 7 or g_idx == 9:  # Open palm, five, stop
                        is_extended = True
                    elif g_idx == 1:  # Closed fist
                        is_extended = False
                    elif g_idx == 2:  # Thumbs up
                        is_extended = (f_idx == 0)
                    elif g_idx == 3:  # Peace
                        is_extended = (f_idx in (1, 2))
                    elif g_idx == 4:  # One Finger
                        is_extended = (f_idx == 1)
                    elif g_idx == 5:  # Three Fingers
                        is_extended = (f_idx in (0, 1, 2))
                    elif g_idx == 6:  # Four Fingers
                        is_extended = (f_idx in (1, 2, 3, 4))
                    elif g_idx == 8:  # OK Sign
                        # Thumb tip touches index tip, other 3 extended
                        is_extended = (f_idx in (2, 3, 4))
                    else:
                        is_extended = True

                    for j in range(4):
                        lm_idx = 1 + f_idx * 4 + j
                        step_y = -0.08 if is_extended else 0.03
                        noise = rng.normal(0, 0.008, 3)
                        coords[lm_idx] = [
                            base_x + noise[0],
                            base_y + j * step_y + noise[1],
                            noise[2],
                        ]

                # Special tweak for OK sign: bring thumb tip and index tip close
                if g_idx == 8:
                    contact_pt = (coords[4] + coords[8]) / 2.0
                    coords[4] = contact_pt + rng.normal(0, 0.003, 3)
                    coords[8] = contact_pt + rng.normal(0, 0.003, 3)

                features = self.extractor.extract_features(coords)
                handedness = "Right" if i % 2 == 0 else "Left"
                record = self.create_record(
                    features=features,
                    gesture_name=g_name,
                    gesture_id=g_idx,
                    handedness=handedness,
                    timestamp=time.time(),
                )
                records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(self.dataset_path, index=False)
        logger.info(f"Generated synthetic bootstrap dataset with {len(df)} samples at {self.dataset_path}")
        return df
