"""Generate a bootstrap synthetic dataset for all 10 hand gestures."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_DATASET_PATH
from src.features.dataset_manager import DatasetManager


def main() -> None:
    print("Generating synthetic bootstrap dataset for quick testing and training...")
    manager = DatasetManager(dataset_path=DEFAULT_DATASET_PATH)
    df = manager.create_synthetic_bootstrap_dataset(samples_per_class=100)
    print(f"Successfully generated {len(df)} samples across {df['gesture_name'].nunique()} classes.")
    print(f"Dataset saved to: {DEFAULT_DATASET_PATH}")
    print("\nClass distribution:")
    print(df["gesture_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
