"""Feature engineering and normalization module for hand landmark coordinates.

Extracts translation-invariant, scale-invariant, and geometric relationship features
from 21 hand landmarks, producing a fixed-size 88-dimensional numerical feature vector.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np

from src.detection.hand_detector import DetectedHand, LandmarkPoint

# Finger joint landmark triplets for angle calculations (A, B, C) where B is the vertex
JOINT_TRIPLETS: List[Tuple[str, int, int, int]] = [
    # Base flexion angles (Wrist -> Base/MCP -> Middle/PIP)
    ("thumb_base_angle", 0, 1, 2),
    ("index_mcp_angle", 0, 5, 6),
    ("middle_mcp_angle", 0, 9, 10),
    ("ring_mcp_angle", 0, 13, 14),
    ("pinky_mcp_angle", 0, 17, 18),
    # Mid-joint flexion angles (MCP -> PIP -> DIP/TIP)
    ("thumb_ip_angle", 2, 3, 4),
    ("index_pip_angle", 5, 6, 7),
    ("middle_pip_angle", 9, 10, 11),
    ("ring_pip_angle", 13, 14, 15),
    ("pinky_pip_angle", 17, 18, 19),
]

# Fingertip landmark indices
FINGERTIP_INDICES: List[Tuple[str, int]] = [
    ("thumb_tip", 4),
    ("index_tip", 8),
    ("middle_tip", 12),
    ("ring_tip", 16),
    ("pinky_tip", 20),
]

# Inter-fingertip distance pairs
FINGERTIP_PAIRS: List[Tuple[str, int, int]] = [
    ("dist_thumb_index", 4, 8),
    ("dist_index_middle", 8, 12),
    ("dist_middle_ring", 12, 16),
    ("dist_ring_pinky", 16, 20),
    ("dist_thumb_pinky", 4, 20),
]

# (Fingertip index, PIP joint index) pairs to measure finger extension ratio
FINGER_PIP_PAIRS: List[Tuple[str, int, int]] = [
    ("thumb_ext", 4, 2),
    ("index_ext", 8, 6),
    ("middle_ext", 12, 10),
    ("ring_ext", 16, 14),
    ("pinky_ext", 20, 18),
]


class LandmarkFeatureExtractor:
    """Extracts a normalized, invariant 88-dimensional feature vector from 21 hand landmarks."""

    FEATURE_DIMENSION: int = 88

    def __init__(self) -> None:
        self._feature_names = self._generate_feature_names()

    @property
    def feature_names(self) -> List[str]:
        """Return the ordered list of 88 human-readable feature column names."""
        return self._feature_names

    def _generate_feature_names(self) -> List[str]:
        names: List[str] = []
        # 1. 21 normalized landmarks (x, y, z) = 63 features
        for i in range(21):
            names.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])

        # 2. Wrist to fingertip distances = 5 features
        for tip_name, _ in FINGERTIP_INDICES:
            names.append(f"wrist_to_{tip_name}_dist")

        # 3. Inter-fingertip distances = 5 features
        for pair_name, _, _ in FINGERTIP_PAIRS:
            names.append(pair_name)

        # 4. Joint cosine angles = 10 features
        for angle_name, _, _, _ in JOINT_TRIPLETS:
            names.append(f"cos_{angle_name}")

        # 5. Finger extension ratios = 5 features
        for ext_name, _, _ in FINGER_PIP_PAIRS:
            names.append(f"ratio_{ext_name}")

        assert len(names) == self.FEATURE_DIMENSION, (
            f"Feature names count ({len(names)}) does not match FEATURE_DIMENSION ({self.FEATURE_DIMENSION})"
        )
        return names

    def extract_features(
        self,
        hand_input: Union[DetectedHand, Sequence[LandmarkPoint], np.ndarray],
    ) -> np.ndarray:
        """Extract a fixed-size 88-dimensional float32 feature vector from hand landmarks.

        Args:
            hand_input: DetectedHand object, list of 21 LandmarkPoint objects,
                        or a (21, 3) numpy array.

        Returns:
            np.ndarray: 1D array of shape (88,) with float32 values.

        Raises:
            ValueError: If input landmarks are malformed, contain NaN/Inf, or have wrong shape.
        """
        coords = self._to_coordinate_array(hand_input)
        self._validate_coordinates(coords)

        # 1. Translation Invariance: center coordinates on the wrist (landmark 0)
        wrist = coords[0]
        centered = coords - wrist

        # 2. Scale Invariance: normalize by palm size (wrist to middle MCP joint distance)
        # Landmark 9 is MIDDLE_FINGER_MCP
        palm_size = float(np.linalg.norm(centered[9]))
        if palm_size < 1e-6:
            # Fallback to max span if middle MCP is collapsed
            palm_size = float(np.max(np.linalg.norm(centered, axis=1)))
            if palm_size < 1e-6:
                palm_size = 1.0  # Prevent zero division on degenerate inputs

        normalized_coords = centered / palm_size

        # 3. Flattened normalized coordinates (21 * 3 = 63 features)
        norm_coords_flat = normalized_coords.flatten()

        # 4. Normalized distances from wrist (origin) to all 5 fingertips (5 features)
        wrist_to_tips = [
            float(np.linalg.norm(normalized_coords[tip_idx]))
            for _, tip_idx in FINGERTIP_INDICES
        ]

        # 5. Normalized distances between fingertips (5 features)
        inter_tip_dists = [
            float(np.linalg.norm(normalized_coords[idx1] - normalized_coords[idx2]))
            for _, idx1, idx2 in FINGERTIP_PAIRS
        ]

        # 6. Joint cosine angles (10 features)
        joint_angles = [
            self._compute_joint_cosine(normalized_coords, idx_a, idx_b, idx_c)
            for _, idx_a, idx_b, idx_c in JOINT_TRIPLETS
        ]

        # 7. Finger extension ratios: ||tip - wrist|| / (||pip - wrist|| + eps) (5 features)
        extension_ratios = []
        for _, tip_idx, pip_idx in FINGER_PIP_PAIRS:
            tip_dist = float(np.linalg.norm(normalized_coords[tip_idx]))
            pip_dist = float(np.linalg.norm(normalized_coords[pip_idx]))
            ratio = tip_dist / (pip_dist + 1e-5)
            extension_ratios.append(ratio)

        # Concatenate all components into a single 1D vector
        features = np.concatenate(
            [
                norm_coords_flat,
                np.array(wrist_to_tips, dtype=np.float32),
                np.array(inter_tip_dists, dtype=np.float32),
                np.array(joint_angles, dtype=np.float32),
                np.array(extension_ratios, dtype=np.float32),
            ],
            dtype=np.float32,
        )

        assert features.shape == (self.FEATURE_DIMENSION,), (
            f"Extracted feature shape {features.shape} != ({self.FEATURE_DIMENSION},)"
        )

        return features

    @staticmethod
    def _compute_joint_cosine(
        coords: np.ndarray,
        idx_a: int,
        idx_b: int,
        idx_c: int,
    ) -> float:
        """Compute cosine of the angle ABC at vertex B using vector dot product."""
        vec_ba = coords[idx_a] - coords[idx_b]
        vec_bc = coords[idx_c] - coords[idx_b]

        norm_ba = np.linalg.norm(vec_ba)
        norm_bc = np.linalg.norm(vec_bc)

        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0

        dot_product = np.dot(vec_ba, vec_bc)
        cosine = dot_product / (norm_ba * norm_bc)
        # Clip numerical overflow outside [-1.0, 1.0]
        return float(np.clip(cosine, -1.0, 1.0))

    @staticmethod
    def _to_coordinate_array(
        hand_input: Union[DetectedHand, Sequence[LandmarkPoint], np.ndarray],
    ) -> np.ndarray:
        """Convert varied hand landmark input types into a standardized (21, 3) float32 numpy array."""
        if isinstance(hand_input, DetectedHand):
            return hand_input.to_numpy()
        elif isinstance(hand_input, np.ndarray):
            if hand_input.shape != (21, 3):
                raise ValueError(
                    f"Expected landmark numpy array of shape (21, 3), got {hand_input.shape}"
                )
            return hand_input.astype(np.float32)
        elif isinstance(hand_input, Sequence):
            if len(hand_input) != 21:
                raise ValueError(
                    f"Expected sequence of 21 landmarks, got {len(hand_input)}"
                )
            return np.array([[lm.x, lm.y, lm.z] for lm in hand_input], dtype=np.float32)
        else:
            raise TypeError(
                f"Unsupported hand input type: {type(hand_input)}. "
                "Expected DetectedHand, Sequence[LandmarkPoint], or np.ndarray."
            )

    @staticmethod
    def _validate_coordinates(coords: np.ndarray) -> None:
        """Ensure coordinate array contains valid finite numbers."""
        if not np.all(np.isfinite(coords)):
            raise ValueError("Landmark coordinates contain NaN or Inf values.")
