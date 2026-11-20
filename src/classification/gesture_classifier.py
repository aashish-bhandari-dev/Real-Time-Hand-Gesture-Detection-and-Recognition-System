"""Gesture classification model wrapper using Scikit-Learn RandomForestClassifier, Joblib, and Geometric Fallback."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.classification.rule_based_classifier import RuleBasedGestureClassifier
from src.config import DEFAULT_MODEL_PATH, MODEL_METADATA_PATH, ClassifierConfig, classifier_config, gesture_config
from src.detection.hand_detector import DetectedHand

logger = logging.getLogger(__name__)


class ModelNotLoadedError(Exception):
    """Raised when inference is attempted on an uninitialized or missing model."""

    pass


class GestureClassifier:
    """Encapsulates model training, persistence, and real-time inference with hybrid geometric rules."""

    def __init__(
        self,
        model: Optional[RandomForestClassifier] = None,
        label_names: Optional[List[str]] = None,
        config: Optional[ClassifierConfig] = None,
    ) -> None:
        self.config = config or classifier_config
        self.label_names = label_names or gesture_config.GESTURES
        self.model: Optional[RandomForestClassifier] = model
        self.metadata: Dict[str, Any] = {}
        self.rule_classifier = RuleBasedGestureClassifier()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: Optional[int] = 20,
        random_state: int = 42,
    ) -> RandomForestClassifier:
        """Fit a Random Forest Classifier on training feature matrix and labels."""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.model.fit(X_train, y_train)
        return self.model

    def predict(
        self,
        features: np.ndarray,
        hand: Optional[DetectedHand] = None,
    ) -> Tuple[str, float]:
        """Predict gesture label with hybrid fallback between ML classifier and anatomical rules.

        Args:
            features: 1D numpy array of shape (88,) or 2D array of shape (1, 88).
            hand: Optional DetectedHand object for anatomical rule evaluation.

        Returns:
            Tuple of (predicted_gesture_label, confidence_score).
        """
        # If hand geometry is available, run geometric rule evaluation
        rule_label = None
        rule_conf = 0.0
        if hand is not None:
            rule_label, rule_conf = self.rule_classifier.predict_hand(hand)

        # If no ML model is loaded, return rule classification directly
        if self.model is None:
            if rule_label is not None:
                return rule_label, rule_conf
            raise ModelNotLoadedError("Neither model nor hand geometry is available for prediction.")

        if features.ndim == 1:
            X = features.reshape(1, -1)
        elif features.ndim == 2:
            X = features
        else:
            raise ValueError(f"Expected 1D or 2D feature array, got ndim={features.ndim}")

        probabilities = self.model.predict_proba(X)[0]
        max_idx = int(np.argmax(probabilities))
        ml_confidence = float(probabilities[max_idx])
        predicted_class_id = self.model.classes_[max_idx]

        if isinstance(predicted_class_id, (int, np.integer)):
            if 0 <= predicted_class_id < len(self.label_names):
                ml_label = self.label_names[predicted_class_id]
            else:
                ml_label = str(predicted_class_id)
        else:
            ml_label = str(predicted_class_id)

        # Hybrid Decision Logic:
        # If rule classifier has high confidence (>= 0.90) and ML model is below threshold or differs,
        # prioritize anatomical rule detection (especially effective before custom model collection)
        is_synthetic = self.metadata.get("is_synthetic", True)
        if rule_label is not None and rule_label != "Unknown Gesture":
            if is_synthetic or ml_confidence < self.config.confidence_threshold:
                return rule_label, max(rule_conf, ml_confidence)

        # Otherwise use ML model if above confidence cutoff
        if ml_confidence >= self.config.confidence_threshold:
            return ml_label, ml_confidence

        if rule_label is not None and rule_label != "Unknown Gesture":
            return rule_label, rule_conf

        return self.config.unknown_label, ml_confidence

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return full class probability distribution."""
        if self.model is None:
            raise ModelNotLoadedError("Model is not loaded.")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.model.predict_proba(features)

    def save(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = MODEL_METADATA_PATH,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Serialize trained model with joblib and export training metadata to JSON."""
        if self.model is None:
            raise ModelNotLoadedError("Cannot save an uninitialized model.")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to: {model_path}")

        meta: Dict[str, Any] = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "n_features_in": getattr(self.model, "n_features_in_", 88),
            "classes": [int(c) if isinstance(c, (int, np.integer)) else str(c) for c in self.model.classes_],
            "label_names": self.label_names,
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "is_synthetic": False,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Metadata saved to: {metadata_path}")

    @classmethod
    def load(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = MODEL_METADATA_PATH,
        config: Optional[ClassifierConfig] = None,
    ) -> GestureClassifier:
        """Load a saved model checkpoint and associated metadata."""
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
                f"Please train a model first by running 'python scripts/train_model.py'."
            )

        model = joblib.load(model_path)

        label_names = gesture_config.GESTURES
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    if "label_names" in metadata:
                        label_names = metadata["label_names"]
            except Exception as e:
                logger.warning(f"Failed to read model metadata: {e}")

        classifier = cls(model=model, label_names=label_names, config=config)
        classifier.metadata = metadata
        logger.info(f"Loaded classifier from {model_path} with {len(label_names)} classes.")
        return classifier
