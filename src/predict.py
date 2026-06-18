"""Command-line and reusable prediction helpers for PetSpeak."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.audio_processing import AudioInfo, AudioProcessingError, load_and_preprocess_audio
from src.config import (
    CANONICAL_CLASSES,
    DISPLAY_CLASS_NAMES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
)
from src.feature_extraction import FeatureExtractionError, extract_features
from src.translation_generator import DISCLAIMER, generate_translation
from src.utils import configure_logging, read_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction output shared by the CLI and Streamlit app."""

    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    interpretation: str
    audio_info: AudioInfo


def load_saved_model(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> tuple[Any, dict[str, Any]]:
    """Load the trained estimator and its descriptive metadata."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run python -m src.train first."
        )
    model = joblib.load(model_path)
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    return model, metadata


def ordered_probabilities(model: Any, feature_row: np.ndarray) -> dict[str, float]:
    """Return probabilities in the project-wide canonical class order."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("The saved model does not support probability prediction.")
    raw_probabilities = np.asarray(model.predict_proba(feature_row), dtype=float)[0]
    model_classes = [str(value) for value in model.classes_]
    probability_lookup = dict(zip(model_classes, raw_probabilities))
    return {
        class_name: float(probability_lookup.get(class_name, 0.0))
        for class_name in CANONICAL_CLASSES
    }


def predict_audio_file(
    audio_path: str | Path,
    model: Any | None = None,
) -> PredictionResult:
    """Preprocess, featurize, and classify one audio file."""
    path = Path(audio_path)
    loaded_model = model if model is not None else load_saved_model()[0]
    audio, audio_info = load_and_preprocess_audio(path)
    feature_row = extract_features(audio).reshape(1, -1)
    probabilities = ordered_probabilities(loaded_model, feature_row)
    predicted_class = max(probabilities, key=probabilities.get)
    audio_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    interpretation = generate_translation(predicted_class, key=audio_digest)
    return PredictionResult(
        predicted_class=predicted_class,
        confidence=probabilities[predicted_class],
        probabilities=probabilities,
        interpretation=interpretation,
        audio_info=audio_info,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the behavioral context of one cat vocalization."
    )
    parser.add_argument("--audio", type=Path, required=True, help="Audio file to classify.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    arguments = parse_args()
    try:
        model, _ = load_saved_model(arguments.model)
        result = predict_audio_file(arguments.audio, model=model)
    except (
        FileNotFoundError,
        AudioProcessingError,
        FeatureExtractionError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("Prediction failed: %s", exc)
        sys.exit(1)

    print(f"Predicted context: {DISPLAY_CLASS_NAMES[result.predicted_class]}")
    print(f"Confidence: {result.confidence * 100:.2f}%")
    print("\nClass probabilities:")
    for class_name in CANONICAL_CLASSES:
        print(
            f"- {DISPLAY_CLASS_NAMES[class_name]}: "
            f"{result.probabilities[class_name] * 100:.2f}%"
        )
    print("\nPlayful interpretation:")
    print(f"“{result.interpretation}”")
    print("\nDisclaimer:")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
