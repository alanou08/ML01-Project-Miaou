"""Central configuration for the PetSpeak project."""

from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
METADATA_PATH: Final[Path] = DATA_DIR / "metadata.csv"
FEATURE_CACHE_PATH: Final[Path] = PROCESSED_DATA_DIR / "features.npz"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
MODEL_PATH: Final[Path] = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH: Final[Path] = MODELS_DIR / "model_metadata.json"
SPLIT_MANIFEST_PATH: Final[Path] = MODELS_DIR / "split_manifest.json"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
MODEL_COMPARISON_PATH: Final[Path] = REPORTS_DIR / "model_comparison.csv"
CLASSIFICATION_REPORT_PATH: Final[Path] = REPORTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH: Final[Path] = REPORTS_DIR / "confusion_matrix.png"
CLASS_DISTRIBUTION_PATH: Final[Path] = REPORTS_DIR / "class_distribution.png"

RANDOM_STATE: Final[int] = 42
SAMPLE_RATE: Final[int] = 16_000
DURATION_SECONDS: Final[float] = 3.0
TARGET_NUM_SAMPLES: Final[int] = int(SAMPLE_RATE * DURATION_SECONDS)
N_MFCC: Final[int] = 20
N_MELS: Final[int] = 64
N_FFT: Final[int] = 1_024
HOP_LENGTH: Final[int] = 256
TRIM_TOP_DB: Final[float] = 35.0
TEST_SIZE: Final[float] = 0.20
MAX_CV_SPLITS: Final[int] = 5

VALID_AUDIO_EXTENSIONS: Final[tuple[str, ...]] = (
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".m4a",
)

CANONICAL_CLASSES: Final[tuple[str, ...]] = (
    "waiting_for_food",
    "isolation",
    "brushing",
)

DISPLAY_CLASS_NAMES: Final[dict[str, str]] = {
    "waiting_for_food": "Waiting for food",
    "isolation": "Isolation",
    "brushing": "Brushing",
}

# Aliases are normalized to lowercase before matching. Multi-word aliases may be
# written with spaces, underscores, or hyphens in filenames and folder names.
LABEL_ALIASES: Final[dict[str, str]] = {
    "food": "waiting_for_food",
    "waiting": "waiting_for_food",
    "waiting for food": "waiting_for_food",
    "waiting_for_food": "waiting_for_food",
    "waiting-food": "waiting_for_food",
    "feeding": "waiting_for_food",
    "feed": "waiting_for_food",
    "hungry": "waiting_for_food",
    "f": "waiting_for_food",
    "isolation": "isolation",
    "isolated": "isolation",
    "alone": "isolation",
    "unfamiliar": "isolation",
    "i": "isolation",
    "brushing": "brushing",
    "brush": "brushing",
    "grooming": "brushing",
    "groom": "brushing",
    "b": "brushing",
}

PREPROCESSING_CONFIG: Final[dict[str, int | float | bool]] = {
    "sample_rate": SAMPLE_RATE,
    "duration_seconds": DURATION_SECONDS,
    "target_num_samples": TARGET_NUM_SAMPLES,
    "mono": True,
    "normalize": True,
    "trim_top_db": TRIM_TOP_DB,
}

FEATURE_CONFIG: Final[dict[str, int]] = {
    "n_mfcc": N_MFCC,
    "n_mels": N_MELS,
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
}
