"""Safe and reusable audio loading and preprocessing."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

from src.config import (
    DURATION_SECONDS,
    SAMPLE_RATE,
    TARGET_NUM_SAMPLES,
    TRIM_TOP_DB,
)

LOGGER = logging.getLogger(__name__)
SILENCE_EPSILON = 1e-8


class AudioProcessingError(RuntimeError):
    """Raised when an audio file cannot be decoded or prepared safely."""


@dataclass(frozen=True)
class AudioInfo:
    """Basic information about a decoded recording."""

    original_sample_rate: int
    original_num_samples: int
    original_duration_seconds: float
    channels: int
    processed_sample_rate: int
    processed_num_samples: int
    processed_duration_seconds: float
    was_silent: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        """Convert the dataclass into a JSON-friendly dictionary."""
        return asdict(self)


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert a librosa-style audio array to mono float32."""
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return np.asarray(librosa.to_mono(array), dtype=np.float32)
    raise AudioProcessingError(
        f"Unsupported audio shape {array.shape}; expected mono or multi-channel audio."
    )


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio while keeping silent input silent."""
    array = np.nan_to_num(
        np.asarray(audio, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if array.size == 0:
        return array
    peak = float(np.max(np.abs(array)))
    if peak <= SILENCE_EPSILON:
        return np.zeros_like(array, dtype=np.float32)
    return np.asarray(array / peak, dtype=np.float32)


def trim_long_silence(
    audio: np.ndarray,
    top_db: float = TRIM_TOP_DB,
) -> np.ndarray:
    """Trim leading and trailing silence without failing on silent audio."""
    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0 or float(np.max(np.abs(array))) <= SILENCE_EPSILON:
        return array
    trimmed, _ = librosa.effects.trim(array, top_db=top_db)
    return np.asarray(trimmed, dtype=np.float32)


def pad_or_crop(
    audio: np.ndarray,
    target_length: int = TARGET_NUM_SAMPLES,
) -> np.ndarray:
    """Center-pad short audio or center-crop long audio to target length."""
    if target_length <= 0:
        raise ValueError("target_length must be positive.")

    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    current_length = array.size

    if current_length == target_length:
        return array.copy()
    if current_length < target_length:
        total_padding = target_length - current_length
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        return np.pad(array, (left_padding, right_padding), mode="constant")

    start = (current_length - target_length) // 2
    return array[start : start + target_length].copy()


def preprocess_audio_array(
    audio: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int = SAMPLE_RATE,
    target_num_samples: int = TARGET_NUM_SAMPLES,
    trim_top_db: float = TRIM_TOP_DB,
) -> np.ndarray:
    """Apply the complete PetSpeak preprocessing pipeline to an array."""
    if original_sample_rate <= 0:
        raise AudioProcessingError("The source sample rate must be positive.")

    mono = convert_to_mono(audio)
    if mono.size == 0:
        LOGGER.warning("Received an empty audio array; replacing it with silence.")
        mono = np.zeros(1, dtype=np.float32)

    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    if original_sample_rate != target_sample_rate:
        mono = librosa.resample(
            mono,
            orig_sr=original_sample_rate,
            target_sr=target_sample_rate,
            res_type="soxr_hq",
        )

    normalized = normalize_audio(mono)
    trimmed = trim_long_silence(normalized, top_db=trim_top_db)
    if trimmed.size == 0:
        trimmed = np.zeros(1, dtype=np.float32)
    fixed = pad_or_crop(trimmed, target_length=target_num_samples)
    return np.asarray(fixed, dtype=np.float32)


def load_and_preprocess_audio(path: str | Path) -> tuple[np.ndarray, AudioInfo]:
    """Decode an audio file and return fixed-length audio plus metadata.

    Raises:
        AudioProcessingError: If the path is missing or cannot be decoded.
    """
    audio_path = Path(path)
    if not audio_path.is_file():
        raise AudioProcessingError(f"Audio file does not exist: {audio_path}")

    try:
        decoded, original_sr = librosa.load(
            audio_path,
            sr=None,
            mono=False,
            dtype=np.float32,
        )
    except Exception as exc:  # Decoder errors depend on installed backends.
        raise AudioProcessingError(
            f"Could not decode '{audio_path.name}': {exc}"
        ) from exc

    decoded_array = np.asarray(decoded, dtype=np.float32)
    if decoded_array.ndim == 1:
        channels = 1
        original_num_samples = decoded_array.size
    elif decoded_array.ndim == 2:
        channels = decoded_array.shape[0]
        original_num_samples = decoded_array.shape[1]
    else:
        raise AudioProcessingError(
            f"Decoded audio has unsupported shape {decoded_array.shape}."
        )

    if original_num_samples == 0:
        LOGGER.warning("Audio file %s is empty; using silence.", audio_path)

    processed = preprocess_audio_array(decoded_array, int(original_sr))
    is_silent = float(np.max(np.abs(processed))) <= SILENCE_EPSILON
    info = AudioInfo(
        original_sample_rate=int(original_sr),
        original_num_samples=int(original_num_samples),
        original_duration_seconds=(
            float(original_num_samples / original_sr) if original_sr else 0.0
        ),
        channels=int(channels),
        processed_sample_rate=SAMPLE_RATE,
        processed_num_samples=int(processed.size),
        processed_duration_seconds=DURATION_SECONDS,
        was_silent=is_silent,
    )
    return processed, info


def load_audio_bytes(
    data: bytes,
    suffix: str = ".wav",
) -> tuple[np.ndarray, AudioInfo]:
    """Decode uploaded audio bytes through the same file-based pipeline."""
    if not data:
        raise AudioProcessingError("The uploaded audio file is empty.")

    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=normalized_suffix,
            delete=False,
        ) as temp_file:
            temp_file.write(data)
            temp_path = Path(temp_file.name)
        return load_and_preprocess_audio(temp_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
