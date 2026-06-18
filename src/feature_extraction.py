"""MFCC-centered fixed-length feature extraction for PetSpeak."""

from __future__ import annotations

import logging

import librosa
import numpy as np

from src.config import HOP_LENGTH, N_FFT, N_MELS, N_MFCC, SAMPLE_RATE

LOGGER = logging.getLogger(__name__)


class FeatureExtractionError(RuntimeError):
    """Raised when a feature vector cannot be computed safely."""


def _mean_and_std(feature_matrix: np.ndarray) -> np.ndarray:
    """Aggregate frame-level rows using per-row mean and standard deviation."""
    matrix = np.nan_to_num(
        np.asarray(feature_matrix, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    means = np.mean(matrix, axis=1)
    standard_deviations = np.std(matrix, axis=1)
    return np.concatenate((means, standard_deviations)).astype(np.float32)


def expected_feature_length(n_mfcc: int = N_MFCC) -> int:
    """Return the exact dimensionality of a PetSpeak feature vector."""
    mfcc_family = n_mfcc * 2 * 3
    scalar_spectral_features = 5 * 2
    chroma_features = 12 * 2
    return mfcc_family + scalar_spectral_features + chroma_features


def extract_features(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Extract one fixed-length feature vector from preprocessed mono audio.

    The vector contains mean and standard deviation statistics for MFCC,
    delta-MFCC, delta-delta-MFCC, zero-crossing rate, spectral centroid,
    spectral bandwidth, spectral rolloff, RMS energy, and 12-bin chroma.
    """
    signal = np.nan_to_num(
        np.asarray(audio, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if signal.size == 0:
        raise FeatureExtractionError("Cannot extract features from empty audio.")
    if sample_rate <= 0:
        raise FeatureExtractionError("sample_rate must be positive.")

    try:
        mel_power = librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel_power, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc, order=1, mode="nearest")
        delta_delta = librosa.feature.delta(mfcc, order=2, mode="nearest")

        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            signal,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )
        spectral_centroid = librosa.feature.spectral_centroid(
            y=signal,
            sr=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=signal,
            sr=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=signal,
            sr=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            roll_percent=0.85,
        )
        rms_energy = librosa.feature.rms(
            y=signal,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )
        if float(np.max(np.abs(signal))) <= 1e-8:
            chroma = np.zeros((12, mfcc.shape[1]), dtype=np.float32)
        else:
            chroma = librosa.feature.chroma_stft(
                y=signal,
                sr=sample_rate,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_chroma=12,
            )

        feature_vector = np.concatenate(
            (
                _mean_and_std(mfcc),
                _mean_and_std(delta),
                _mean_and_std(delta_delta),
                _mean_and_std(zero_crossing_rate),
                _mean_and_std(spectral_centroid),
                _mean_and_std(spectral_bandwidth),
                _mean_and_std(spectral_rolloff),
                _mean_and_std(rms_energy),
                _mean_and_std(chroma),
            )
        ).astype(np.float32)
    except Exception as exc:
        raise FeatureExtractionError(f"Feature extraction failed: {exc}") from exc

    feature_vector = np.nan_to_num(
        feature_vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    expected = expected_feature_length()
    if feature_vector.shape != (expected,):
        raise FeatureExtractionError(
            f"Unexpected feature shape {feature_vector.shape}; expected ({expected},)."
        )
    return feature_vector
