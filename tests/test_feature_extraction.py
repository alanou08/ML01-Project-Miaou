"""Tests for fixed-length audio features."""

from __future__ import annotations

import numpy as np

from src.config import SAMPLE_RATE, TARGET_NUM_SAMPLES
from src.feature_extraction import expected_feature_length, extract_features


def test_feature_vector_has_fixed_length() -> None:
    time = np.arange(TARGET_NUM_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    synthetic_meow = 0.5 * np.sin(2 * np.pi * 440.0 * time)

    features = extract_features(synthetic_meow.astype(np.float32))

    assert features.shape == (expected_feature_length(),)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_silent_audio_features_are_finite() -> None:
    silence = np.zeros(TARGET_NUM_SAMPLES, dtype=np.float32)

    features = extract_features(silence)

    assert features.shape == (expected_feature_length(),)
    assert np.all(np.isfinite(features))
