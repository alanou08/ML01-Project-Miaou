"""Tests for the reusable audio preprocessing pipeline."""

from __future__ import annotations

import numpy as np

from src.audio_processing import normalize_audio, pad_or_crop, preprocess_audio_array
from src.config import SAMPLE_RATE, TARGET_NUM_SAMPLES


def test_padding_of_short_audio_centers_signal() -> None:
    short = np.ones(1_000, dtype=np.float32)
    result = pad_or_crop(short, target_length=2_000)

    assert result.shape == (2_000,)
    assert np.allclose(result[500:1_500], 1.0)
    assert np.allclose(result[:500], 0.0)
    assert np.allclose(result[1_500:], 0.0)


def test_cropping_of_long_audio_uses_center() -> None:
    long_audio = np.arange(10, dtype=np.float32)
    result = pad_or_crop(long_audio, target_length=4)

    assert np.array_equal(result, np.array([3, 4, 5, 6], dtype=np.float32))


def test_normalization_reaches_unit_peak() -> None:
    audio = np.array([-2.0, 0.0, 1.0], dtype=np.float32)
    result = normalize_audio(audio)

    assert np.isclose(np.max(np.abs(result)), 1.0)
    assert np.allclose(result, np.array([-1.0, 0.0, 0.5], dtype=np.float32))


def test_silent_audio_is_safe_and_fixed_length() -> None:
    silence = np.zeros(2_000, dtype=np.float32)
    result = preprocess_audio_array(silence, original_sample_rate=SAMPLE_RATE)

    assert result.shape == (TARGET_NUM_SAMPLES,)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, 0.0)
