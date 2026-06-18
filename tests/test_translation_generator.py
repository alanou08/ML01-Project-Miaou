"""Tests for rule-based playful interpretations."""

from __future__ import annotations

import pytest

from src.config import CANONICAL_CLASSES
from src.translation_generator import TRANSLATIONS, generate_translation


@pytest.mark.parametrize("class_name", CANONICAL_CLASSES)
def test_translation_exists_for_every_class(class_name: str) -> None:
    translation = generate_translation(class_name, seed=42)

    assert translation in TRANSLATIONS[class_name]
    assert len(TRANSLATIONS[class_name]) >= 6


def test_translation_is_deterministic_for_same_seed_and_key() -> None:
    first = generate_translation("waiting_for_food", seed=42, key="audio-a")
    second = generate_translation("waiting_for_food", seed=42, key="audio-a")

    assert first == second


def test_unknown_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown class"):
        generate_translation("playful_zoomies")
