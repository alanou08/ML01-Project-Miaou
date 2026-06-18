"""Rule-based playful interpretations for predicted behavioral contexts."""

from __future__ import annotations

import hashlib
import random

from src.config import CANONICAL_CLASSES, RANDOM_STATE

DISCLAIMER = (
    "This is an experimental behavioral-context prediction, not a literal "
    "translation of animal language."
)

TRANSLATIONS: dict[str, tuple[str, ...]] = {
    "waiting_for_food": (
        "Excuse me, human. Dinner appears to be late.",
        "My food bowl looks suspiciously empty.",
        "I have checked twice, and the snacks are still missing.",
        "This is your official reminder that I am extremely hungry.",
        "The kitchen is nearby, and I believe you know what to do.",
        "I would like to report a serious shortage of cat food.",
        "Please begin the feeding ceremony immediately.",
        "I can see the bowl, but I cannot see the meal.",
    ),
    "isolation": (
        "Where did everybody go?",
        "I would prefer not to be alone right now.",
        "Hello? A little company would be appreciated.",
        "This room is unfamiliar, and I would like my human back.",
        "I am requesting immediate emotional support.",
        "Could someone please return and explain this situation?",
        "I have searched the area and found zero humans.",
        "Being alone was not included in today's plan.",
    ),
    "brushing": (
        "This brushing session is acceptable.",
        "You may continue giving me attention.",
        "Careful with the fur, human. I have standards.",
        "Yes, that spot is approved. Continue gently.",
        "My coat maintenance is progressing nicely.",
        "I am supervising this grooming session very closely.",
        "The brush may proceed, provided the service remains excellent.",
        "This spa treatment has earned a cautious purr of approval.",
    ),
}


def generate_translation(
    predicted_class: str,
    seed: int = RANDOM_STATE,
    key: str | bytes | None = None,
) -> str:
    """Select a deterministic playful sentence for a canonical class.

    Args:
        predicted_class: One of the three canonical PetSpeak class names.
        seed: Base seed used for reproducibility.
        key: Optional stable value, such as an audio digest, used to vary the
            result while keeping repeated predictions deterministic.
    """
    if predicted_class not in TRANSLATIONS:
        raise ValueError(
            f"Unknown class '{predicted_class}'. Expected one of "
            f"{list(CANONICAL_CLASSES)}."
        )

    combined_seed = int(seed)
    if key is not None:
        key_bytes = key if isinstance(key, bytes) else key.encode("utf-8")
        digest = hashlib.sha256(key_bytes).digest()
        combined_seed ^= int.from_bytes(digest[:8], byteorder="big", signed=False)

    generator = random.Random(combined_seed)
    return generator.choice(TRANSLATIONS[predicted_class])
