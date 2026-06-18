"""General utilities shared across PetSpeak modules."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a concise project-wide logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def ensure_project_directories() -> None:
    """Create directories that may be absent in a freshly copied project."""
    for directory in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def set_reproducible_seed(seed: int) -> None:
    """Seed Python and NumPy random number generators."""
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable, human-readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def stable_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 hash for JSON-serializable data."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
