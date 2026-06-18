"""Build editable metadata for audio files stored under data/raw/."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    CANONICAL_CLASSES,
    LABEL_ALIASES,
    METADATA_PATH,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    VALID_AUDIO_EXTENSIONS,
)
from src.utils import configure_logging, ensure_project_directories

LOGGER = logging.getLogger(__name__)

# Official CatMeows pattern: C_NNNNN_BB_SS_OOOOO_RXX.wav
OFFICIAL_FILENAME_PATTERN = re.compile(
    r"^(?P<context>[BFI])_(?P<cat_id>[A-Za-z0-9]{5})_"
    r"(?P<breed>[A-Za-z]{2})_(?P<sex>[A-Za-z]{2})_"
    r"(?P<owner>[A-Za-z0-9]{5})_(?P<session>R?[123])(?P<count>\d{2})$",
    flags=re.IGNORECASE,
)
CAT_TOKEN_PATTERN = re.compile(
    r"(?:^|[_\-\s])(?:cat|kitty|subject|animal|id)[_\-\s]*"
    r"(?P<cat_id>[A-Za-z0-9]+)(?:$|[_\-\s])",
    flags=re.IGNORECASE,
)

GENERIC_DIRECTORY_NAMES = {
    "audio",
    "audios",
    "data",
    "dataset",
    "raw",
    "recordings",
    "samples",
    "sounds",
    "train",
    "test",
    "validation",
    "val",
}


@dataclass(frozen=True)
class DetectionResult:
    """Detected metadata and an explanation suitable for CSV review."""

    label: str | None
    cat_id: str | None
    notes: tuple[str, ...]


def normalize_text(value: str) -> str:
    """Normalize a path token for conservative alias matching."""
    normalized = re.sub(r"[_\-]+", " ", value.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _canonical_from_official_code(code: str) -> str | None:
    return {
        "F": "waiting_for_food",
        "I": "isolation",
        "B": "brushing",
    }.get(code.upper())


def detect_official_filename(path: Path) -> tuple[str | None, str | None]:
    """Parse the published CatMeows filename convention when present."""
    match = OFFICIAL_FILENAME_PATTERN.match(path.stem)
    if not match:
        return None, None
    return (
        _canonical_from_official_code(match.group("context")),
        match.group("cat_id").upper(),
    )


def _label_candidates(path: Path, raw_dir: Path) -> set[str]:
    candidates: set[str] = set()
    try:
        relative = path.relative_to(raw_dir)
        parts = list(relative.parts[:-1]) + [path.stem]
    except ValueError:
        parts = list(path.parts[-4:-1]) + [path.stem]

    # Whole folder/stem matching handles names such as waiting_for_food.
    for part in parts:
        normalized_part = normalize_text(part)
        canonical = LABEL_ALIASES.get(normalized_part)
        if canonical:
            candidates.add(canonical)

        # Token matching is useful for names such as cat12_food_recording.
        tokens = [token for token in re.split(r"[^a-z0-9]+", part.lower()) if token]
        for index, token in enumerate(tokens):
            # Single-letter aliases are accepted only as the first filename token
            # or as a complete folder name to avoid matching arbitrary initials.
            if len(token) == 1 and not (
                normalized_part == token or (part == path.stem and index == 0)
            ):
                continue
            canonical = LABEL_ALIASES.get(token)
            if canonical:
                candidates.add(canonical)
    return candidates


def detect_label(path: Path, raw_dir: Path = RAW_DATA_DIR) -> tuple[str | None, str]:
    """Detect a class label without guessing when evidence conflicts."""
    official_label, _ = detect_official_filename(path)
    if official_label:
        return official_label, "label from official CatMeows filename"

    candidates = _label_candidates(path, raw_dir)
    if len(candidates) == 1:
        return next(iter(candidates)), "label from filename or folder alias"
    if len(candidates) > 1:
        return None, f"conflicting label candidates: {sorted(candidates)}"
    return None, "class label not detected"


def detect_cat_id(path: Path, raw_dir: Path = RAW_DATA_DIR) -> tuple[str | None, str]:
    """Detect a cat identifier from the official pattern or explicit ID tokens."""
    _, official_cat_id = detect_official_filename(path)
    if official_cat_id:
        return official_cat_id, "cat_id from official CatMeows filename"

    search_values = [path.stem]
    try:
        relative = path.relative_to(raw_dir)
        search_values.extend(reversed(relative.parts[:-1]))
    except ValueError:
        search_values.extend(reversed(path.parts[-4:-1]))

    for value in search_values:
        match = CAT_TOKEN_PATTERN.search(f"_{value}_")
        if match:
            return match.group("cat_id").upper(), "cat_id from explicit cat/subject token"

    # Conservative fallback for filenames beginning C_<cat-id>_... where C is a
    # recognized context abbreviation, even if the remaining official fields vary.
    tokens = [token for token in re.split(r"[_\-\s]+", path.stem) if token]
    if len(tokens) >= 2 and tokens[0].upper() in {"B", "F", "I"}:
        candidate = tokens[1]
        if re.fullmatch(r"[A-Za-z0-9]{2,20}", candidate):
            return candidate.upper(), "cat_id inferred from second filename field"

    # Folder structures such as food/cat_001/file.wav are handled above. A bare
    # directory name is accepted only if it looks ID-like and is not a class or
    # generic dataset folder.
    for value in search_values[1:]:
        normalized = normalize_text(value)
        if normalized in GENERIC_DIRECTORY_NAMES or normalized in LABEL_ALIASES:
            continue
        if re.fullmatch(r"[A-Za-z]*\d+[A-Za-z0-9]*", value):
            return value.upper(), "cat_id inferred from ID-like directory name"

    return None, "cat identifier not detected"


def inspect_audio_path(path: Path, raw_dir: Path = RAW_DATA_DIR) -> DetectionResult:
    """Detect editable metadata for one audio path."""
    label, label_note = detect_label(path, raw_dir)
    cat_id, cat_note = detect_cat_id(path, raw_dir)
    return DetectionResult(label=label, cat_id=cat_id, notes=(label_note, cat_note))


def scan_audio_files(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Recursively find supported audio files in stable order."""
    if not raw_dir.exists():
        return []
    valid_extensions = {extension.lower() for extension in VALID_AUDIO_EXTENSIONS}
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in valid_extensions
    )


def build_metadata(
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = METADATA_PATH,
) -> pd.DataFrame:
    """Scan raw audio, detect metadata, and save an editable CSV."""
    ensure_project_directories()
    files = scan_audio_files(raw_dir)
    rows: list[dict[str, str]] = []

    for audio_path in files:
        detection = inspect_audio_path(audio_path, raw_dir)
        try:
            stored_path = audio_path.resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            stored_path = audio_path.resolve()

        issues: list[str] = []
        if detection.label not in CANONICAL_CLASSES:
            issues.append("label")
        if not detection.cat_id:
            issues.append("cat_id")
        status = "valid" if not issues else "manual_review"
        rows.append(
            {
                "audio_path": stored_path.as_posix(),
                "label": detection.label or "",
                "cat_id": detection.cat_id or "",
                "status": status,
                "notes": "; ".join(detection.notes),
            }
        )

    dataframe = pd.DataFrame(
        rows,
        columns=["audio_path", "label", "cat_id", "status", "notes"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    unresolved = dataframe[dataframe["status"] != "valid"] if not dataframe.empty else dataframe
    LOGGER.info("Found %d supported audio files.", len(dataframe))
    LOGGER.info("Saved metadata to %s", output_path)
    if unresolved.empty:
        LOGGER.info("All files have a detected class and cat identifier.")
    else:
        LOGGER.warning(
            "%d file(s) require manual review. Edit label/cat_id in %s.",
            len(unresolved),
            output_path,
        )
        for row in unresolved.itertuples(index=False):
            LOGGER.warning("Unresolved: %s | %s", row.audio_path, row.notes)
    return dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build editable PetSpeak metadata from a raw audio directory."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory recursively scanned for audio files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=METADATA_PATH,
        help="CSV file to create or replace.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    arguments = parse_args()
    build_metadata(arguments.raw_dir, arguments.output)


if __name__ == "__main__":
    main()
