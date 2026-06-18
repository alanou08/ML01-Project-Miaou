"""Metadata validation, grouped splitting, and feature caching."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - retained for older compatible sklearn.
    StratifiedGroupKFold = None  # type: ignore[assignment,misc]

from src.audio_processing import AudioProcessingError, load_and_preprocess_audio
from src.config import (
    CANONICAL_CLASSES,
    FEATURE_CACHE_PATH,
    FEATURE_CONFIG,
    MAX_CV_SPLITS,
    METADATA_PATH,
    PREPROCESSING_CONFIG,
    PROJECT_ROOT,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.feature_extraction import FeatureExtractionError, extract_features
from src.utils import stable_hash

LOGGER = logging.getLogger(__name__)
REQUIRED_METADATA_COLUMNS = {"audio_path", "label", "cat_id"}


@dataclass(frozen=True)
class FeatureDataset:
    """In-memory feature matrix and aligned metadata arrays."""

    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    paths: np.ndarray

    def __len__(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class DatasetSplit:
    """Indices for a leakage-safe grouped train/test split."""

    train_indices: np.ndarray
    test_indices: np.ndarray


def resolve_audio_path(value: str | Path) -> Path:
    """Resolve CSV paths relative to the project root when needed."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_metadata(path: Path = METADATA_PATH) -> pd.DataFrame:
    """Load metadata and retain only rows with valid, existing inputs."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {path}. Run python -m src.build_metadata first."
        )

    dataframe = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_columns = REQUIRED_METADATA_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()
    dataframe["label"] = dataframe["label"].str.strip()
    dataframe["cat_id"] = dataframe["cat_id"].str.strip()
    dataframe["audio_path"] = dataframe["audio_path"].str.strip()

    valid_label_mask = dataframe["label"].isin(CANONICAL_CLASSES)
    valid_cat_mask = dataframe["cat_id"].ne("")
    valid_path_mask = dataframe["audio_path"].ne("")
    valid_mask = valid_label_mask & valid_cat_mask & valid_path_mask

    for row in dataframe.loc[~valid_mask].itertuples(index=False):
        LOGGER.warning(
            "Skipping invalid metadata row: path=%r label=%r cat_id=%r",
            row.audio_path,
            row.label,
            row.cat_id,
        )

    dataframe = dataframe.loc[valid_mask].reset_index(drop=True)
    if dataframe.empty:
        raise ValueError(
            "No valid metadata rows remain. Correct data/metadata.csv and try again."
        )
    return dataframe


def _cache_fingerprint(metadata: pd.DataFrame) -> str:
    records: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        path = resolve_audio_path(row.audio_path)
        try:
            stat = path.stat()
            file_state: dict[str, object] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        except OSError:
            file_state = {"size": None, "mtime_ns": None}
        records.append(
            {
                "audio_path": str(row.audio_path),
                "label": str(row.label),
                "cat_id": str(row.cat_id),
                **file_state,
            }
        )
    return stable_hash(
        {
            "records": records,
            "preprocessing": PREPROCESSING_CONFIG,
            "features": FEATURE_CONFIG,
            "classes": CANONICAL_CLASSES,
        }
    )


def _load_cache(cache_path: Path, expected_fingerprint: str) -> FeatureDataset | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            fingerprint = str(cache["fingerprint"].item())
            if fingerprint != expected_fingerprint:
                LOGGER.info("Feature cache is stale and will be rebuilt.")
                return None
            dataset = FeatureDataset(
                features=np.asarray(cache["features"], dtype=np.float32),
                labels=np.asarray(cache["labels"], dtype=str),
                groups=np.asarray(cache["groups"], dtype=str),
                paths=np.asarray(cache["paths"], dtype=str),
            )
        LOGGER.info("Loaded %d feature vectors from %s", len(dataset), cache_path)
        return dataset
    except Exception as exc:
        LOGGER.warning("Could not read feature cache %s: %s", cache_path, exc)
        return None


def build_feature_dataset(
    metadata: pd.DataFrame,
    cache_path: Path = FEATURE_CACHE_PATH,
    force_rebuild: bool = False,
) -> FeatureDataset:
    """Load a matching cache or extract features while skipping invalid files."""
    fingerprint = _cache_fingerprint(metadata)
    if not force_rebuild:
        cached = _load_cache(cache_path, fingerprint)
        if cached is not None:
            return cached

    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    paths: list[str] = []

    for index, row in enumerate(metadata.itertuples(index=False), start=1):
        path = resolve_audio_path(row.audio_path)
        try:
            audio, _ = load_and_preprocess_audio(path)
            vector = extract_features(audio)
        except (AudioProcessingError, FeatureExtractionError, OSError, ValueError) as exc:
            LOGGER.error("Skipping unreadable file %s: %s", path, exc)
            continue

        features.append(vector)
        labels.append(str(row.label))
        groups.append(str(row.cat_id))
        paths.append(str(row.audio_path))
        if index % 50 == 0:
            LOGGER.info("Processed %d/%d metadata rows.", index, len(metadata))

    if not features:
        raise RuntimeError("No valid audio features could be extracted.")

    dataset = FeatureDataset(
        features=np.vstack(features).astype(np.float32),
        labels=np.asarray(labels, dtype=str),
        groups=np.asarray(groups, dtype=str),
        paths=np.asarray(paths, dtype=str),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        fingerprint=np.asarray(fingerprint),
        features=dataset.features,
        labels=dataset.labels,
        groups=dataset.groups,
        paths=dataset.paths,
        cache_metadata=np.asarray(
            json.dumps(
                {
                    "preprocessing": PREPROCESSING_CONFIG,
                    "features": FEATURE_CONFIG,
                    "classes": CANONICAL_CLASSES,
                },
                sort_keys=True,
            )
        ),
    )
    LOGGER.info("Saved %d feature vectors to %s", len(dataset), cache_path)
    return dataset


def _distribution_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_counts = np.array(
        [np.sum(reference == label) for label in CANONICAL_CLASSES], dtype=float
    )
    candidate_counts = np.array(
        [np.sum(candidate == label) for label in CANONICAL_CLASSES], dtype=float
    )
    reference_distribution = reference_counts / max(reference_counts.sum(), 1.0)
    candidate_distribution = candidate_counts / max(candidate_counts.sum(), 1.0)
    return float(np.abs(reference_distribution - candidate_distribution).sum())


def grouped_train_test_split(
    labels: np.ndarray,
    groups: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Choose a GroupShuffleSplit with all classes and close class balance."""
    labels = np.asarray(labels, dtype=str)
    groups = np.asarray(groups, dtype=str)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("At least two distinct cats are required for a grouped split.")

    splitter = GroupShuffleSplit(
        n_splits=200,
        test_size=test_size,
        random_state=random_state,
    )
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    dummy = np.zeros((labels.size, 1), dtype=np.float32)

    for train_indices, test_indices in splitter.split(dummy, labels, groups):
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        if set(train_labels) != set(CANONICAL_CLASSES):
            continue
        if set(test_labels) != set(CANONICAL_CLASSES):
            continue
        overlap = set(groups[train_indices]).intersection(groups[test_indices])
        if overlap:
            continue
        score = _distribution_distance(labels, test_labels)
        size_penalty = abs((len(test_indices) / len(labels)) - test_size)
        total_score = score + size_penalty
        if best is None or total_score < best[0]:
            best = (total_score, train_indices, test_indices)

    if best is None:
        raise ValueError(
            "Could not create a grouped train/test split containing all classes in "
            "both parts. Check cat IDs and class coverage."
        )
    return DatasetSplit(
        train_indices=np.asarray(best[1], dtype=int),
        test_indices=np.asarray(best[2], dtype=int),
    )


def make_grouped_cv(
    labels: np.ndarray,
    groups: np.ndarray,
    max_splits: int = MAX_CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> object:
    """Create grouped cross-validation, preferring stratification when feasible."""
    labels = np.asarray(labels, dtype=str)
    groups = np.asarray(groups, dtype=str)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Grouped cross-validation needs at least two cats.")

    groups_per_class = [
        np.unique(groups[labels == class_name]).size
        for class_name in CANONICAL_CLASSES
    ]
    feasible_splits = min(max_splits, unique_groups.size, min(groups_per_class))

    if StratifiedGroupKFold is not None and feasible_splits >= 2:
        LOGGER.info("Using StratifiedGroupKFold with %d folds.", feasible_splits)
        return StratifiedGroupKFold(
            n_splits=int(feasible_splits),
            shuffle=True,
            random_state=random_state,
        )

    fallback_splits = min(max_splits, unique_groups.size)
    if fallback_splits < 2:
        raise ValueError("Not enough cats for grouped cross-validation.")
    LOGGER.warning(
        "Stratified grouped CV is unavailable or infeasible; using GroupKFold "
        "with %d folds.",
        fallback_splits,
    )
    return GroupKFold(n_splits=int(fallback_splits))


def iter_cv_splits(
    cv: object,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Typed helper mainly useful for diagnostics and tests."""
    yield from cv.split(features, labels, groups)  # type: ignore[attr-defined]
