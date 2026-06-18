"""Evaluation utilities and standalone evaluation command for PetSpeak."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    CANONICAL_CLASSES,
    CLASSIFICATION_REPORT_PATH,
    CLASS_DISTRIBUTION_PATH,
    CONFUSION_MATRIX_PATH,
    DISPLAY_CLASS_NAMES,
    FEATURE_CACHE_PATH,
    METADATA_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    SPLIT_MANIFEST_PATH,
)
from src.dataset import FeatureDataset, build_feature_dataset, load_metadata
from src.utils import configure_logging, read_json

LOGGER = logging.getLogger(__name__)


def evaluate_predictions(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, Any]:
    """Compute aggregate and per-class classification metrics."""
    true_array = np.asarray(true_labels, dtype=str)
    predicted_array = np.asarray(predicted_labels, dtype=str)

    report_dict = classification_report(
        true_array,
        predicted_array,
        labels=list(CANONICAL_CLASSES),
        target_names=list(CANONICAL_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true_array, predicted_array)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_array, predicted_array)
        ),
        "precision_macro": float(
            precision_score(
                true_array,
                predicted_array,
                labels=list(CANONICAL_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                true_array,
                predicted_array,
                labels=list(CANONICAL_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                true_array,
                predicted_array,
                labels=list(CANONICAL_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                true_array,
                predicted_array,
                labels=list(CANONICAL_CLASSES),
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": {
            class_name: {
                metric: float(report_dict[class_name][metric])
                for metric in ("precision", "recall", "f1-score", "support")
            }
            for class_name in CANONICAL_CLASSES
        },
    }


def classification_report_text(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> str:
    """Create the human-readable sklearn classification report."""
    return classification_report(
        true_labels,
        predicted_labels,
        labels=list(CANONICAL_CLASSES),
        target_names=[DISPLAY_CLASS_NAMES[name] for name in CANONICAL_CLASSES],
        digits=4,
        zero_division=0,
    )


def save_classification_report(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    output_path: Path = CLASSIFICATION_REPORT_PATH,
    model_name: str | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    """Save a readable report with aggregate metrics and per-class details."""
    sections: list[str] = []
    if model_name:
        sections.append(f"Selected model: {model_name}\n")
    if extra_metrics:
        sections.append("Aggregate metrics:")
        for key in (
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
        ):
            if key in extra_metrics:
                sections.append(f"- {key}: {float(extra_metrics[key]):.4f}")
        sections.append("")
    sections.append("Classification report:")
    sections.append(classification_report_text(true_labels, predicted_labels))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    output_path: Path = CONFUSION_MATRIX_PATH,
) -> None:
    """Save a labeled confusion matrix using matplotlib only."""
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(CANONICAL_CLASSES),
    )
    display_names = [DISPLAY_CLASS_NAMES[name] for name in CANONICAL_CLASSES]

    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title="PetSpeak confusion matrix",
        xlabel="Predicted behavioral context",
        ylabel="True behavioral context",
        xticks=np.arange(len(display_names)),
        yticks=np.arange(len(display_names)),
        xticklabels=display_names,
        yticklabels=display_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right")

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_class_distribution(
    labels: np.ndarray,
    output_path: Path = CLASS_DISTRIBUTION_PATH,
) -> None:
    """Save the number of usable recordings in each canonical class."""
    counts = [int(np.sum(labels == name)) for name in CANONICAL_CLASSES]
    display_names = [DISPLAY_CLASS_NAMES[name] for name in CANONICAL_CLASSES]

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    bars = axis.bar(display_names, counts)
    axis.set_title("Distribution of valid CatMeows recordings")
    axis.set_xlabel("Behavioral context")
    axis.set_ylabel("Number of recordings")
    axis.tick_params(axis="x", rotation=20)
    for bar, count in zip(bars, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _indices_for_paths(dataset: FeatureDataset, wanted_paths: list[str]) -> np.ndarray:
    path_to_index = {path: index for index, path in enumerate(dataset.paths.tolist())}
    missing = [path for path in wanted_paths if path not in path_to_index]
    if missing:
        raise ValueError(
            "The feature cache no longer matches the saved split. Missing paths: "
            + ", ".join(missing[:5])
        )
    return np.asarray([path_to_index[path] for path in wanted_paths], dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate the saved PetSpeak model on its held-out cats."
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--cache", type=Path, default=FEATURE_CACHE_PATH)
    parser.add_argument("--split", type=Path, default=SPLIT_MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    arguments = parse_args()
    if not arguments.model.is_file():
        raise FileNotFoundError(
            f"Model not found at {arguments.model}. Run python -m src.train first."
        )
    if not arguments.split.is_file():
        raise FileNotFoundError(
            f"Split manifest not found at {arguments.split}. Run python -m src.train first."
        )

    metadata = load_metadata(arguments.metadata)
    dataset = build_feature_dataset(metadata, arguments.cache)
    split_manifest = read_json(arguments.split)
    test_indices = _indices_for_paths(dataset, split_manifest["test_paths"])

    model = joblib.load(arguments.model)
    true_labels = dataset.labels[test_indices]
    predicted_labels = np.asarray(
        model.predict(dataset.features[test_indices]), dtype=str
    )
    metrics = evaluate_predictions(true_labels, predicted_labels)

    model_name = "Saved PetSpeak model"
    if MODEL_METADATA_PATH.is_file():
        model_name = str(read_json(MODEL_METADATA_PATH).get("model_name", model_name))

    save_classification_report(
        true_labels,
        predicted_labels,
        model_name=model_name,
        extra_metrics=metrics,
    )
    plot_confusion_matrix(true_labels, predicted_labels)
    plot_class_distribution(dataset.labels)

    LOGGER.info("Evaluation complete for %d held-out recordings.", len(test_indices))
    print(f"Model: {model_name}")
    print(f"Held-out macro F1: {metrics['f1_macro']:.4f}")
    print(f"Held-out balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Reports saved in: {CLASSIFICATION_REPORT_PATH.parent}")


if __name__ == "__main__":
    main()
