"""Train, compare, evaluate, and save PetSpeak classifiers."""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    CANONICAL_CLASSES,
    CLASSIFICATION_REPORT_PATH,
    CLASS_DISTRIBUTION_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_CACHE_PATH,
    FEATURE_CONFIG,
    METADATA_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PREPROCESSING_CONFIG,
    RANDOM_STATE,
    REPORTS_DIR,
    SPLIT_MANIFEST_PATH,
)
from src.dataset import (
    FeatureDataset,
    build_feature_dataset,
    grouped_train_test_split,
    load_metadata,
    make_grouped_cv,
)
from src.evaluate import (
    evaluate_predictions,
    plot_class_distribution,
    plot_confusion_matrix,
    save_classification_report,
)
from src.utils import (
    configure_logging,
    ensure_project_directories,
    set_reproducible_seed,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def model_searches(cv: object) -> dict[str, GridSearchCV]:
    """Create modest grouped hyperparameter searches for both required models."""
    random_forest = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    svm_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "Random Forest": GridSearchCV(
            estimator=random_forest,
            param_grid={
                "max_depth": [None, 20],
                "min_samples_leaf": [1],
                "max_features": ["sqrt"],
            },
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
            error_score="raise",
        ),
        "RBF SVM": GridSearchCV(
            estimator=svm_pipeline,
            param_grid={
                "classifier__C": [1.0, 10.0, 30.0],
                "classifier__gamma": ["scale", 0.01],
            },
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
            error_score="raise",
        ),
    }


def _comparison_row(
    model_name: str,
    search: GridSearchCV,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "selection_metric": "grouped_cv_macro_f1",
        "best_cv_macro_f1": float(search.best_score_),
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_precision_macro": metrics["precision_macro"],
        "test_recall_macro": metrics["recall_macro"],
        "test_f1_macro": metrics["f1_macro"],
        "test_f1_weighted": metrics["f1_weighted"],
        "best_parameters": str(search.best_params_),
    }


def _class_distribution(labels: np.ndarray) -> dict[str, int]:
    return {
        class_name: int(np.sum(labels == class_name))
        for class_name in CANONICAL_CLASSES
    }


def _print_summary(
    dataset: FeatureDataset,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    comparison: pd.DataFrame,
    selected_model_name: str,
) -> None:
    train_cats = sorted(np.unique(dataset.groups[train_indices]).tolist())
    test_cats = sorted(np.unique(dataset.groups[test_indices]).tolist())

    print("\n" + "=" * 72)
    print("PetSpeak training summary")
    print("=" * 72)
    print(f"Valid audio files: {len(dataset)}")
    print(f"Distinct cats: {np.unique(dataset.groups).size}")
    print(f"Class distribution: {_class_distribution(dataset.labels)}")
    print(f"Training recordings: {len(train_indices)}")
    print(f"Test recordings: {len(test_indices)}")
    print(f"Training cats ({len(train_cats)}): {', '.join(train_cats)}")
    print(f"Test cats ({len(test_cats)}): {', '.join(test_cats)}")
    print("\nModel performance (selection uses grouped CV macro F1 only):")
    for row in comparison.itertuples(index=False):
        print(
            f"- {row.model_name}: CV macro F1={row.best_cv_macro_f1:.4f}, "
            f"test macro F1={row.test_f1_macro:.4f}, "
            f"test balanced accuracy={row.test_balanced_accuracy:.4f}"
        )
    print(f"\nSelected best model: {selected_model_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved model metadata: {MODEL_METADATA_PATH}")
    print(f"Saved reports: {REPORTS_DIR}")
    print("=" * 72)


def train_project(
    metadata_path: Path = METADATA_PATH,
    cache_path: Path = FEATURE_CACHE_PATH,
    force_features: bool = False,
) -> None:
    """Run the complete leakage-safe model development workflow."""
    ensure_project_directories()
    set_reproducible_seed(RANDOM_STATE)

    metadata = load_metadata(metadata_path)
    dataset = build_feature_dataset(
        metadata,
        cache_path=cache_path,
        force_rebuild=force_features,
    )
    if set(dataset.labels) != set(CANONICAL_CLASSES):
        raise ValueError(
            "All three canonical classes must be present after feature extraction."
        )

    split = grouped_train_test_split(dataset.labels, dataset.groups)
    train_indices = split.train_indices
    test_indices = split.test_indices
    if set(dataset.groups[train_indices]).intersection(dataset.groups[test_indices]):
        raise RuntimeError("Group leakage detected between training and test cats.")

    train_features = dataset.features[train_indices]
    train_labels = dataset.labels[train_indices]
    train_groups = dataset.groups[train_indices]
    test_features = dataset.features[test_indices]
    test_labels = dataset.labels[test_indices]

    cv = make_grouped_cv(train_labels, train_groups)
    searches = model_searches(cv)
    fitted_searches: dict[str, GridSearchCV] = {}
    test_predictions: dict[str, np.ndarray] = {}
    test_metrics: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []

    for model_name, search in searches.items():
        LOGGER.info("Tuning %s with grouped cross-validation.", model_name)
        search.fit(train_features, train_labels, groups=train_groups)
        predictions = np.asarray(search.best_estimator_.predict(test_features), dtype=str)
        metrics = evaluate_predictions(test_labels, predictions)
        fitted_searches[model_name] = search
        test_predictions[model_name] = predictions
        test_metrics[model_name] = metrics
        comparison_rows.append(_comparison_row(model_name, search, metrics))
        LOGGER.info(
            "%s: best grouped CV macro F1=%.4f; held-out macro F1=%.4f",
            model_name,
            search.best_score_,
            metrics["f1_macro"],
        )

    # The held-out test metrics are reported but never used for model selection.
    selected_model_name = max(
        fitted_searches,
        key=lambda name: fitted_searches[name].best_score_,
    )
    selected_search = fitted_searches[selected_model_name]
    selected_model = selected_search.best_estimator_
    selected_predictions = test_predictions[selected_model_name]
    selected_metrics = test_metrics[selected_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "best_cv_macro_f1", ascending=False
    )
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    save_classification_report(
        test_labels,
        selected_predictions,
        CLASSIFICATION_REPORT_PATH,
        model_name=selected_model_name,
        extra_metrics=selected_metrics,
    )
    plot_confusion_matrix(
        test_labels,
        selected_predictions,
        CONFUSION_MATRIX_PATH,
    )
    plot_class_distribution(dataset.labels, CLASS_DISTRIBUTION_PATH)

    split_manifest = {
        "random_state": RANDOM_STATE,
        "train_paths": dataset.paths[train_indices].tolist(),
        "test_paths": dataset.paths[test_indices].tolist(),
        "train_cats": sorted(np.unique(dataset.groups[train_indices]).tolist()),
        "test_cats": sorted(np.unique(dataset.groups[test_indices]).tolist()),
    }
    write_json(SPLIT_MANIFEST_PATH, split_manifest)

    model_metadata = {
        "project": "PetSpeak — Cat Vocalization Context Classifier",
        "model_name": selected_model_name,
        "model_path": str(MODEL_PATH.relative_to(MODEL_PATH.parents[1])),
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "highest grouped cross-validation macro F1 on training cats",
        "best_cv_macro_f1": float(selected_search.best_score_),
        "best_parameters": selected_search.best_params_,
        "evaluation_metrics": selected_metrics,
        "class_labels": list(CANONICAL_CLASSES),
        "preprocessing_config": PREPROCESSING_CONFIG,
        "feature_config": FEATURE_CONFIG,
        "data_summary": {
            "valid_audio_files": len(dataset),
            "distinct_cats": int(np.unique(dataset.groups).size),
            "class_distribution": _class_distribution(dataset.labels),
            "train_recordings": int(len(train_indices)),
            "test_recordings": int(len(test_indices)),
            "train_cats": split_manifest["train_cats"],
            "test_cats": split_manifest["test_cats"],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "librosa": librosa.__version__,
            "platform": platform.platform(),
        },
    }
    write_json(MODEL_METADATA_PATH, model_metadata)
    _print_summary(
        dataset,
        train_indices,
        test_indices,
        comparison,
        selected_model_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare PetSpeak context classifiers."
    )
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--cache", type=Path, default=FEATURE_CACHE_PATH)
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Ignore any existing feature cache and rebuild it.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    arguments = parse_args()
    try:
        train_project(
            metadata_path=arguments.metadata,
            cache_path=arguments.cache,
            force_features=arguments.force_features,
        )
    except Exception as exc:
        LOGGER.exception("Training failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
