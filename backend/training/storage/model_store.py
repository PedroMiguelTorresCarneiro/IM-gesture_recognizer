from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.tree import export_text

from backend.training.config import (
    MODELS_DIR,
    RANDOM_STATE,
    TEMPORAL_AGGREGATION,
    TEMPORAL_PYRAMID_LEVELS,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)


# ============================================================
# FILENAMES
# ============================================================

MODEL_FILENAME = "model.joblib"

METADATA_FILENAME = "metadata.json"

FEATURE_NAMES_FILENAME = (
    "feature_names.json"
)

METRICS_FILENAME = "metrics.json"

FEATURE_IMPORTANCE_FILENAME = (
    "feature_importance.json"
)

DECISION_RULES_FILENAME = (
    "decision_rules.txt"
)

VALIDATION_SAMPLES_FILENAME = (
    "validation_samples.json"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelStoreError(
    RuntimeError
):
    """
    Raised when a trained model or its metadata cannot be
    persisted or loaded safely.
    """

    pass


# ============================================================
# PATH HELPERS
# ============================================================

def _validate_path_component(
    value: str,
    field_name: str,
) -> None:
    """
    Prevent path traversal when building model directories.
    """

    if not value:

        raise ModelStoreError(
            f"{field_name} cannot be empty."
        )


    path = Path(
        value
    )


    if (
        path.name != value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):

        raise ModelStoreError(
            f"Invalid {field_name}: "
            f"'{value}'."
        )


def get_model_dir(
    gesture_id: str,
    model_name: str,
) -> Path:
    """
    Return the directory containing the latest stored model.

    Example:

        data/models/swipe-left/decision_tree/
    """

    _validate_path_component(
        gesture_id,
        "gesture_id",
    )

    _validate_path_component(
        model_name,
        "model_name",
    )


    return (
        MODELS_DIR
        / gesture_id
        / model_name
    )


# ============================================================
# JSON HELPERS
# ============================================================

def _write_json(
    path: Path,
    data: Any,
) -> None:
    """
    Write JSON using a temporary file followed by replace.

    This reduces the chance of leaving a partially written
    metadata file.
    """

    temporary_path = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )


    try:

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4,
            )


        temporary_path.replace(
            path
        )

    except OSError as exc:

        if temporary_path.exists():

            temporary_path.unlink(
                missing_ok=True
            )


        raise ModelStoreError(
            f"Could not write "
            f"'{path.name}'."
        ) from exc


def _read_json(
    path: Path,
) -> Any:
    """
    Read one JSON artifact.
    """

    if not path.exists():

        raise ModelStoreError(
            f"Stored training artifact "
            f"'{path.name}' does not exist."
        )


    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise ModelStoreError(
            f"Could not read "
            f"'{path.name}'."
        ) from exc


# ============================================================
# DATASET SIGNATURE
# ============================================================

def build_dataset_signature(
    sample_ids: list[str],
    sample_types: list[str],
) -> str:
    """
    Create a stable signature for the sample set used during
    training.

    This allows us to detect later whether samples were added
    or deleted.

    Note:
        In this V1 the signature represents sample identity
        and label, not the contents of features.npz.
    """

    if (
        len(sample_ids)
        != len(sample_types)
    ):

        raise ModelStoreError(
            "sample_ids and sample_types "
            "must have the same length."
        )


    entries = sorted(
        zip(
            sample_ids,
            sample_types,
        )
    )


    payload = "\n".join(
        f"{sample_type}:{sample_id}"
        for sample_id, sample_type
        in entries
    )


    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def _build_feature_importance(
    model: Any,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """
    Build a human-readable feature importance list when the
    estimator exposes feature_importances_.
    """

    estimator = model.model


    if not hasattr(
        estimator,
        "feature_importances_",
    ):

        return []


    importances = np.asarray(
        estimator.feature_importances_,
        dtype=np.float64,
    )


    if (
        importances.shape[0]
        != len(
            feature_names
        )
    ):

        raise ModelStoreError(
            "Model feature importance count "
            "does not match feature names."
        )


    result = []


    for index, importance in enumerate(
        importances
    ):

        value = float(
            importance
        )


        if value <= 0:

            continue


        result.append(
            {
                "index": index,
                "name": (
                    feature_names[
                        index
                    ]
                ),
                "importance": value,
            }
        )


    result.sort(
        key=lambda item: item[
            "importance"
        ],
        reverse=True,
    )


    return result


# ============================================================
# VALIDATION SAMPLE PREDICTIONS
# ============================================================

def _build_validation_samples(
    result: Any,
) -> list[dict[str, Any]]:
    """
    Store predictions only for the holdout validation samples.

    These samples were not passed to model.fit().
    """

    dataset = result.dataset


    samples = []


    for (
        position,
        dataset_index,
    ) in enumerate(
        result.validation_indices
    ):

        dataset_index = int(
            dataset_index
        )


        true_label = int(
            dataset.y[
                dataset_index
            ]
        )


        predicted_label = int(
            result.validation_predictions[
                position
            ]
        )


        probabilities = (
            result.validation_probabilities[
                position
            ]
        )


        sample_result = {
            "sample_id": (
                dataset.sample_ids[
                    dataset_index
                ]
            ),

            "sample_type": (
                dataset.sample_types[
                    dataset_index
                ]
            ),

            "true_label": (
                true_label
            ),

            "predicted_label": (
                predicted_label
            ),

            "correct": (
                true_label
                == predicted_label
            ),
        }


        if (
            probabilities.shape[0]
            >= 2
        ):

            sample_result[
                "probability_negative"
            ] = float(
                probabilities[0]
            )

            sample_result[
                "probability_positive"
            ] = float(
                probabilities[1]
            )


        samples.append(
            sample_result
        )


    return samples

# ============================================================
# REPRESENTATION METADATA
# ============================================================

def _build_representation_metadata(
    representation_name: str,
) -> dict[str, Any]:
    """
    Build metadata describing the fixed-size representation.
    """

    metadata = {
        "name": representation_name,
    }


    if (
        representation_name
        == "temporal_pyramid"
    ):

        metadata.update(
            {
                "type": (
                    "temporal_pyramid"
                ),

                "levels": list(
                    TEMPORAL_PYRAMID_LEVELS
                ),

                "regions": int(
                    sum(
                        TEMPORAL_PYRAMID_LEVELS
                    )
                ),

                "aggregation": (
                    TEMPORAL_AGGREGATION
                ),
            }
        )


    return metadata


# ============================================================
# SAVE TRAINING RESULT
# ============================================================

def save_training_result(
    result: Any,
) -> dict[str, Any]:
    """
    Persist the latest trained model and its associated
    training information.

    Current directory structure:

        data/models/
        └── <gesture_id>/
            └── <model_name>/
                ├── model.joblib
                ├── metadata.json
                ├── feature_names.json
                ├── metrics.json
                ├── feature_importance.json
                ├── decision_rules.txt
                └── training_samples.json

    Each new training run replaces the previous model for the
    same gesture + classifier.
    """

    dataset = result.dataset

    model = result.model


    model_dir = get_model_dir(
        dataset.gesture_id,
        model.name,
    )


    try:

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as exc:

        raise ModelStoreError(
            "Could not create model "
            "storage directory."
        ) from exc


    trained_at = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


    dataset_signature = (
        build_dataset_signature(
            dataset.sample_ids,
            dataset.sample_types,
        )
    )


    representation_metadata = (
        _build_representation_metadata(
            dataset.representation_name
        )
    )


    feature_importance = (
        _build_feature_importance(
            model,
            dataset.feature_names,
        )
    )


    validation_samples = (
        _build_validation_samples(
            result
        )
    )


    # ========================================================
    # DECISION TREE RULES
    # ========================================================

    decision_rules = ""


    if (
        model.name
        == "decision_tree"
    ):

        try:

            decision_rules = (
                export_text(
                    model.model,
                    feature_names=(
                        dataset.feature_names
                    ),
                )
            )

        except Exception as exc:

            raise ModelStoreError(
                "Could not export "
                "Decision Tree rules."
            ) from exc


    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "schema_version": "1.0",

        "trained_at": trained_at,

        "gesture": {
            "id": (
                dataset.gesture_id
            ),

            "name": (
                dataset.gesture_name
            ),

            "target": (
                dataset.target
            ),
        },

        "model": {
            "name": (
                model.name
            ),
        },

        "representation": (
            representation_metadata
        ),

        "input": {
            "feature_count": (
                dataset.feature_count
            ),

            "dtype": str(
                dataset.X.dtype
            ),
        },

        "labels": {
            "0": "negative",
            "1": "positive",
        },

        "dataset": {
            "samples": (
                dataset.sample_count
            ),

            "positive": (
                dataset.positive_count
            ),

            "negative": (
                dataset.negative_count
            ),

            "sample_ids": (
                list(
                    dataset.sample_ids
                )
            ),

            "sample_types": (
                list(
                    dataset.sample_types
                )
            ),

            "signature": (
                dataset_signature
            ),

            "split": {
                "strategy": (
                    "stratified_holdout"
                ),

                "random_state": (
                    RANDOM_STATE
                ),

                "configured_train_ratio": (
                    TRAIN_RATIO
                ),

                "configured_validation_ratio": (
                    VALIDATION_RATIO
                ),

                "actual_train_ratio": (
                    result.train_ratio
                ),

                "actual_validation_ratio": (
                    result.validation_ratio
                ),

                "training_samples": (
                    result.training_sample_count
                ),

                "training_positive": (
                    result.training_positive_count
                ),

                "training_negative": (
                    result.training_negative_count
                ),

                "validation_samples": (
                    result.validation_sample_count
                ),

                "validation_positive": (
                    result.validation_positive_count
                ),

                "validation_negative": (
                    result.validation_negative_count
                ),

                "training_sample_ids": (
                    result.training_sample_ids
                ),

                "validation_sample_ids": (
                    result.validation_sample_ids
                ),
            },
        },
    }


    # ========================================================
    # MODEL-SPECIFIC METADATA
    # ========================================================

    if (
        model.name
        == "decision_tree"
    ):

        metadata[
            "model"
        ][
            "tree_depth"
        ] = model.tree_depth

        metadata[
            "model"
        ][
            "leaf_count"
        ] = model.leaf_count


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    metrics = {
        "evaluation_type": (
            "holdout_validation"
        ),

        "accuracy": (
            result.accuracy
        ),

        "precision": (
            result.precision
        ),

        "recall": (
            result.recall
        ),

        "f1": (
            result.f1
        ),

        "confusion_matrix": {
            "tn": (
                result.true_negative
            ),

            "fp": (
                result.false_positive
            ),

            "fn": (
                result.false_negative
            ),

            "tp": (
                result.true_positive
            ),
        },

        "warning": (
            "Metrics were calculated on "
            "the holdout validation set, "
            "which was not used to fit "
            "the model."
        ),
    }


    # ========================================================
    # FEATURE NAMES
    # ========================================================

    feature_names_data = {
        "feature_count": (
            dataset.feature_count
        ),

        "feature_names": (
            list(
                dataset.feature_names
            )
        ),
    }


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance_data = {
        "used_feature_count": len(
            feature_importance
        ),

        "used_features": (
            feature_importance
        ),
    }


    # ========================================================
    # WRITE MODEL
    # ========================================================

    model_path = (
        model_dir
        / MODEL_FILENAME
    )


    temporary_model_path = (
        model_dir
        / (
            MODEL_FILENAME
            + ".tmp"
        )
    )


    try:

        joblib.dump(
            model.model,
            temporary_model_path,
        )


        temporary_model_path.replace(
            model_path
        )

    except Exception as exc:

        temporary_model_path.unlink(
            missing_ok=True
        )


        raise ModelStoreError(
            "Could not persist trained model."
        ) from exc


    # ========================================================
    # WRITE JSON ARTIFACTS
    # ========================================================

    _write_json(
        model_dir
        / METADATA_FILENAME,
        metadata,
    )


    _write_json(
        model_dir
        / FEATURE_NAMES_FILENAME,
        feature_names_data,
    )


    _write_json(
        model_dir
        / METRICS_FILENAME,
        metrics,
    )


    _write_json(
        model_dir
        / FEATURE_IMPORTANCE_FILENAME,
        feature_importance_data,
    )


    _write_json(
        model_dir
        / VALIDATION_SAMPLES_FILENAME,
        validation_samples,
    )


    # ========================================================
    # WRITE DECISION RULES
    # ========================================================

    decision_rules_path = (
        model_dir
        / DECISION_RULES_FILENAME
    )


    try:

        decision_rules_path.write_text(
            decision_rules,
            encoding="utf-8",
        )

    except OSError as exc:

        raise ModelStoreError(
            "Could not write "
            "decision rules."
        ) from exc


    # ========================================================
    # RETURN STORED INFORMATION
    # ========================================================

    return load_training_info(
        dataset.gesture_id,
        model.name,
    )


# ============================================================
# CHECK STORED MODEL
# ============================================================

def training_exists(
    gesture_id: str,
    model_name: str,
) -> bool:
    """
    Return True when the required artifacts for a stored
    training run exist.
    """

    model_dir = get_model_dir(
        gesture_id,
        model_name,
    )


    required_files = (
        MODEL_FILENAME,
        METADATA_FILENAME,
        FEATURE_NAMES_FILENAME,
        METRICS_FILENAME,
    )


    return all(
        (
            model_dir
            / filename
        ).exists()

        for filename
        in required_files
    )


# ============================================================
# LOAD TRAINED ESTIMATOR
# ============================================================

def load_estimator(
    gesture_id: str,
    model_name: str,
) -> Any:
    """
    Load the original persisted scikit-learn estimator.
    """

    model_dir = get_model_dir(
        gesture_id,
        model_name,
    )


    model_path = (
        model_dir
        / MODEL_FILENAME
    )


    if not model_path.exists():

        raise ModelStoreError(
            "Stored model does not exist."
        )


    try:

        return joblib.load(
            model_path
        )

    except Exception as exc:

        raise ModelStoreError(
            "Could not load stored model."
        ) from exc


# ============================================================
# LOAD TRAINING INFORMATION
# ============================================================

def load_training_info(
    gesture_id: str,
    model_name: str,
) -> dict[str, Any]:
    """
    Load the information required to display the last training
    result in the frontend.

    The model itself is NOT loaded by this function.
    """

    model_dir = get_model_dir(
        gesture_id,
        model_name,
    )


    if not training_exists(
        gesture_id,
        model_name,
    ):

        raise ModelStoreError(
            "No stored training exists "
            f"for gesture '{gesture_id}' "
            f"and model '{model_name}'."
        )


    metadata = _read_json(
        model_dir
        / METADATA_FILENAME
    )


    metrics = _read_json(
        model_dir
        / METRICS_FILENAME
    )


    feature_importance = (
        _read_json(
            model_dir
            / FEATURE_IMPORTANCE_FILENAME
        )
        if (
            model_dir
            / FEATURE_IMPORTANCE_FILENAME
        ).exists()
        else {
            "used_feature_count": 0,
            "used_features": [],
        }
    )


    validation_samples = (
        _read_json(
            model_dir
            / VALIDATION_SAMPLES_FILENAME
        )
        if (
            model_dir
            / VALIDATION_SAMPLES_FILENAME
        ).exists()
        else []
    )


    rules_path = (
        model_dir
        / DECISION_RULES_FILENAME
    )


    if rules_path.exists():

        try:

            decision_rules = (
                rules_path.read_text(
                    encoding="utf-8",
                )
            )

        except OSError as exc:

            raise ModelStoreError(
                "Could not read "
                "decision rules."
            ) from exc

    else:

        decision_rules = ""


    model_metadata = (
        metadata.get(
            "model",
            {},
        )
    )


    dataset_metadata = (
        metadata.get(
            "dataset",
            {},
        )
    )
    
    split_metadata = (
        dataset_metadata.get(
            "split",
            {},
        )
    )


    return {
        "status": "stored",

        "trained_at": (
            metadata.get(
                "trained_at"
            )
        ),

        "gesture": (
            metadata.get(
                "gesture",
                {},
            )
        ),

        "model": {
            "name": (
                model_metadata.get(
                    "name"
                )
            ),

            "tree_depth": (
                model_metadata.get(
                    "tree_depth"
                )
            ),

            "leaf_count": (
                model_metadata.get(
                    "leaf_count"
                )
            ),
        },

        "representation": (
            metadata.get(
                "representation",
                {},
            )
        ),

        "dataset": {
            "samples": (
                dataset_metadata.get(
                    "samples"
                )
            ),

            "positive": (
                dataset_metadata.get(
                    "positive"
                )
            ),

            "negative": (
                dataset_metadata.get(
                    "negative"
                )
            ),

            "features": (
                metadata
                .get(
                    "input",
                    {},
                )
                .get(
                    "feature_count"
                )
            ),

            "signature": (
                dataset_metadata.get(
                    "signature"
                )
            ),
            
            "split": {
                "strategy": (
                    split_metadata.get(
                        "strategy"
                    )
                ),

                "random_state": (
                    split_metadata.get(
                        "random_state"
                    )
                ),

                "train_ratio": (
                    split_metadata.get(
                        "actual_train_ratio"
                    )
                ),

                "validation_ratio": (
                    split_metadata.get(
                        "actual_validation_ratio"
                    )
                ),

                "training_samples": (
                    split_metadata.get(
                        "training_samples"
                    )
                ),

                "training_positive": (
                    split_metadata.get(
                        "training_positive"
                    )
                ),

                "training_negative": (
                    split_metadata.get(
                        "training_negative"
                    )
                ),

                "validation_samples": (
                    split_metadata.get(
                        "validation_samples"
                    )
                ),

                "validation_positive": (
                    split_metadata.get(
                        "validation_positive"
                    )
                ),

                "validation_negative": (
                    split_metadata.get(
                        "validation_negative"
                    )
                ),
            },
        },

        "evaluation": {
            "type": (
                metrics.get(
                    "evaluation_type"
                )
            ),

            "accuracy": (
                metrics.get(
                    "accuracy"
                )
            ),

            "precision": (
                metrics.get(
                    "precision"
                )
            ),

            "recall": (
                metrics.get(
                    "recall"
                )
            ),

            "f1": (
                metrics.get(
                    "f1"
                )
            ),

            "confusion_matrix": (
                metrics.get(
                    "confusion_matrix",
                    {},
                )
            ),

            "warning": (
                metrics.get(
                    "warning"
                )
            ),
        },

        "decision_tree": {
            "rules": (
                decision_rules
            ),

            "used_feature_count": (
                feature_importance.get(
                    "used_feature_count",
                    0,
                )
            ),

            "used_features": (
                feature_importance.get(
                    "used_features",
                    [],
                )
            ),
        },

        "validation_samples": (
            validation_samples
        ),
    }
    
# ============================================================
# LOAD FEATURE NAMES
# ============================================================

def load_feature_names(
    gesture_id: str,
    model_name: str,
) -> list[str]:
    """
    Load the ordered feature names used when the stored model
    was trained.

    The order is critical because classifier input columns must
    match the exact training representation.
    """

    model_dir = get_model_dir(
        gesture_id,
        model_name,
    )


    data = _read_json(
        model_dir
        / FEATURE_NAMES_FILENAME
    )


    feature_names = (
        data.get(
            "feature_names"
        )
    )


    if not isinstance(
        feature_names,
        list,
    ):

        raise ModelStoreError(
            "Stored feature names are invalid."
        )


    feature_count = (
        data.get(
            "feature_count"
        )
    )


    if (
        feature_count
        is not None
        and int(feature_count)
        != len(feature_names)
    ):

        raise ModelStoreError(
            "Stored feature count does not "
            "match feature_names."
        )


    return [
        str(name)
        for name in feature_names
    ]