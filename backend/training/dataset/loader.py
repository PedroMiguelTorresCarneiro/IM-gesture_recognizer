from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.training.config import (
    FEATURES_FILENAME,
    FEATURES_METADATA_FILENAME,
    GESTURE_METADATA_FILENAME,
    GESTURES_DIR,
    SAMPLE_METADATA_FILENAME,
    SUPPORTED_SAMPLE_TYPES,
)


# ============================================================
# REQUIRED FEATURE ARRAYS
# ============================================================

REQUIRED_FEATURE_KEYS = (
    "timestamps",
    "landmarks",
    "distances",
    "distance_pairs",
    "velocities",
    "speeds",
    "accelerations",
    "acceleration_magnitudes",
)


# ============================================================
# EXCEPTIONS
# ============================================================

class DatasetLoadError(RuntimeError):
    """
    Raised when a gesture dataset or sample cannot be loaded safely.
    """

    pass


# ============================================================
# SAMPLE MODEL
# ============================================================

@dataclass
class TrainingSample:
    """
    Represents one recorded gesture sample prepared for training.

    This object still contains the original variable-length temporal
    feature arrays.

    No temporal aggregation or fixed-size representation is created
    at this stage.
    """

    sample_id: str

    gesture_id: str
    gesture_name: str

    target: str

    sample_type: str
    label: int

    sample_dir: Path

    sample_metadata: dict[str, Any]
    features_metadata: dict[str, Any]

    features: dict[str, np.ndarray]

    @property
    def timestamps(self) -> np.ndarray:
        return self.features["timestamps"]

    @property
    def frame_count(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def duration_seconds(self) -> float:
        """
        Duration represented by the extracted feature timestamps.

        The timestamps are relative to the beginning of the selected
        gesture clip.
        """

        if self.frame_count == 0:
            return 0.0

        return float(self.timestamps[-1])

    def get_feature(
        self,
        name: str,
    ) -> np.ndarray:
        """
        Return one stored feature array.
        """

        if name not in self.features:
            raise KeyError(
                f"Feature '{name}' does not exist in sample "
                f"'{self.sample_id}'."
            )

        return self.features[name]


# ============================================================
# JSON READER
# ============================================================

def _read_json(
    path: Path,
) -> dict[str, Any]:
    """
    Read and validate a JSON object.
    """

    if not path.is_file():
        raise DatasetLoadError(
            f"Required JSON file does not exist: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as exc:
        raise DatasetLoadError(
            f"Invalid JSON file: {path}"
        ) from exc

    except OSError as exc:
        raise DatasetLoadError(
            f"Could not read JSON file: {path}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise DatasetLoadError(
            f"Expected a JSON object in: {path}"
        )

    return data


# ============================================================
# FEATURE READER
# ============================================================

def _read_features(
    path: Path,
) -> dict[str, np.ndarray]:
    """
    Load all arrays stored inside features.npz.

    The NPZ file is closed immediately after loading and the returned
    dictionary contains independent NumPy arrays.
    """

    if not path.is_file():
        raise DatasetLoadError(
            f"Required feature file does not exist: {path}"
        )

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as npz_file:

            available_keys = set(
                npz_file.files
            )

            missing_keys = (
                set(REQUIRED_FEATURE_KEYS)
                - available_keys
            )

            if missing_keys:
                missing_text = ", ".join(
                    sorted(missing_keys)
                )

                raise DatasetLoadError(
                    f"Missing required feature arrays in "
                    f"{path}: {missing_text}"
                )

            features = {
                key: np.asarray(
                    npz_file[key]
                ).copy()
                for key in npz_file.files
            }

    except DatasetLoadError:
        raise

    except Exception as exc:
        raise DatasetLoadError(
            f"Could not load feature file: {path}"
        ) from exc

    return features


# ============================================================
# FEATURE VALIDATION
# ============================================================

def _validate_feature_shapes(
    features: dict[str, np.ndarray],
    sample_id: str,
) -> None:
    """
    Validate the temporal dimensions of the stored feature arrays.

    All per-frame arrays must contain the same number of temporal
    observations T.

    distance_pairs is excluded because it describes the mapping
    between distance columns and landmark IDs, not temporal data.
    """

    timestamps = features["timestamps"]

    if timestamps.ndim != 1:
        raise DatasetLoadError(
            f"Sample '{sample_id}' has invalid timestamps shape: "
            f"{timestamps.shape}. Expected (T,)."
        )

    frame_count = timestamps.shape[0]

    if frame_count == 0:
        raise DatasetLoadError(
            f"Sample '{sample_id}' contains zero frames."
        )

    temporal_features = (
        "landmarks",
        "distances",
        "velocities",
        "speeds",
        "accelerations",
        "acceleration_magnitudes",
    )

    for feature_name in temporal_features:

        array = features[
            feature_name
        ]

        if array.ndim == 0:
            raise DatasetLoadError(
                f"Sample '{sample_id}' has invalid scalar feature "
                f"'{feature_name}'."
            )

        if array.shape[0] != frame_count:
            raise DatasetLoadError(
                f"Temporal length mismatch in sample "
                f"'{sample_id}'. "
                f"'timestamps' contains {frame_count} frames but "
                f"'{feature_name}' contains {array.shape[0]}."
            )

    distance_pairs = features[
        "distance_pairs"
    ]

    distances = features[
        "distances"
    ]

    if distance_pairs.ndim != 2:
        raise DatasetLoadError(
            f"Sample '{sample_id}' has invalid distance_pairs "
            f"shape: {distance_pairs.shape}."
        )

    if (
        distance_pairs.shape[1] != 2
    ):
        raise DatasetLoadError(
            f"Sample '{sample_id}' has invalid distance_pairs "
            f"shape: {distance_pairs.shape}. Expected (D, 2)."
        )

    if distances.ndim != 2:
        raise DatasetLoadError(
            f"Sample '{sample_id}' has invalid distances shape: "
            f"{distances.shape}. Expected (T, D)."
        )

    if (
        distances.shape[1]
        != distance_pairs.shape[0]
    ):
        raise DatasetLoadError(
            f"Distance mapping mismatch in sample "
            f"'{sample_id}'. "
            f"'distances' has {distances.shape[1]} columns but "
            f"'distance_pairs' contains "
            f"{distance_pairs.shape[0]} pairs."
        )


# ============================================================
# SAMPLE VALIDATION
# ============================================================

def _validate_sample_metadata(
    sample_metadata: dict[str, Any],
    *,
    gesture_id: str,
    gesture_target: str,
    expected_sample_type: str,
    sample_dir: Path,
) -> None:
    """
    Verify that the sample belongs to the gesture and sample category
    from which it was loaded.
    """

    metadata_sample_id = sample_metadata.get(
        "id"
    )

    if (
        metadata_sample_id
        and metadata_sample_id
        != sample_dir.name
    ):
        raise DatasetLoadError(
            f"Sample directory '{sample_dir.name}' does not match "
            f"sample.json id '{metadata_sample_id}'."
        )

    metadata_gesture_id = (
        sample_metadata.get(
            "gesture_id"
        )
    )

    if (
        metadata_gesture_id
        != gesture_id
    ):
        raise DatasetLoadError(
            f"Sample '{sample_dir.name}' belongs to gesture "
            f"'{metadata_gesture_id}', expected '{gesture_id}'."
        )

    metadata_sample_type = (
        sample_metadata.get(
            "sample_type"
        )
    )

    if (
        metadata_sample_type
        != expected_sample_type
    ):
        raise DatasetLoadError(
            f"Sample '{sample_dir.name}' has sample_type "
            f"'{metadata_sample_type}', but is stored inside "
            f"'{expected_sample_type}/'."
        )

    metadata_target = (
        sample_metadata.get(
            "target"
        )
    )

    if (
        metadata_target
        and metadata_target
        != gesture_target
    ):
        raise DatasetLoadError(
            f"Sample '{sample_dir.name}' target "
            f"'{metadata_target}' does not match gesture target "
            f"'{gesture_target}'."
        )


# ============================================================
# SAMPLE LOADER
# ============================================================

def _load_sample(
    *,
    gesture_id: str,
    gesture_name: str,
    gesture_target: str,
    sample_type: str,
    sample_dir: Path,
) -> TrainingSample:
    """
    Load one positive or negative sample.
    """

    if (
        sample_type
        not in SUPPORTED_SAMPLE_TYPES
    ):
        raise DatasetLoadError(
            f"Unsupported sample type: {sample_type}"
        )

    sample_metadata_path = (
        sample_dir
        / SAMPLE_METADATA_FILENAME
    )

    features_metadata_path = (
        sample_dir
        / FEATURES_METADATA_FILENAME
    )

    features_path = (
        sample_dir
        / FEATURES_FILENAME
    )

    sample_metadata = _read_json(
        sample_metadata_path
    )

    features_metadata = _read_json(
        features_metadata_path
    )

    _validate_sample_metadata(
        sample_metadata,
        gesture_id=gesture_id,
        gesture_target=gesture_target,
        expected_sample_type=sample_type,
        sample_dir=sample_dir,
    )

    features = _read_features(
        features_path
    )

    _validate_feature_shapes(
        features,
        sample_id=sample_dir.name,
    )

    label = SUPPORTED_SAMPLE_TYPES[
        sample_type
    ]

    return TrainingSample(
        sample_id=sample_dir.name,

        gesture_id=gesture_id,
        gesture_name=gesture_name,

        target=gesture_target,

        sample_type=sample_type,
        label=label,

        sample_dir=sample_dir,

        sample_metadata=sample_metadata,
        features_metadata=features_metadata,

        features=features,
    )


# ============================================================
# GESTURE DATASET LOADER
# ============================================================

def load_gesture_samples(
    gesture_id: str,
) -> list[TrainingSample]:
    """
    Load all positive and negative training samples for one gesture.

    Example:

        samples = load_gesture_samples(
            "swipe-left"
        )

    The returned samples still contain variable-length temporal
    sequences.

    Fixed-size representations are created later by the
    representation module.
    """

    gesture_dir = (
        GESTURES_DIR
        / gesture_id
    )

    if not gesture_dir.is_dir():
        raise DatasetLoadError(
            f"Gesture does not exist: {gesture_id}"
        )

    gesture_metadata_path = (
        gesture_dir
        / GESTURE_METADATA_FILENAME
    )

    gesture_metadata = _read_json(
        gesture_metadata_path
    )

    metadata_gesture_id = (
        gesture_metadata.get(
            "id"
        )
    )

    if (
        metadata_gesture_id
        != gesture_id
    ):
        raise DatasetLoadError(
            f"Gesture directory '{gesture_id}' does not match "
            f"gesture.json id '{metadata_gesture_id}'."
        )

    gesture_name = (
        gesture_metadata.get(
            "name",
            gesture_id,
        )
    )

    gesture_target = (
        gesture_metadata.get(
            "target"
        )
    )

    if gesture_target not in {
        "hand",
        "body",
    }:
        raise DatasetLoadError(
            f"Gesture '{gesture_id}' has unsupported target: "
            f"{gesture_target}"
        )

    samples: list[
        TrainingSample
    ] = []

    for sample_type in (
        "positive",
        "negative",
    ):

        sample_root = (
            gesture_dir
            / sample_type
        )

        if not sample_root.exists():
            continue

        if not sample_root.is_dir():
            raise DatasetLoadError(
                f"Expected directory: {sample_root}"
            )

        sample_dirs = sorted(
            path
            for path in sample_root.iterdir()
            if path.is_dir()
        )

        for sample_dir in sample_dirs:

            sample = _load_sample(
                gesture_id=gesture_id,
                gesture_name=gesture_name,
                gesture_target=gesture_target,
                sample_type=sample_type,
                sample_dir=sample_dir,
            )

            samples.append(
                sample
            )

    if not samples:
        raise DatasetLoadError(
            f"No training samples found for gesture "
            f"'{gesture_id}'."
        )

    return samples