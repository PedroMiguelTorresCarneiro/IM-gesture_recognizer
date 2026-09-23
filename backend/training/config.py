from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

GESTURES_DIR = (
    DATA_DIR
    / "gestures"
)

MODELS_DIR = (
    DATA_DIR
    / "models"
)


# ============================================================
# FILENAMES
# ============================================================

GESTURE_METADATA_FILENAME = (
    "gesture.json"
)

SAMPLE_METADATA_FILENAME = (
    "sample.json"
)

FEATURES_METADATA_FILENAME = (
    "features.json"
)

FEATURES_FILENAME = (
    "features.npz"
)


# ============================================================
# LABELS
# ============================================================

POSITIVE_LABEL = 1

NEGATIVE_LABEL = 0


SUPPORTED_SAMPLE_TYPES = {
    "positive": POSITIVE_LABEL,
    "negative": NEGATIVE_LABEL,
}


# ============================================================
# FEATURE FAMILIES
# ============================================================

FEATURE_FAMILIES = (
    "landmarks",
    "distances",
    "velocities",
    "speeds",
    "accelerations",
    "acceleration_magnitudes",
)


# ============================================================
# REPRESENTATION
# ============================================================

DEFAULT_REPRESENTATION = (
    "temporal_pyramid"
)


TEMPORAL_PYRAMID_LEVELS = (
    1,
    2,
    4,
)


TEMPORAL_PYRAMID_REGION_COUNT = sum(
    TEMPORAL_PYRAMID_LEVELS
)


TEMPORAL_AGGREGATION = (
    "mean"
)


# ============================================================
# MODELS
# ============================================================

DEFAULT_MODEL = (
    "decision_tree"
)


SUPPORTED_MODELS = (
    "decision_tree",
    "random_forest",
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

TRAIN_RATIO = 0.80

VALIDATION_RATIO = 0.20


# Keep the split reproducible.
#
# Using the same dataset with the same random state will
# generate the same train / validation split.

RANDOM_STATE = 42