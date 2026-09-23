from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# backend/feature_extraction/config.py
FEATURE_EXTRACTION_DIR = Path(__file__).resolve().parent

BACKEND_DIR = FEATURE_EXTRACTION_DIR.parent

PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"

GESTURES_DIR = DATA_DIR / "gestures"


# ============================================================
# SAMPLE FILES
# ============================================================

GESTURE_METADATA_FILENAME = "gesture.json"

SAMPLE_METADATA_FILENAME = "sample.json"

SOURCE_VIDEO_FILENAME = "original.webm"

FEATURES_FILENAME = "features.npz"

FEATURES_METADATA_FILENAME = "features.json"


# ============================================================
# FEATURE EXTRACTION
# ============================================================

FEATURE_EXTRACTION_VERSION = "1.0"

SUPPORTED_GESTURE_TYPES = {
    "hand",
    "body",
}


# ============================================================
# MEDIAPIPE
# ============================================================

HAND_LANDMARK_COUNT = 21

POSE_LANDMARK_COUNT = 33


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

EXTRACT_LANDMARKS = True

EXTRACT_DISTANCES = True

EXTRACT_VELOCITIES = True

EXTRACT_ACCELERATIONS = True


# ============================================================
# PROCESSING
# ============================================================

# Nesta primeira versão:
# - não fazemos smoothing
# - não fazemos interpolação
# - não fazemos normalização adicional
# - não fazemos resampling temporal

ENABLE_SMOOTHING = False

ENABLE_INTERPOLATION = False

ENABLE_NORMALIZATION = False

ENABLE_TEMPORAL_RESAMPLING = False