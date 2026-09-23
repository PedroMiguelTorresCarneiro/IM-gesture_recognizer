from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.feature_extraction.config import (
    FEATURE_EXTRACTION_VERSION,
    FEATURES_FILENAME,
    FEATURES_METADATA_FILENAME,
)


# ============================================================
# NPZ WRITER
# ============================================================

def write_features_npz(
    sample_dir: Path,
    timestamps: np.ndarray,
    landmarks: np.ndarray,
    distances: np.ndarray,
    distance_pairs: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    accelerations: np.ndarray,
    acceleration_magnitudes: np.ndarray,
) -> Path:
    """
    Guarda as features num ficheiro NumPy comprimido.
    """

    sample_dir = Path(
        sample_dir
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        sample_dir
        / FEATURES_FILENAME
    )

    np.savez_compressed(
        output_path,

        timestamps=np.asarray(
            timestamps,
            dtype=np.float64,
        ),

        landmarks=np.asarray(
            landmarks,
            dtype=np.float32,
        ),

        distances=np.asarray(
            distances,
            dtype=np.float32,
        ),

        distance_pairs=np.asarray(
            distance_pairs,
            dtype=np.int32,
        ),

        velocities=np.asarray(
            velocities,
            dtype=np.float32,
        ),

        speeds=np.asarray(
            speeds,
            dtype=np.float32,
        ),

        accelerations=np.asarray(
            accelerations,
            dtype=np.float32,
        ),

        acceleration_magnitudes=np.asarray(
            acceleration_magnitudes,
            dtype=np.float32,
        ),
    )

    return output_path


# ============================================================
# METADATA WRITER
# ============================================================

def write_features_metadata(
    sample_dir: Path,
    *,
    gesture_name: str,
    gesture_type: str,
    label: str,
    sample_id: str,
    source_video: str,
    fps: float,
    frame_count: int,
    duration_seconds: float,
    clip_start_ms: int,
    clip_end_ms: int,
    clip_duration_ms: int,
    landmark_count: int,
    distance_count: int,
    landmarker: str,
    valid_landmark_frames: int,
    missing_landmark_frames: int,
) -> Path:
    """
    Guarda metadata legível da extração.
    """

    sample_dir = Path(
        sample_dir
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        sample_dir
        / FEATURES_METADATA_FILENAME
    )

    metadata: dict[str, Any] = {

        "version": (
            FEATURE_EXTRACTION_VERSION
        ),

        "gesture": (
            gesture_name
        ),

        "gesture_type": (
            gesture_type
        ),

        "label": (
            label
        ),

        "sample_id": (
            sample_id
        ),

        "source": (
            source_video
        ),

        "clip": {
            "start_ms": int(
                clip_start_ms
            ),
            "end_ms": int(
                clip_end_ms
            ),
            "requested_duration_ms": int(
                clip_duration_ms
            ),
        },

        "fps": float(
            fps
        ),

        "frame_count": int(
            frame_count
        ),

        "extracted_duration_seconds": float(
            duration_seconds
        ),

        "landmarker": (
            landmarker
        ),

        "landmark_count": int(
            landmark_count
        ),

        "distance_count": int(
            distance_count
        ),

        "valid_landmark_frames": int(
            valid_landmark_frames
        ),

        "missing_landmark_frames": int(
            missing_landmark_frames
        ),

        "features": {
            "landmarks": True,
            "distances": True,
            "velocities": True,
            "speeds": True,
            "accelerations": True,
            "acceleration_magnitudes": True,
        },

        "processing": {
            "smoothing": False,
            "interpolation": False,
            "normalization": False,
            "temporal_resampling": False,
        },
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# COMPLETE WRITER
# ============================================================

def write_feature_extraction(
    sample_dir: Path,
    *,
    timestamps: np.ndarray,
    landmarks: np.ndarray,
    distances: np.ndarray,
    distance_pairs: list[
        tuple[int, int]
    ],
    velocities: np.ndarray,
    speeds: np.ndarray,
    accelerations: np.ndarray,
    acceleration_magnitudes: np.ndarray,
    gesture_name: str,
    gesture_type: str,
    label: str,
    sample_id: str,
    source_video: str,
    fps: float,
    frame_count: int,
    duration_seconds: float,
    clip_start_ms: int,
    clip_end_ms: int,
    clip_duration_ms: int,
    landmarker: str,
) -> dict[str, Path]:
    """
    Guarda:

        features.npz
        features.json
    """

    sample_dir = Path(
        sample_dir
    )

    landmark_count = (
        landmarks.shape[1]
        if landmarks.ndim == 3
        else 0
    )

    distance_count = (
        distances.shape[1]
        if distances.ndim == 2
        else 0
    )

    # --------------------------------------------------------
    # Detection statistics
    # --------------------------------------------------------

    valid_frame_mask = (
        np.isfinite(
            landmarks
        )
        .all(axis=2)
        .any(axis=1)
    )

    valid_landmark_frames = int(
        valid_frame_mask.sum()
    )

    missing_landmark_frames = int(
        frame_count
        - valid_landmark_frames
    )

    # --------------------------------------------------------
    # NPZ
    # --------------------------------------------------------

    npz_path = write_features_npz(
        sample_dir=sample_dir,

        timestamps=timestamps,

        landmarks=landmarks,

        distances=distances,

        distance_pairs=np.asarray(
            distance_pairs,
            dtype=np.int32,
        ),

        velocities=velocities,

        speeds=speeds,

        accelerations=accelerations,

        acceleration_magnitudes=(
            acceleration_magnitudes
        ),
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    metadata_path = (
        write_features_metadata(

            sample_dir=sample_dir,

            gesture_name=(
                gesture_name
            ),

            gesture_type=(
                gesture_type
            ),

            label=label,

            sample_id=sample_id,

            source_video=(
                source_video
            ),

            fps=fps,

            frame_count=(
                frame_count
            ),

            duration_seconds=(
                duration_seconds
            ),

            clip_start_ms=(
                clip_start_ms
            ),

            clip_end_ms=(
                clip_end_ms
            ),

            clip_duration_ms=(
                clip_duration_ms
            ),

            landmark_count=(
                landmark_count
            ),

            distance_count=(
                distance_count
            ),

            landmarker=(
                landmarker
            ),

            valid_landmark_frames=(
                valid_landmark_frames
            ),

            missing_landmark_frames=(
                missing_landmark_frames
            ),
        )
    )

    return {
        "features": (
            npz_path
        ),
        "metadata": (
            metadata_path
        ),
    }