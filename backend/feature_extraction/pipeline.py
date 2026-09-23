from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import av
import numpy as np

from backend.feature_extraction.config import (
    FEATURES_FILENAME,
    FEATURES_METADATA_FILENAME,
    GESTURE_METADATA_FILENAME,
    GESTURES_DIR,
    SAMPLE_METADATA_FILENAME,
    SOURCE_VIDEO_FILENAME,
    SUPPORTED_GESTURE_TYPES,
)

from backend.feature_extraction.extractors.hand import (
    HandExtractor,
)

from backend.feature_extraction.extractors.pose import (
    PoseExtractor,
)

from backend.feature_extraction.features.distances import (
    calculate_sequence_distances,
)

from backend.feature_extraction.features.kinematics import (
    calculate_kinematics,
)

from backend.feature_extraction.storage.writer import (
    write_feature_extraction,
)


# ============================================================
# JSON
# ============================================================

def load_json(
    path: Path,
) -> dict[str, Any]:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# GESTURE TYPE
# ============================================================

def get_gesture_type(
    gesture_metadata: dict[str, Any],
) -> str:

    gesture_type = (
        gesture_metadata.get("target")
        or gesture_metadata.get("type")
    )

    if gesture_type is None:
        raise ValueError(
            "gesture.json does not contain "
            "'target' or 'type'."
        )

    gesture_type = str(
        gesture_type
    ).strip().lower()

    if (
        gesture_type
        not in SUPPORTED_GESTURE_TYPES
    ):
        raise ValueError(
            f"Unsupported gesture type: "
            f"{gesture_type}. "
            f"Supported types: "
            f"{sorted(SUPPORTED_GESTURE_TYPES)}"
        )

    return gesture_type


# ============================================================
# GESTURE NAME
# ============================================================

def get_gesture_name(
    gesture_dir: Path,
    gesture_metadata: dict[str, Any],
) -> str:

    value = gesture_metadata.get(
        "name"
    )

    if value:
        return str(value)

    return gesture_dir.name


# ============================================================
# TRIM
# ============================================================

def get_trim_interval(
    sample_metadata: dict[str, Any],
) -> tuple[int, int]:

    trim = sample_metadata.get(
        "trim"
    )

    if not isinstance(trim, dict):
        raise ValueError(
            "sample.json does not contain "
            "valid trim metadata."
        )

    start_ms = trim.get(
        "start_ms"
    )

    end_ms = trim.get(
        "end_ms"
    )

    if start_ms is None:
        raise ValueError(
            "sample.json trim.start_ms is missing."
        )

    if end_ms is None:
        raise ValueError(
            "sample.json trim.end_ms is missing."
        )

    start_ms = int(
        start_ms
    )

    end_ms = int(
        end_ms
    )

    if start_ms < 0:
        raise ValueError(
            "trim.start_ms cannot be negative."
        )

    if end_ms <= start_ms:
        raise ValueError(
            "trim.end_ms must be greater than "
            "trim.start_ms."
        )

    return (
        start_ms,
        end_ms,
    )


# ============================================================
# SOURCE VIDEO
# ============================================================

def get_source_video_path(
    sample_dir: Path,
    sample_metadata: dict[str, Any],
) -> Path:

    source_filename = (
        sample_metadata.get("video")
        or SOURCE_VIDEO_FILENAME
    )

    video_path = (
        sample_dir
        / str(source_filename)
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    return video_path


# ============================================================
# EXTRACTOR FACTORY
# ============================================================

def create_extractor(
    gesture_type: str,
):

    if gesture_type == "hand":
        return (
            HandExtractor(),
            "mediapipe_hands",
        )

    if gesture_type == "body":
        return (
            PoseExtractor(),
            "mediapipe_pose",
        )

    raise ValueError(
        f"No extractor available for "
        f"gesture type: {gesture_type}"
    )


# ============================================================
# EFFECTIVE FPS
# ============================================================

def calculate_effective_fps(
    timestamps: np.ndarray,
) -> float:
    """
    Calcula uma estimativa do FPS efetivo com base
    nos timestamps reais dos frames.

    Não é utilizada para calcular velocidade ou
    aceleração.

    Serve apenas como metadata.
    """

    if timestamps.size < 2:
        return 0.0

    deltas = np.diff(
        timestamps
    )

    valid_deltas = deltas[
        np.isfinite(deltas)
        & (deltas > 0)
    ]

    if valid_deltas.size == 0:
        return 0.0

    median_delta = float(
        np.median(
            valid_deltas
        )
    )

    if median_delta <= 0:
        return 0.0

    return (
        1.0
        / median_delta
    )


# ============================================================
# VIDEO LANDMARK EXTRACTION
# ============================================================

def extract_video_landmarks(
    video_path: Path,
    gesture_type: str,
    start_ms: int,
    end_ms: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    str,
]:
    """
    Extrai apenas os frames dentro do clip confirmado.

    Utiliza os Presentation Timestamps reais do vídeo
    através de PyAV.

    Isto permite trabalhar corretamente com WebM de
    frame rate variável.

    Os timestamps devolvidos começam em 0 relativamente
    ao primeiro frame efetivamente incluído no clip.
    """

    video_path = Path(
        video_path
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    extractor, landmarker_name = (
        create_extractor(
            gesture_type
        )
    )

    landmarks_frames: list[
        np.ndarray
    ] = []

    source_timestamps: list[
        float
    ] = []

    first_video_timestamp: (
        float | None
    ) = None

    first_clip_timestamp: (
        float | None
    ) = None

    container = None

    try:

        container = av.open(
            str(video_path)
        )

        if not container.streams.video:
            raise RuntimeError(
                f"No video stream found: "
                f"{video_path}"
            )

        video_stream = (
            container.streams.video[0]
        )

        for video_frame in container.decode(
            video_stream
        ):

            # -----------------------------------------------
            # Real timestamp from container
            # -----------------------------------------------

            if video_frame.time is None:
                continue

            absolute_time_s = float(
                video_frame.time
            )

            # Normalizamos o início do ficheiro para 0.
            #
            # Desta forma o tempo corresponde à timeline
            # que o elemento <video> no browser apresenta.
            if first_video_timestamp is None:
                first_video_timestamp = (
                    absolute_time_s
                )

            source_time_s = (
                absolute_time_s
                - first_video_timestamp
            )

            source_time_ms = (
                source_time_s
                * 1000.0
            )

            # -----------------------------------------------
            # Before selected clip
            # -----------------------------------------------

            if source_time_ms < start_ms:
                continue

            # -----------------------------------------------
            # After selected clip
            # -----------------------------------------------

            if source_time_ms >= end_ms:
                break

            # -----------------------------------------------
            # Convert PyAV frame -> OpenCV BGR array
            # -----------------------------------------------

            frame = (
                video_frame.to_ndarray(
                    format="bgr24"
                )
            )

            # -----------------------------------------------
            # MediaPipe
            # -----------------------------------------------

            frame_landmarks = (
                extractor.extract(
                    frame
                )
            )

            landmarks_frames.append(
                frame_landmarks
            )

            source_timestamps.append(
                source_time_s
            )

            if first_clip_timestamp is None:
                first_clip_timestamp = (
                    source_time_s
                )

    finally:

        extractor.close()

        if container is not None:
            container.close()

    # --------------------------------------------------------
    # Validate clip
    # --------------------------------------------------------

    if not landmarks_frames:

        raise RuntimeError(
            "No video frames were found inside "
            f"the selected clip interval "
            f"{start_ms}ms - {end_ms}ms."
        )

    landmarks = np.stack(
        landmarks_frames,
        axis=0,
    ).astype(
        np.float32
    )

    source_timestamps_array = (
        np.asarray(
            source_timestamps,
            dtype=np.float64,
        )
    )

    # --------------------------------------------------------
    # Local gesture timeline
    # --------------------------------------------------------

    timestamps = (
        source_timestamps_array
        - source_timestamps_array[0]
    )

    # --------------------------------------------------------
    # Effective FPS - metadata only
    # --------------------------------------------------------

    effective_fps = (
        calculate_effective_fps(
            timestamps
        )
    )

    return (
        landmarks,
        timestamps,
        effective_fps,
        landmarker_name,
    )


# ============================================================
# SAMPLE EXTRACTION
# ============================================================

def extract_sample(
    gesture_dir: Path,
    sample_dir: Path,
    label: str,
    *,
    force: bool = False,
) -> dict[str, Any]:

    gesture_dir = Path(
        gesture_dir
    )

    sample_dir = Path(
        sample_dir
    )

    if label not in {
        "positive",
        "negative",
    }:
        raise ValueError(
            "label must be 'positive' "
            "or 'negative'."
        )

    gesture_metadata_path = (
        gesture_dir
        / GESTURE_METADATA_FILENAME
    )

    sample_metadata_path = (
        sample_dir
        / SAMPLE_METADATA_FILENAME
    )

    features_path = (
        sample_dir
        / FEATURES_FILENAME
    )

    features_metadata_path = (
        sample_dir
        / FEATURES_METADATA_FILENAME
    )

    # --------------------------------------------------------
    # Skip
    # --------------------------------------------------------

    if (
        not force
        and features_path.exists()
        and features_metadata_path.exists()
    ):

        return {
            "status": "skipped",
            "label": label,
            "sample_id": (
                sample_dir.name
            ),
            "reason": (
                "Features already exist."
            ),
            "features": str(
                features_path
            ),
            "metadata": str(
                features_metadata_path
            ),
        }

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    gesture_metadata = load_json(
        gesture_metadata_path
    )

    sample_metadata = load_json(
        sample_metadata_path
    )

    gesture_type = (
        get_gesture_type(
            gesture_metadata
        )
    )

    gesture_name = (
        get_gesture_name(
            gesture_dir,
            gesture_metadata,
        )
    )

    sample_id = (
        sample_dir.name
    )

    start_ms, end_ms = (
        get_trim_interval(
            sample_metadata
        )
    )

    clip_duration_ms = (
        end_ms
        - start_ms
    )

    video_path = (
        get_source_video_path(
            sample_dir,
            sample_metadata,
        )
    )

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    (
        landmarks,
        timestamps,
        effective_fps,
        landmarker_name,
    ) = extract_video_landmarks(
        video_path=video_path,
        gesture_type=gesture_type,
        start_ms=start_ms,
        end_ms=end_ms,
    )

    frame_count = int(
        landmarks.shape[0]
    )

    # --------------------------------------------------------
    # Distances
    # --------------------------------------------------------

    (
        distances,
        distance_pairs,
    ) = calculate_sequence_distances(
        landmarks
    )

    # --------------------------------------------------------
    # Kinematics
    # --------------------------------------------------------

    kinematics = (
        calculate_kinematics(
            landmarks=landmarks,
            timestamps=timestamps,
        )
    )

    velocities = (
        kinematics[
            "velocities"
        ]
    )

    speeds = (
        kinematics[
            "speeds"
        ]
    )

    accelerations = (
        kinematics[
            "accelerations"
        ]
    )

    acceleration_magnitudes = (
        kinematics[
            "acceleration_magnitudes"
        ]
    )

    # --------------------------------------------------------
    # Extracted duration
    # --------------------------------------------------------

    if timestamps.size >= 2:

        extracted_duration_seconds = float(
            timestamps[-1]
            - timestamps[0]
        )

    else:

        extracted_duration_seconds = 0.0

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    output = write_feature_extraction(
        sample_dir=sample_dir,

        timestamps=timestamps,

        landmarks=landmarks,

        distances=distances,

        distance_pairs=(
            distance_pairs
        ),

        velocities=(
            velocities
        ),

        speeds=speeds,

        accelerations=(
            accelerations
        ),

        acceleration_magnitudes=(
            acceleration_magnitudes
        ),

        gesture_name=(
            gesture_name
        ),

        gesture_type=(
            gesture_type
        ),

        label=label,

        sample_id=(
            sample_id
        ),

        source_video=(
            video_path.name
        ),

        fps=effective_fps,

        frame_count=(
            frame_count
        ),

        duration_seconds=(
            extracted_duration_seconds
        ),

        clip_start_ms=(
            start_ms
        ),

        clip_end_ms=(
            end_ms
        ),

        clip_duration_ms=(
            clip_duration_ms
        ),

        landmarker=(
            landmarker_name
        ),
    )

    return {
        "status": "processed",

        "gesture": (
            gesture_name
        ),

        "gesture_type": (
            gesture_type
        ),

        "label": label,

        "sample_id": (
            sample_id
        ),

        "fps": (
            effective_fps
        ),

        "frame_count": (
            frame_count
        ),

        "clip_start_ms": (
            start_ms
        ),

        "clip_end_ms": (
            end_ms
        ),

        "clip_duration_ms": (
            clip_duration_ms
        ),

        "duration_seconds": (
            extracted_duration_seconds
        ),

        "features": str(
            output["features"]
        ),

        "metadata": str(
            output["metadata"]
        ),
    }


# ============================================================
# GESTURE EXTRACTION
# ============================================================

def extract_gesture(
    gesture_name: str,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:

    gesture_dir = (
        GESTURES_DIR
        / gesture_name
    )

    if not gesture_dir.exists():
        raise FileNotFoundError(
            f"Gesture not found: "
            f"{gesture_dir}"
        )

    results: list[
        dict[str, Any]
    ] = []

    for label in (
        "positive",
        "negative",
    ):

        label_dir = (
            gesture_dir
            / label
        )

        if not label_dir.exists():
            continue

        sample_dirs = sorted(
            path
            for path in label_dir.iterdir()
            if path.is_dir()
        )

        for sample_dir in sample_dirs:

            try:

                result = extract_sample(
                    gesture_dir=gesture_dir,
                    sample_dir=sample_dir,
                    label=label,
                    force=force,
                )

            except Exception as exc:

                result = {
                    "status": "error",
                    "gesture": (
                        gesture_name
                    ),
                    "label": label,
                    "sample_id": (
                        sample_dir.name
                    ),
                    "error": str(exc),
                }

            results.append(
                result
            )

    return results


# ============================================================
# ALL GESTURES
# ============================================================

def extract_all_gestures(
    *,
    force: bool = False,
) -> dict[
    str,
    list[dict[str, Any]],
]:

    if not GESTURES_DIR.exists():
        raise FileNotFoundError(
            f"Gestures directory not found: "
            f"{GESTURES_DIR}"
        )

    all_results: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    gesture_dirs = sorted(
        path
        for path in GESTURES_DIR.iterdir()
        if path.is_dir()
    )

    for gesture_dir in gesture_dirs:

        gesture_name = (
            gesture_dir.name
        )

        all_results[
            gesture_name
        ] = extract_gesture(
            gesture_name,
            force=force,
        )

    return all_results


# ============================================================
# TERMINAL OUTPUT
# ============================================================

def print_results(
    results: list[dict[str, Any]],
) -> None:

    processed = 0
    skipped = 0
    errors = 0

    for result in results:

        status = result.get(
            "status"
        )

        sample_id = result.get(
            "sample_id",
            "?",
        )

        label = result.get(
            "label",
            "?",
        )

        if status == "processed":

            processed += 1

            fps = result.get(
                "fps",
                0.0,
            )

            print(
                f"[OK] "
                f"{label} / "
                f"{sample_id} "
                f"({result['frame_count']} frames, "
                f"~{fps:.2f} FPS)"
            )

        elif status == "skipped":

            skipped += 1

            print(
                f"[SKIP] "
                f"{label} / "
                f"{sample_id}"
            )

        else:

            errors += 1

            print(
                f"[ERROR] "
                f"{label} / "
                f"{sample_id}: "
                f"{result.get('error')}"
            )

    print()

    print(
        "Summary:"
    )

    print(
        f"  Processed: {processed}"
    )

    print(
        f"  Skipped:   {skipped}"
    )

    print(
        f"  Errors:    {errors}"
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract gesture features "
            "from recorded clips."
        )
    )

    target = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    target.add_argument(
        "--gesture",
        type=str,
        help=(
            "Process a single gesture."
        ),
    )

    target.add_argument(
        "--all",
        action="store_true",
        help=(
            "Process all gestures."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-extract existing features."
        ),
    )

    args = (
        parser.parse_args()
    )

    if args.gesture:

        print()

        print(
            f"Extracting gesture: "
            f"{args.gesture}"
        )

        print()

        results = extract_gesture(
            args.gesture,
            force=args.force,
        )

        print_results(
            results
        )

        return

    if args.all:

        all_results = (
            extract_all_gestures(
                force=args.force
            )
        )

        for (
            gesture_name,
            results,
        ) in all_results.items():

            print()

            print(
                "================================"
            )

            print(
                f"Gesture: {gesture_name}"
            )

            print(
                "================================"
            )

            print_results(
                results
            )


if __name__ == "__main__":
    main()