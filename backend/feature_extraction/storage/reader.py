from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.feature_extraction.config import (
    FEATURES_FILENAME,
)


# ============================================================
# LOAD FEATURE FILE
# ============================================================

def load_features(
    sample_dir: Path,
) -> dict[str, np.ndarray]:
    """
    Carrega o features.npz de um sample.

    Returns
    -------
    dict[str, np.ndarray]
        Arrays existentes no ficheiro.
    """

    sample_dir = Path(
        sample_dir
    )

    features_path = (
        sample_dir
        / FEATURES_FILENAME
    )

    if not features_path.exists():

        raise FileNotFoundError(
            f"Features file not found: "
            f"{features_path}"
        )

    with np.load(
        features_path,
        allow_pickle=False,
    ) as data:

        return {
            key: data[key].copy()
            for key in data.files
        }


# ============================================================
# LOAD LANDMARK SEQUENCE
# ============================================================

def load_landmark_sequence(
    sample_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Carrega apenas os dados necessários para
    visualizar as landmarks no frontend.

    Returns
    -------
    timestamps:
        shape = (T,)

    landmarks:
        shape = (T, N, 3)
    """

    data = load_features(
        sample_dir
    )

    if "timestamps" not in data:

        raise ValueError(
            "features.npz does not contain "
            "'timestamps'."
        )

    if "landmarks" not in data:

        raise ValueError(
            "features.npz does not contain "
            "'landmarks'."
        )

    timestamps = np.asarray(
        data["timestamps"],
        dtype=np.float64,
    )

    landmarks = np.asarray(
        data["landmarks"],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if timestamps.ndim != 1:

        raise ValueError(
            "timestamps must have shape (T,)."
        )

    if landmarks.ndim != 3:

        raise ValueError(
            "landmarks must have shape "
            "(T, N, 3)."
        )

    if landmarks.shape[2] != 3:

        raise ValueError(
            "landmarks must contain "
            "x, y and z."
        )

    if (
        timestamps.shape[0]
        != landmarks.shape[0]
    ):

        raise ValueError(
            "timestamps and landmarks "
            "frame counts do not match."
        )

    return (
        timestamps,
        landmarks,
    )


# ============================================================
# JSON SAFE CONVERSION
# ============================================================

def landmarks_to_json_safe(
    landmarks: np.ndarray,
) -> list:
    """
    Converte o array de landmarks para uma estrutura
    compatível com JSON.

    Valores NaN / inf são convertidos para None.

    Isto é importante porque JSON standard não possui
    representação válida para NaN.
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    result: list = []

    for frame in landmarks:

        frame_result: list = []

        for landmark in frame:

            point: list = []

            for value in landmark:

                if np.isfinite(
                    value
                ):

                    point.append(
                        float(value)
                    )

                else:

                    point.append(
                        None
                    )

            frame_result.append(
                point
            )

        result.append(
            frame_result
        )

    return result


# ============================================================
# TIMESTAMPS JSON SAFE
# ============================================================

def timestamps_to_json_safe(
    timestamps: np.ndarray,
) -> list[float | None]:
    """
    Converte timestamps NumPy para valores JSON-safe.
    """

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    result: list[
        float | None
    ] = []

    for value in timestamps:

        if np.isfinite(
            value
        ):

            result.append(
                float(value)
            )

        else:

            result.append(
                None
            )

    return result