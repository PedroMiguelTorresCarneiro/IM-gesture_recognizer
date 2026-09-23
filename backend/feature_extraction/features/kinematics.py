from __future__ import annotations

import numpy as np


# ============================================================
# VELOCITY
# ============================================================

def calculate_velocities(
    landmarks: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Calcula a velocidade de cada landmark ao longo
    da sequência temporal.

    Expected landmarks shape:
        (T, N, 3)

    Expected timestamps shape:
        (T,)

    Returns
    -------
    velocities:
        Array com shape (T, N, 3)

    speeds:
        Array com shape (T, N)

    Notes
    -----
    - O primeiro frame não tem velocidade calculável,
      pelo que fica preenchido com NaN.

    - Se o frame atual ou anterior tiver landmarks
      inválidos/NaN, a velocidade correspondente fica NaN.

    - A velocidade é calculada apenas com informação
      presente e passada:

          v[t] = (p[t] - p[t-1]) / dt

      Isto mantém compatibilidade com uma futura
      utilização em tempo real.
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    _validate_inputs(
        landmarks,
        timestamps,
    )

    frame_count = landmarks.shape[0]
    landmark_count = landmarks.shape[1]

    velocities = np.full(
        (
            frame_count,
            landmark_count,
            3,
        ),
        np.nan,
        dtype=np.float32,
    )

    speeds = np.full(
        (
            frame_count,
            landmark_count,
        ),
        np.nan,
        dtype=np.float32,
    )

    for frame_index in range(
        1,
        frame_count,
    ):
        current_timestamp = timestamps[
            frame_index
        ]

        previous_timestamp = timestamps[
            frame_index - 1
        ]

        dt = (
            current_timestamp
            - previous_timestamp
        )

        if not np.isfinite(dt):
            continue

        if dt <= 0:
            continue

        current_landmarks = landmarks[
            frame_index
        ]

        previous_landmarks = landmarks[
            frame_index - 1
        ]

        valid_landmarks = (
            np.isfinite(
                current_landmarks
            ).all(axis=1)
            &
            np.isfinite(
                previous_landmarks
            ).all(axis=1)
        )

        if not valid_landmarks.any():
            continue

        displacement = (
            current_landmarks[
                valid_landmarks
            ]
            -
            previous_landmarks[
                valid_landmarks
            ]
        )

        frame_velocities = (
            displacement / dt
        )

        velocities[
            frame_index,
            valid_landmarks,
        ] = frame_velocities

        speeds[
            frame_index,
            valid_landmarks,
        ] = np.linalg.norm(
            frame_velocities,
            axis=1,
        )

    return velocities, speeds


# ============================================================
# ACCELERATION
# ============================================================

def calculate_accelerations(
    velocities: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Calcula a aceleração de cada landmark ao longo
    da sequência temporal.

    Expected velocities shape:
        (T, N, 3)

    Expected timestamps shape:
        (T,)

    Returns
    -------
    accelerations:
        Array com shape (T, N, 3)

    acceleration_magnitudes:
        Array com shape (T, N)

    Notes
    -----
    A aceleração é calculada por:

        a[t] = (v[t] - v[t-1]) / dt

    Como a velocidade do primeiro frame não existe,
    normalmente os primeiros dois frames não terão
    aceleração válida.
    """

    velocities = np.asarray(
        velocities,
        dtype=np.float32,
    )

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    _validate_inputs(
        velocities,
        timestamps,
    )

    frame_count = velocities.shape[0]
    landmark_count = velocities.shape[1]

    accelerations = np.full(
        (
            frame_count,
            landmark_count,
            3,
        ),
        np.nan,
        dtype=np.float32,
    )

    acceleration_magnitudes = np.full(
        (
            frame_count,
            landmark_count,
        ),
        np.nan,
        dtype=np.float32,
    )

    for frame_index in range(
        1,
        frame_count,
    ):
        current_timestamp = timestamps[
            frame_index
        ]

        previous_timestamp = timestamps[
            frame_index - 1
        ]

        dt = (
            current_timestamp
            - previous_timestamp
        )

        if not np.isfinite(dt):
            continue

        if dt <= 0:
            continue

        current_velocities = velocities[
            frame_index
        ]

        previous_velocities = velocities[
            frame_index - 1
        ]

        valid_landmarks = (
            np.isfinite(
                current_velocities
            ).all(axis=1)
            &
            np.isfinite(
                previous_velocities
            ).all(axis=1)
        )

        if not valid_landmarks.any():
            continue

        velocity_difference = (
            current_velocities[
                valid_landmarks
            ]
            -
            previous_velocities[
                valid_landmarks
            ]
        )

        frame_accelerations = (
            velocity_difference / dt
        )

        accelerations[
            frame_index,
            valid_landmarks,
        ] = frame_accelerations

        acceleration_magnitudes[
            frame_index,
            valid_landmarks,
        ] = np.linalg.norm(
            frame_accelerations,
            axis=1,
        )

    return (
        accelerations,
        acceleration_magnitudes,
    )


# ============================================================
# COMPLETE KINEMATICS PIPELINE
# ============================================================

def calculate_kinematics(
    landmarks: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Calcula todas as features cinemáticas atualmente
    suportadas.

    Returns
    -------
    dict:
        {
            "velocities": ...,
            "speeds": ...,
            "accelerations": ...,
            "acceleration_magnitudes": ...,
        }
    """

    velocities, speeds = (
        calculate_velocities(
            landmarks,
            timestamps,
        )
    )

    (
        accelerations,
        acceleration_magnitudes,
    ) = calculate_accelerations(
        velocities,
        timestamps,
    )

    return {
        "velocities": velocities,
        "speeds": speeds,
        "accelerations": accelerations,
        "acceleration_magnitudes": (
            acceleration_magnitudes
        ),
    }


# ============================================================
# VALIDATION
# ============================================================

def _validate_inputs(
    sequence: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    """
    Validação comum para sequências temporais
    com shape (T, N, 3).
    """

    if sequence.ndim != 3:
        raise ValueError(
            "Sequence must have shape "
            "(T, N, 3)."
        )

    if sequence.shape[2] != 3:
        raise ValueError(
            "Sequence must contain x, y and z "
            "components."
        )

    if timestamps.ndim != 1:
        raise ValueError(
            "timestamps must have shape (T,)."
        )

    if (
        sequence.shape[0]
        != timestamps.shape[0]
    ):
        raise ValueError(
            "Sequence frame count and timestamps "
            "length must match."
        )