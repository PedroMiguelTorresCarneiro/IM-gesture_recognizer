from __future__ import annotations

from itertools import combinations

import numpy as np


# ============================================================
# DISTANCE PAIRS
# ============================================================

def generate_landmark_pairs(
    landmark_count: int,
) -> list[tuple[int, int]]:
    """
    Gera todos os pares únicos de landmarks.

    Exemplo:
        landmark_count = 3

        [
            (0, 1),
            (0, 2),
            (1, 2),
        ]

    Parameters
    ----------
    landmark_count:
        Número total de landmarks.

    Returns
    -------
    list[tuple[int, int]]
        Lista de pares únicos.
    """

    if landmark_count < 2:
        raise ValueError(
            "landmark_count must be at least 2."
        )

    return list(
        combinations(
            range(landmark_count),
            2,
        )
    )


# ============================================================
# DISTANCES FOR ONE FRAME
# ============================================================

def calculate_frame_distances(
    landmarks: np.ndarray,
    pairs: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Calcula as distâncias Euclidianas 3D entre todos
    os pares de landmarks de um frame.

    Expected input shape:
        (N, 3)

    Output shape:
        (N * (N - 1) / 2,)

    Caso um dos landmarks de um par contenha NaN,
    a distância correspondente será também NaN.

    Parameters
    ----------
    landmarks:
        Array NumPy com shape (N, 3).

    pairs:
        Lista opcional de pares de landmarks.
        Se não for fornecida, são gerados todos os
        pares possíveis.

    Returns
    -------
    np.ndarray
        Vetor de distâncias.
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    if landmarks.ndim != 2:
        raise ValueError(
            "landmarks must have shape (N, 3)."
        )

    if landmarks.shape[1] != 3:
        raise ValueError(
            "landmarks must contain x, y and z "
            "coordinates."
        )

    landmark_count = landmarks.shape[0]

    if pairs is None:
        pairs = generate_landmark_pairs(
            landmark_count
        )

    distances = np.full(
        len(pairs),
        np.nan,
        dtype=np.float32,
    )

    for index, (landmark_a, landmark_b) in enumerate(
        pairs
    ):
        point_a = landmarks[landmark_a]
        point_b = landmarks[landmark_b]

        if (
            np.isnan(point_a).any()
            or np.isnan(point_b).any()
        ):
            continue

        difference = point_b - point_a

        distances[index] = np.linalg.norm(
            difference
        )

    return distances


# ============================================================
# DISTANCES FOR COMPLETE SEQUENCE
# ============================================================

def calculate_sequence_distances(
    landmarks: np.ndarray,
    pairs: list[tuple[int, int]] | None = None,
) -> tuple[
    np.ndarray,
    list[tuple[int, int]],
]:
    """
    Calcula as distâncias para todos os frames de
    uma sequência.

    Expected input shape:
        (T, N, 3)

    Output shape:
        (T, D)

    onde:
        T = número de frames
        N = número de landmarks
        D = número de pares únicos

    Parameters
    ----------
    landmarks:
        Array com shape (T, N, 3).

    pairs:
        Lista opcional de pares.

    Returns
    -------
    tuple:
        distances:
            Array com shape (T, D).

        pairs:
            Lista de pares correspondente às colunas
            do array distances.
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    if landmarks.ndim != 3:
        raise ValueError(
            "landmarks must have shape (T, N, 3)."
        )

    if landmarks.shape[2] != 3:
        raise ValueError(
            "landmarks must contain x, y and z "
            "coordinates."
        )

    frame_count = landmarks.shape[0]

    landmark_count = landmarks.shape[1]

    if pairs is None:
        pairs = generate_landmark_pairs(
            landmark_count
        )

    distances = np.full(
        (
            frame_count,
            len(pairs),
        ),
        np.nan,
        dtype=np.float32,
    )

    for frame_index in range(frame_count):
        distances[frame_index] = (
            calculate_frame_distances(
                landmarks[frame_index],
                pairs,
            )
        )

    return distances, pairs