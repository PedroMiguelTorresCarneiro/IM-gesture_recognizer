from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from backend.feature_extraction.config import (
    HAND_LANDMARK_COUNT,
)


class HandExtractor:
    """
    Extrator de landmarks da mão utilizando MediaPipe Hands.

    Para cada frame devolve:

        shape = (21, 3)

    correspondente a:

        x, y, z

    Se nenhuma mão for detetada, devolve um array
    preenchido com NaN.

    Nesta primeira versão é considerada apenas uma mão.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:

        self._mp_hands = mp.solutions.hands

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=(
                min_detection_confidence
            ),
            min_tracking_confidence=(
                min_tracking_confidence
            ),
        )

    # ========================================================
    # EXTRACT
    # ========================================================

    def extract(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Extrai os 21 landmarks da mão para um frame.

        Parameters
        ----------
        frame:
            Frame OpenCV em formato BGR.

        Returns
        -------
        np.ndarray
            Array com shape (21, 3).

            Cada linha corresponde a:

                [x, y, z]

            Se não existir deteção:

                todas as posições = NaN
        """

        if frame is None:
            return self._empty_landmarks()

        if frame.size == 0:
            return self._empty_landmarks()

        # ----------------------------------------------------
        # OpenCV BGR -> MediaPipe RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # MediaPipe não precisa de modificar o frame.
        rgb_frame.flags.writeable = False

        results = self._hands.process(
            rgb_frame
        )

        # ----------------------------------------------------
        # No detection
        # ----------------------------------------------------

        if not results.multi_hand_landmarks:
            return self._empty_landmarks()

        # Nesta V1 utilizamos apenas uma mão.
        #
        # max_num_hands=1 garante que o MediaPipe devolve,
        # no máximo, uma deteção.
        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )

        landmarks = np.full(
            (
                HAND_LANDMARK_COUNT,
                3,
            ),
            np.nan,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Copy landmarks
        # ----------------------------------------------------

        for index, landmark in enumerate(
            hand_landmarks.landmark
        ):
            landmarks[index] = [
                landmark.x,
                landmark.y,
                landmark.z,
            ]

        return landmarks

    # ========================================================
    # EMPTY LANDMARKS
    # ========================================================

    @staticmethod
    def _empty_landmarks() -> np.ndarray:
        """
        Cria a representação de um frame onde nenhuma
        mão foi detetada.
        """

        return np.full(
            (
                HAND_LANDMARK_COUNT,
                3,
            ),
            np.nan,
            dtype=np.float32,
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:
        """
        Liberta os recursos utilizados pelo MediaPipe.
        """

        self._hands.close()

    # ========================================================
    # CONTEXT MANAGER
    # ========================================================

    def __enter__(self) -> "HandExtractor":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()