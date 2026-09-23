from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from backend.feature_extraction.config import (
    POSE_LANDMARK_COUNT,
)


class PoseExtractor:
    """
    Extrator de landmarks corporais utilizando MediaPipe Pose.

    Para cada frame devolve:

        shape = (33, 3)

    correspondente a:

        x, y, z

    Se nenhuma pose for detetada, devolve um array
    preenchido com NaN.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:

        self._mp_pose = mp.solutions.pose

        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=False,
            enable_segmentation=False,
            smooth_segmentation=False,
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
        Extrai os 33 landmarks corporais de um frame.

        Parameters
        ----------
        frame:
            Frame OpenCV em formato BGR.

        Returns
        -------
        np.ndarray
            Array com shape (33, 3).

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

        rgb_frame.flags.writeable = False

        results = self._pose.process(
            rgb_frame
        )

        # ----------------------------------------------------
        # No detection
        # ----------------------------------------------------

        if results.pose_landmarks is None:
            return self._empty_landmarks()

        pose_landmarks = (
            results.pose_landmarks.landmark
        )

        landmarks = np.full(
            (
                POSE_LANDMARK_COUNT,
                3,
            ),
            np.nan,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Copy landmarks
        # ----------------------------------------------------

        for index, landmark in enumerate(
            pose_landmarks
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
        Representação de um frame onde nenhuma pose
        foi detetada.
        """

        return np.full(
            (
                POSE_LANDMARK_COUNT,
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

        self._pose.close()

    # ========================================================
    # CONTEXT MANAGER
    # ========================================================

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()