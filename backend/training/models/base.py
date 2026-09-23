from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    """
    Common interface for gesture classifiers.

    All classifiers receive a fixed-size feature matrix:

        X.shape = (n_samples, n_features)

    and a binary label vector:

        y.shape = (n_samples,)

    Current labels:

        0 -> negative
        1 -> positive
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique classifier identifier.
        """

        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Train the classifier.
        """

        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict binary labels.
        """

        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class probabilities.
        """

        raise NotImplementedError