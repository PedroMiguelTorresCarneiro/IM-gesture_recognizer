from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from backend.training.dataset.loader import TrainingSample


# ============================================================
# REPRESENTATION RESULT
# ============================================================

@dataclass
class RepresentationResult:
    """
    Fixed-size representation generated from one temporal sample.

    vector
        Numerical feature vector that can be consumed by a
        traditional machine-learning classifier.

    feature_names
        Human-readable name for every position in the vector.

    The following invariant must always hold:

        len(vector) == len(feature_names)
    """

    vector: np.ndarray
    feature_names: list[str]

    def __post_init__(self) -> None:
        """
        Validate representation consistency.
        """

        if self.vector.ndim != 1:
            raise ValueError(
                "Representation vector must be one-dimensional. "
                f"Received shape: {self.vector.shape}"
            )

        if (
            self.vector.shape[0]
            != len(self.feature_names)
        ):
            raise ValueError(
                "Representation vector size does not match "
                "the number of feature names. "
                f"Vector size: {self.vector.shape[0]}, "
                f"feature names: {len(self.feature_names)}."
            )

    @property
    def feature_count(self) -> int:
        """
        Number of classifier input features.
        """

        return int(
            self.vector.shape[0]
        )


# ============================================================
# BASE REPRESENTATION
# ============================================================

class BaseRepresentation(ABC):
    """
    Base interface for temporal gesture representations.

    A representation receives one variable-length TrainingSample
    and converts it into a fixed-size one-dimensional vector.

    Implementations must preserve human-readable feature names so
    that classifier decisions can later be interpreted.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique representation identifier.

        Example:

            temporal_pyramid
        """

        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        sample: TrainingSample,
    ) -> RepresentationResult:
        """
        Transform one variable-length gesture sample into a
        fixed-size representation.

        Parameters
        ----------
        sample:
            Training sample containing the original temporal
            feature arrays.

        Returns
        -------
        RepresentationResult
            Fixed-size numerical vector and corresponding
            human-readable feature names.
        """

        raise NotImplementedError