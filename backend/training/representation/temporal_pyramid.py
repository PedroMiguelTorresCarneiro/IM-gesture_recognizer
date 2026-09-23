from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from backend.training.config import (
    FEATURE_FAMILIES,
    TEMPORAL_AGGREGATION,
    TEMPORAL_PYRAMID_LEVELS,
)
from backend.training.dataset.loader import TrainingSample
from backend.training.representation.base import (
    BaseRepresentation,
    RepresentationResult,
)


# ============================================================
# CONSTANTS
# ============================================================

AXIS_NAMES = (
    "x",
    "y",
    "z",
)


# ============================================================
# REGION MODEL
# ============================================================

@dataclass(frozen=True)
class TemporalRegion:
    """
    One region of the temporal pyramid.

    start and end are expressed in normalized gesture time:

        0.0 -> beginning of the gesture
        1.0 -> end of the gesture
    """

    index: int
    level: int
    position: int

    start: float
    end: float

    @property
    def name(self) -> str:
        return f"R{self.index}"


# ============================================================
# TEMPORAL PYRAMID
# ============================================================

class TemporalPyramidRepresentation(
    BaseRepresentation
):
    """
    Convert a variable-length gesture sequence into a fixed-size
    feature vector using a temporal pyramid.

    Default configuration:

        levels = (1, 2, 4)

    Producing:

        R0 -> 0%   - 100%

        R1 -> 0%   - 50%
        R2 -> 50%  - 100%

        R3 -> 0%   - 25%
        R4 -> 25%  - 50%
        R5 -> 50%  - 75%
        R6 -> 75%  - 100%

    For each region, every scalar temporal feature is aggregated
    independently.

    V1 aggregation:

        mean

    The result is a fixed-size vector regardless of the original
    number of frames.
    """

    def __init__(
        self,
        levels: tuple[int, ...] | None = None,
        aggregation: str | None = None,
    ) -> None:

        self.levels = (
            levels
            if levels is not None
            else TEMPORAL_PYRAMID_LEVELS
        )

        self.aggregation = (
            aggregation
            if aggregation is not None
            else TEMPORAL_AGGREGATION
        )

        self._validate_configuration()

        self.regions = self._build_regions()

    # ========================================================
    # BASE REPRESENTATION INTERFACE
    # ========================================================

    @property
    def name(self) -> str:
        return "temporal_pyramid"

    def transform(
        self,
        sample: TrainingSample,
    ) -> RepresentationResult:
        """
        Transform one variable-length TrainingSample into a
        fixed-size temporal-pyramid representation.
        """

        timestamps = np.asarray(
            sample.timestamps,
            dtype=np.float64,
        )

        normalized_time = (
            self._normalize_timestamps(
                timestamps
            )
        )

        scalar_features, scalar_names = (
            self._build_scalar_features(
                sample
            )
        )

        region_vectors: list[
            np.ndarray
        ] = []

        feature_names: list[
            str
        ] = []

        for region in self.regions:

            mask = self._region_mask(
                normalized_time,
                region,
            )

            region_data = (
                scalar_features[
                    mask
                ]
            )

            aggregated = (
                self._aggregate_region(
                    region_data,
                    scalar_features.shape[1],
                )
            )

            region_vectors.append(
                aggregated
            )

            feature_names.extend(
                self._build_region_feature_names(
                    region,
                    scalar_names,
                )
            )

        vector = np.concatenate(
            region_vectors,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        return RepresentationResult(
            vector=vector,
            feature_names=feature_names,
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def _validate_configuration(
        self,
    ) -> None:
        """
        Validate pyramid configuration.
        """

        if not self.levels:
            raise ValueError(
                "Temporal pyramid must contain "
                "at least one level."
            )

        for level in self.levels:

            if not isinstance(
                level,
                int,
            ):
                raise ValueError(
                    "Temporal pyramid levels "
                    "must be integers."
                )

            if level <= 0:
                raise ValueError(
                    "Temporal pyramid levels "
                    "must be greater than zero."
                )

        if self.aggregation != "mean":
            raise ValueError(
                "Unsupported temporal aggregation: "
                f"'{self.aggregation}'. "
                "V1 currently supports only 'mean'."
            )

    # ========================================================
    # REGION CONSTRUCTION
    # ========================================================

    def _build_regions(
        self,
    ) -> list[TemporalRegion]:
        """
        Build regions in pyramid order.

        For levels:

            (1, 2, 4)

        produces:

            R0
            R1 R2
            R3 R4 R5 R6
        """

        regions: list[
            TemporalRegion
        ] = []

        region_index = 0

        for level in self.levels:

            width = 1.0 / level

            for position in range(
                level
            ):

                start = (
                    position
                    * width
                )

                end = (
                    (position + 1)
                    * width
                )

                regions.append(
                    TemporalRegion(
                        index=region_index,
                        level=level,
                        position=position,
                        start=start,
                        end=end,
                    )
                )

                region_index += 1

        return regions

    # ========================================================
    # TIMESTAMP NORMALIZATION
    # ========================================================

    def _normalize_timestamps(
        self,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize real video timestamps into [0, 1].

        Temporal regions are therefore based on relative gesture
        time rather than frame index.
        """

        if timestamps.ndim != 1:
            raise ValueError(
                "timestamps must be "
                "one-dimensional."
            )

        if timestamps.size == 0:
            raise ValueError(
                "Cannot build temporal pyramid "
                "from an empty timestamp sequence."
            )

        if not np.all(
            np.isfinite(
                timestamps
            )
        ):
            raise ValueError(
                "timestamps contain NaN or "
                "infinite values."
            )

        if timestamps.size == 1:
            return np.zeros(
                shape=(1,),
                dtype=np.float64,
            )

        deltas = np.diff(
            timestamps
        )

        if np.any(
            deltas < 0
        ):
            raise ValueError(
                "timestamps must be ordered "
                "chronologically."
            )

        start = float(
            timestamps[0]
        )

        end = float(
            timestamps[-1]
        )

        duration = (
            end
            - start
        )

        if duration <= 0:
            raise ValueError(
                "Gesture timestamps do not "
                "contain a positive temporal span."
            )

        normalized = (
            timestamps
            - start
        ) / duration

        return np.clip(
            normalized,
            0.0,
            1.0,
        )

    # ========================================================
    # REGION MASK
    # ========================================================

    def _region_mask(
        self,
        normalized_time: np.ndarray,
        region: TemporalRegion,
    ) -> np.ndarray:
        """
        Select frames belonging to one temporal region.

        Regions use:

            [start, end)

        except regions ending at 1.0, which include the final
        timestamp.
        """

        if np.isclose(
            region.end,
            1.0,
        ):
            return (
                (normalized_time >= region.start)
                &
                (normalized_time <= region.end)
            )

        return (
            (normalized_time >= region.start)
            &
            (normalized_time < region.end)
        )

    # ========================================================
    # SCALAR FEATURE MATRIX
    # ========================================================

    def _build_scalar_features(
        self,
        sample: TrainingSample,
    ) -> tuple[
        np.ndarray,
        list[str],
    ]:
        """
        Convert all selected temporal feature families into one
        matrix:

            (T, F)

        where:

            T = number of frames
            F = scalar features per frame

        HAND currently produces:

            F = 441

        when all feature families are enabled.
        """

        matrices: list[
            np.ndarray
        ] = []

        names: list[
            str
        ] = []

        for family in FEATURE_FAMILIES:

            if family not in sample.features:
                raise ValueError(
                    f"Sample '{sample.sample_id}' "
                    f"is missing feature family "
                    f"'{family}'."
                )

            array = np.asarray(
                sample.features[
                    family
                ]
            )

            matrix, family_names = (
                self._flatten_feature_family(
                    sample,
                    family,
                    array,
                )
            )

            matrices.append(
                matrix
            )

            names.extend(
                family_names
            )

        if not matrices:
            raise ValueError(
                "No temporal feature families "
                "were configured."
            )

        frame_count = (
            sample.frame_count
        )

        for matrix in matrices:

            if (
                matrix.shape[0]
                != frame_count
            ):
                raise ValueError(
                    "Temporal feature matrix has "
                    "an inconsistent frame count."
                )

        combined = np.concatenate(
            matrices,
            axis=1,
        )

        if (
            combined.shape[1]
            != len(names)
        ):
            raise RuntimeError(
                "Scalar feature matrix size "
                "does not match generated "
                "feature names."
            )

        return (
            combined,
            names,
        )

    # ========================================================
    # FEATURE FAMILY FLATTENING
    # ========================================================

    def _flatten_feature_family(
        self,
        sample: TrainingSample,
        family: str,
        array: np.ndarray,
    ) -> tuple[
        np.ndarray,
        list[str],
    ]:
        """
        Flatten one feature family into scalar channels while
        preserving meaningful names.
        """

        if family == "landmarks":
            return (
                self._flatten_xyz_family(
                    array,
                    prefix="landmark",
                )
            )

        if family == "distances":
            return (
                self._flatten_distances(
                    sample,
                    array,
                )
            )

        if family == "velocities":
            return (
                self._flatten_xyz_family(
                    array,
                    prefix="velocity",
                )
            )

        if family == "speeds":
            return (
                self._flatten_landmark_scalar_family(
                    array,
                    prefix="speed",
                )
            )

        if family == "accelerations":
            return (
                self._flatten_xyz_family(
                    array,
                    prefix="acceleration",
                )
            )

        if (
            family
            == "acceleration_magnitudes"
        ):
            return (
                self._flatten_landmark_scalar_family(
                    array,
                    prefix="acceleration_magnitude",
                )
            )

        raise ValueError(
            f"Unsupported feature family: "
            f"'{family}'."
        )

    # ========================================================
    # XYZ FEATURES
    # ========================================================

    def _flatten_xyz_family(
        self,
        array: np.ndarray,
        *,
        prefix: str,
    ) -> tuple[
        np.ndarray,
        list[str],
    ]:
        """
        Flatten:

            (T, N, 3)

        into:

            (T, N * 3)
        """

        if array.ndim != 3:
            raise ValueError(
                f"Feature family '{prefix}' "
                f"must have shape (T, N, 3). "
                f"Received: {array.shape}"
            )

        if (
            array.shape[2]
            != 3
        ):
            raise ValueError(
                f"Feature family '{prefix}' "
                "must contain x, y and z."
            )

        frame_count = (
            array.shape[0]
        )

        landmark_count = (
            array.shape[1]
        )

        matrix = array.reshape(
            frame_count,
            landmark_count * 3,
        )

        names: list[
            str
        ] = []

        for landmark_index in range(
            landmark_count
        ):

            for axis in AXIS_NAMES:

                names.append(
                    f"{prefix}_"
                    f"{landmark_index}_"
                    f"{axis}"
                )

        return (
            matrix,
            names,
        )

    # ========================================================
    # LANDMARK SCALAR FEATURES
    # ========================================================

    def _flatten_landmark_scalar_family(
        self,
        array: np.ndarray,
        *,
        prefix: str,
    ) -> tuple[
        np.ndarray,
        list[str],
    ]:
        """
        Validate scalar landmark features:

            (T, N)
        """

        if array.ndim != 2:
            raise ValueError(
                f"Feature family '{prefix}' "
                f"must have shape (T, N). "
                f"Received: {array.shape}"
            )

        landmark_count = (
            array.shape[1]
        )

        names = [
            f"{prefix}_{index}"
            for index in range(
                landmark_count
            )
        ]

        return (
            array,
            names,
        )

    # ========================================================
    # DISTANCE FEATURES
    # ========================================================

    def _flatten_distances(
        self,
        sample: TrainingSample,
        array: np.ndarray,
    ) -> tuple[
        np.ndarray,
        list[str],
    ]:
        """
        Preserve the mapping between distance columns and the
        landmark pair that generated them.
        """

        if array.ndim != 2:
            raise ValueError(
                "Distances must have shape "
                f"(T, D). Received: {array.shape}"
            )

        distance_pairs = np.asarray(
            sample.features[
                "distance_pairs"
            ]
        )

        if (
            distance_pairs.ndim != 2
            or distance_pairs.shape[1] != 2
        ):
            raise ValueError(
                "distance_pairs must have "
                "shape (D, 2)."
            )

        if (
            distance_pairs.shape[0]
            != array.shape[1]
        ):
            raise ValueError(
                "Distance feature count does not "
                "match distance_pairs."
            )

        names: list[
            str
        ] = []

        for pair in distance_pairs:

            first = int(
                pair[0]
            )

            second = int(
                pair[1]
            )

            names.append(
                f"distance_"
                f"{first}_"
                f"{second}"
            )

        return (
            array,
            names,
        )

    # ========================================================
    # AGGREGATION
    # ========================================================

    def _aggregate_region(
        self,
        region_data: np.ndarray,
        feature_count: int,
    ) -> np.ndarray:
        """
        Aggregate every scalar feature independently inside one
        temporal region.

        NaN values are intentionally ignored when valid values
        exist.

        If a complete feature channel is unavailable inside the
        region, its aggregated value remains NaN.
        """

        if region_data.ndim != 2:
            raise ValueError(
                "Region data must have shape "
                "(frames, features)."
            )

        if region_data.shape[0] == 0:
            return np.full(
                shape=(feature_count,),
                fill_value=np.nan,
                dtype=np.float32,
            )

        if self.aggregation == "mean":

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    category=RuntimeWarning,
                )

                result = np.nanmean(
                    region_data,
                    axis=0,
                )

            return np.asarray(
                result,
                dtype=np.float32,
            )

        raise ValueError(
            "Unsupported aggregation: "
            f"{self.aggregation}"
        )

    # ========================================================
    # FEATURE NAMES
    # ========================================================

    def _build_region_feature_names(
        self,
        region: TemporalRegion,
        scalar_names: list[str],
    ) -> list[str]:
        """
        Prefix scalar feature names with the temporal region.

        Example:

            R3_landmark_8_x_mean
            R5_distance_0_8_mean
            R6_velocity_12_y_mean
        """

        return [
            (
                f"{region.name}_"
                f"{name}_"
                f"{self.aggregation}"
            )
            for name in scalar_names
        ]