# from __future__ import annotations

# from dataclasses import dataclass

# import numpy as np

# from backend.training.config import (
#     DEFAULT_MODEL,
# )
# from backend.training.dataset.loader import (
#     TrainingSample,
#     load_gesture_samples,
# )
# from backend.training.models.base import (
#     BaseClassifier,
# )
# from backend.training.models.decision_tree import (
#     DecisionTreeModel,
# )
# from backend.training.representation.base import (
#     BaseRepresentation,
# )
# from backend.training.representation.temporal_pyramid import (
#     TemporalPyramidRepresentation,
# )


# # ============================================================
# # EXCEPTIONS
# # ============================================================

# class TrainingPipelineError(
#     RuntimeError
# ):
#     """
#     Raised when the training pipeline cannot build or train a
#     gesture classifier safely.
#     """

#     pass


# # ============================================================
# # TRAINING DATASET
# # ============================================================

# @dataclass
# class TrainingDataset:
#     """
#     Fixed-size dataset ready to be consumed by a classifier.

#     X
#         Feature matrix:

#             (n_samples, n_features)

#     y
#         Binary labels:

#             0 -> negative
#             1 -> positive

#     feature_names
#         Human-readable name for every column in X.

#     sample_ids
#         Original UUID associated with every row in X.

#     sample_types
#         Original positive / negative type for every row.
#     """

#     X: np.ndarray
#     y: np.ndarray

#     feature_names: list[str]

#     sample_ids: list[str]
#     sample_types: list[str]

#     gesture_id: str
#     gesture_name: str
#     target: str

#     representation_name: str

#     @property
#     def sample_count(
#         self,
#     ) -> int:
#         return int(
#             self.X.shape[0]
#         )

#     @property
#     def feature_count(
#         self,
#     ) -> int:
#         return int(
#             self.X.shape[1]
#         )

#     @property
#     def positive_count(
#         self,
#     ) -> int:
#         return int(
#             np.sum(
#                 self.y == 1
#             )
#         )

#     @property
#     def negative_count(
#         self,
#     ) -> int:
#         return int(
#             np.sum(
#                 self.y == 0
#             )
#         )


# # ============================================================
# # TRAINING RESULT
# # ============================================================

# @dataclass
# class TrainingResult:
#     """
#     Result of one classifier training execution.

#     Note:

#     training_predictions are predictions performed on the same
#     data used for fitting.

#     They are useful for pipeline validation only and MUST NOT be
#     interpreted as real model evaluation.
#     """

#     dataset: TrainingDataset

#     model: BaseClassifier

#     training_predictions: np.ndarray
#     training_probabilities: np.ndarray

#     @property
#     def model_name(
#         self,
#     ) -> str:
#         return self.model.name

#     @property
#     def training_accuracy(
#         self,
#     ) -> float:
#         """
#         Accuracy on the training samples.

#         This is NOT a generalization metric.
#         """

#         return float(
#             np.mean(
#                 self.training_predictions
#                 == self.dataset.y
#             )
#         )


# # ============================================================
# # MODEL FACTORY
# # ============================================================

# def create_model(
#     model_name: str,
# ) -> BaseClassifier:
#     """
#     Create a classifier using its registered name.

#     Additional classifiers can be added here later without
#     changing the rest of the training pipeline.
#     """

#     if model_name == "decision_tree":
#         return DecisionTreeModel()

#     raise TrainingPipelineError(
#         f"Unsupported classifier: "
#         f"'{model_name}'."
#     )


# # ============================================================
# # REPRESENTATION FACTORY
# # ============================================================

# def create_representation(
#     representation_name: str = "temporal_pyramid",
# ) -> BaseRepresentation:
#     """
#     Create the temporal representation used to transform
#     variable-length samples into fixed-size feature vectors.
#     """

#     if (
#         representation_name
#         == "temporal_pyramid"
#     ):
#         return (
#             TemporalPyramidRepresentation()
#         )

#     raise TrainingPipelineError(
#         f"Unsupported representation: "
#         f"'{representation_name}'."
#     )


# # ============================================================
# # DATASET BUILDING
# # ============================================================

# def build_training_dataset(
#     samples: list[TrainingSample],
#     representation: BaseRepresentation,
# ) -> TrainingDataset:
#     """
#     Transform variable-length gesture samples into one fixed-size
#     classifier dataset.

#     Every sample must produce:

#         - the same number of features
#         - the exact same feature names
#         - finite numerical values
#     """

#     if not samples:
#         raise TrainingPipelineError(
#             "Cannot build a training dataset "
#             "without samples."
#         )

#     first_sample = samples[0]

#     gesture_id = (
#         first_sample.gesture_id
#     )

#     gesture_name = (
#         first_sample.gesture_name
#     )

#     target = (
#         first_sample.target
#     )

#     vectors: list[
#         np.ndarray
#     ] = []

#     labels: list[
#         int
#     ] = []

#     sample_ids: list[
#         str
#     ] = []

#     sample_types: list[
#         str
#     ] = []

#     reference_feature_names: (
#         list[str] | None
#     ) = None

#     for sample in samples:

#         # ----------------------------------------------------
#         # Dataset consistency
#         # ----------------------------------------------------

#         if (
#             sample.gesture_id
#             != gesture_id
#         ):
#             raise TrainingPipelineError(
#                 "Samples from multiple gestures "
#                 "cannot be mixed in the same "
#                 "binary classifier dataset."
#             )

#         if (
#             sample.target
#             != target
#         ):
#             raise TrainingPipelineError(
#                 "Samples with different targets "
#                 "cannot be mixed in the same dataset."
#             )

#         # ----------------------------------------------------
#         # Temporal representation
#         # ----------------------------------------------------

#         result = representation.transform(
#             sample
#         )

#         # ----------------------------------------------------
#         # Validate feature names
#         # ----------------------------------------------------

#         if (
#             reference_feature_names
#             is None
#         ):
#             reference_feature_names = (
#                 list(
#                     result.feature_names
#                 )
#             )

#         elif (
#             result.feature_names
#             != reference_feature_names
#         ):
#             raise TrainingPipelineError(
#                 "Samples generated inconsistent "
#                 "feature names."
#             )

#         # ----------------------------------------------------
#         # Validate numerical vector
#         # ----------------------------------------------------

#         if not np.all(
#             np.isfinite(
#                 result.vector
#             )
#         ):
#             invalid_count = int(
#                 np.sum(
#                     ~np.isfinite(
#                         result.vector
#                     )
#                 )
#             )

#             raise TrainingPipelineError(
#                 f"Sample '{sample.sample_id}' "
#                 f"contains {invalid_count} "
#                 "non-finite values after temporal "
#                 "representation."
#             )

#         vectors.append(
#             result.vector
#         )

#         labels.append(
#             sample.label
#         )

#         sample_ids.append(
#             sample.sample_id
#         )

#         sample_types.append(
#             sample.sample_type
#         )

#     # --------------------------------------------------------
#     # Stack samples
#     # --------------------------------------------------------

#     try:
#         X = np.stack(
#             vectors,
#             axis=0,
#         ).astype(
#             np.float32,
#             copy=False,
#         )

#     except ValueError as exc:
#         raise TrainingPipelineError(
#             "Could not create the training matrix. "
#             "Samples appear to have different "
#             "feature dimensions."
#         ) from exc

#     y = np.asarray(
#         labels,
#         dtype=np.int64,
#     )

#     if (
#         reference_feature_names
#         is None
#     ):
#         raise TrainingPipelineError(
#             "Feature names were not generated."
#         )

#     # --------------------------------------------------------
#     # Final validation
#     # --------------------------------------------------------

#     if X.ndim != 2:
#         raise TrainingPipelineError(
#             "Training matrix must be "
#             "two-dimensional."
#         )

#     if y.ndim != 1:
#         raise TrainingPipelineError(
#             "Label vector must be "
#             "one-dimensional."
#         )

#     if (
#         X.shape[0]
#         != y.shape[0]
#     ):
#         raise TrainingPipelineError(
#             "Training matrix and label vector "
#             "contain different sample counts."
#         )

#     if (
#         X.shape[1]
#         != len(
#             reference_feature_names
#         )
#     ):
#         raise TrainingPipelineError(
#             "Feature matrix width does not match "
#             "the number of feature names."
#         )

#     return TrainingDataset(
#         X=X,
#         y=y,

#         feature_names=(
#             reference_feature_names
#         ),

#         sample_ids=sample_ids,
#         sample_types=sample_types,

#         gesture_id=gesture_id,
#         gesture_name=gesture_name,
#         target=target,

#         representation_name=(
#             representation.name
#         ),
#     )


# # ============================================================
# # TRAINING PIPELINE
# # ============================================================

# def train_gesture(
#     gesture_id: str,
#     *,
#     model_name: str = DEFAULT_MODEL,
#     representation_name: str = "temporal_pyramid",
# ) -> TrainingResult:
#     """
#     Train one binary gesture classifier.

#     Current architecture:

#         gesture samples
#             ↓
#         dataset loader
#             ↓
#         temporal representation
#             ↓
#         fixed-size X / y
#             ↓
#         classifier
#             ↓
#         trained model

#     Example:

#         result = train_gesture(
#             "swipe-left"
#         )
#     """

#     # --------------------------------------------------------
#     # Load variable-length samples
#     # --------------------------------------------------------

#     samples = load_gesture_samples(
#         gesture_id
#     )

#     # --------------------------------------------------------
#     # Representation
#     # --------------------------------------------------------

#     representation = (
#         create_representation(
#             representation_name
#         )
#     )

#     dataset = (
#         build_training_dataset(
#             samples,
#             representation,
#         )
#     )

#     # --------------------------------------------------------
#     # Basic binary-class validation
#     # --------------------------------------------------------

#     unique_labels = np.unique(
#         dataset.y
#     )

#     if (
#         unique_labels.size
#         < 2
#     ):
#         raise TrainingPipelineError(
#             f"Gesture '{gesture_id}' requires "
#             "at least one positive and one "
#             "negative sample."
#         )

#     # --------------------------------------------------------
#     # Classifier
#     # --------------------------------------------------------

#     model = create_model(
#         model_name
#     )

#     model.fit(
#         dataset.X,
#         dataset.y,
#     )

#     # --------------------------------------------------------
#     # Technical training-set predictions
#     # --------------------------------------------------------

#     predictions = model.predict(
#         dataset.X
#     )

#     probabilities = (
#         model.predict_proba(
#             dataset.X
#         )
#     )

#     return TrainingResult(
#         dataset=dataset,
#         model=model,
#         training_predictions=predictions,
#         training_probabilities=probabilities,
#     )


from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import (
    train_test_split,
)

from backend.training.config import (
    DEFAULT_MODEL,
    DEFAULT_REPRESENTATION,
    RANDOM_STATE,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)

from backend.training.dataset.loader import (
    TrainingSample,
    load_gesture_samples,
)

from backend.training.models.base import (
    BaseClassifier,
)

from backend.training.models.decision_tree import (
    DecisionTreeModel,
)

from backend.training.representation.base import (
    BaseRepresentation,
)

from backend.training.representation.temporal_pyramid import (
    TemporalPyramidRepresentation,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class TrainingPipelineError(
    RuntimeError
):
    """
    Raised when a training dataset cannot be created or
    evaluated safely.
    """

    pass


# ============================================================
# DATASET
# ============================================================

@dataclass
class TrainingDataset:

    X: np.ndarray

    y: np.ndarray

    feature_names: list[str]

    sample_ids: list[str]

    sample_types: list[str]

    gesture_id: str

    gesture_name: str

    target: str

    representation_name: str


    @property
    def sample_count(
        self,
    ) -> int:

        return int(
            self.X.shape[0]
        )


    @property
    def feature_count(
        self,
    ) -> int:

        return int(
            self.X.shape[1]
        )


    @property
    def positive_count(
        self,
    ) -> int:

        return int(
            np.sum(
                self.y == 1
            )
        )


    @property
    def negative_count(
        self,
    ) -> int:

        return int(
            np.sum(
                self.y == 0
            )
        )


# ============================================================
# TRAINING RESULT
# ============================================================

@dataclass
class TrainingResult:

    dataset: TrainingDataset

    model: BaseClassifier

    train_indices: np.ndarray

    validation_indices: np.ndarray

    validation_predictions: np.ndarray

    validation_probabilities: np.ndarray

    accuracy: float

    precision: float

    recall: float

    f1: float

    true_negative: int

    false_positive: int

    false_negative: int

    true_positive: int


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    @property
    def model_name(
        self,
    ) -> str:

        return (
            self.model.name
        )


    # --------------------------------------------------------
    # TRAINING COUNTS
    # --------------------------------------------------------

    @property
    def training_sample_count(
        self,
    ) -> int:

        return int(
            self.train_indices.shape[0]
        )


    @property
    def validation_sample_count(
        self,
    ) -> int:

        return int(
            self.validation_indices.shape[0]
        )


    @property
    def training_positive_count(
        self,
    ) -> int:

        return int(
            np.sum(
                self.dataset.y[
                    self.train_indices
                ]
                == 1
            )
        )


    @property
    def training_negative_count(
        self,
    ) -> int:

        return int(
            np.sum(
                self.dataset.y[
                    self.train_indices
                ]
                == 0
            )
        )


    @property
    def validation_positive_count(
        self,
    ) -> int:

        return int(
            np.sum(
                self.dataset.y[
                    self.validation_indices
                ]
                == 1
            )
        )


    @property
    def validation_negative_count(
        self,
    ) -> int:

        return int(
            np.sum(
                self.dataset.y[
                    self.validation_indices
                ]
                == 0
            )
        )


    # --------------------------------------------------------
    # SAMPLE IDS
    # --------------------------------------------------------

    @property
    def training_sample_ids(
        self,
    ) -> list[str]:

        return [
            self.dataset.sample_ids[
                int(index)
            ]

            for index
            in self.train_indices
        ]


    @property
    def validation_sample_ids(
        self,
    ) -> list[str]:

        return [
            self.dataset.sample_ids[
                int(index)
            ]

            for index
            in self.validation_indices
        ]


    # --------------------------------------------------------
    # SPLIT INFORMATION
    # --------------------------------------------------------

    @property
    def train_ratio(
        self,
    ) -> float:

        return float(
            self.training_sample_count
            / self.dataset.sample_count
        )


    @property
    def validation_ratio(
        self,
    ) -> float:

        return float(
            self.validation_sample_count
            / self.dataset.sample_count
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    @property
    def confusion_matrix(
        self,
    ) -> dict[str, int]:

        return {
            "tn": (
                self.true_negative
            ),

            "fp": (
                self.false_positive
            ),

            "fn": (
                self.false_negative
            ),

            "tp": (
                self.true_positive
            ),
        }


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    model_name: str,
) -> BaseClassifier:

    if (
        model_name
        == "decision_tree"
    ):

        return (
            DecisionTreeModel(
                random_state=(
                    RANDOM_STATE
                )
            )
        )


    raise TrainingPipelineError(
        f"Unsupported model: "
        f"'{model_name}'."
    )


# ============================================================
# REPRESENTATION FACTORY
# ============================================================

def create_representation(
    representation_name: str,
) -> BaseRepresentation:

    if (
        representation_name
        == "temporal_pyramid"
    ):

        return (
            TemporalPyramidRepresentation()
        )


    raise TrainingPipelineError(
        f"Unsupported representation: "
        f"'{representation_name}'."
    )


# ============================================================
# BUILD DATASET
# ============================================================

def build_training_dataset(
    samples: list[
        TrainingSample
    ],
    representation: BaseRepresentation,
) -> TrainingDataset:

    if not samples:

        raise TrainingPipelineError(
            "No samples were loaded."
        )


    # ========================================================
    # REFERENCE SAMPLE
    # ========================================================

    first_sample = samples[0]

    gesture_id = (
        first_sample.gesture_id
    )

    gesture_name = (
        first_sample.gesture_name
    )

    target = (
        first_sample.target
    )


    # ========================================================
    # OUTPUT COLLECTIONS
    # ========================================================

    vectors: list[
        np.ndarray
    ] = []

    labels: list[
        int
    ] = []

    sample_ids: list[
        str
    ] = []

    sample_types: list[
        str
    ] = []

    reference_feature_names: (
        list[str]
        | None
    ) = None


    # ========================================================
    # TRANSFORM SAMPLES
    # ========================================================

    for sample in samples:

        # ----------------------------------------------------
        # Gesture consistency
        # ----------------------------------------------------

        if (
            sample.gesture_id
            != gesture_id
        ):

            raise TrainingPipelineError(
                "Training samples contain "
                "multiple gesture IDs."
            )


        if (
            sample.target
            != target
        ):

            raise TrainingPipelineError(
                "Training samples contain "
                "multiple gesture targets."
            )


        # ----------------------------------------------------
        # Fixed representation
        # ----------------------------------------------------

        result = (
            representation.transform(
                sample
            )
        )


        # ----------------------------------------------------
        # Non-finite values
        # ----------------------------------------------------

        finite_mask = np.isfinite(
            result.vector
        )


        if not np.all(
            finite_mask
        ):

            invalid_count = int(
                np.sum(
                    ~finite_mask
                )
            )


            raise TrainingPipelineError(
                f"Sample '{sample.sample_id}' "
                f"contains {invalid_count} "
                "non-finite values after "
                "temporal representation."
            )


        # ----------------------------------------------------
        # Feature ordering consistency
        # ----------------------------------------------------

        if (
            reference_feature_names
            is None
        ):

            reference_feature_names = (
                list(
                    result.feature_names
                )
            )

        elif (
            result.feature_names
            != reference_feature_names
        ):

            raise TrainingPipelineError(
                "Feature names or feature ordering "
                "differ between samples."
            )


        vectors.append(
            result.vector
        )

        labels.append(
            int(
                sample.label
            )
        )

        sample_ids.append(
            sample.sample_id
        )

        sample_types.append(
            sample.sample_type
        )


    if (
        reference_feature_names
        is None
    ):

        raise TrainingPipelineError(
            "Could not determine "
            "feature names."
        )


    # ========================================================
    # MATRIX
    # ========================================================

    X = np.stack(
        vectors,
        axis=0,
    ).astype(
        np.float32
    )


    y = np.asarray(
        labels,
        dtype=np.int64,
    )


    if X.ndim != 2:

        raise TrainingPipelineError(
            "Training matrix X must "
            "be two-dimensional."
        )


    if y.ndim != 1:

        raise TrainingPipelineError(
            "Training labels y must "
            "be one-dimensional."
        )


    if (
        X.shape[0]
        != y.shape[0]
    ):

        raise TrainingPipelineError(
            "Sample count does not match "
            "label count."
        )


    return TrainingDataset(
        X=X,

        y=y,

        feature_names=(
            reference_feature_names
        ),

        sample_ids=(
            sample_ids
        ),

        sample_types=(
            sample_types
        ),

        gesture_id=(
            gesture_id
        ),

        gesture_name=(
            gesture_name
        ),

        target=(
            target
        ),

        representation_name=(
            representation.name
        ),
    )


# ============================================================
# VALIDATE TRAIN / VALIDATION SPLIT
# ============================================================

def validate_split(
    dataset: TrainingDataset,
) -> None:
    """
    Validate that a stratified 80/20 split can be created.

    Stratified splitting requires enough examples of both
    classes to place examples in training and validation.
    """

    if (
        dataset.positive_count
        == 0
        or dataset.negative_count
        == 0
    ):

        raise TrainingPipelineError(
            "Training requires both positive "
            "and negative samples."
        )


    if (
        dataset.positive_count
        < 2
        or dataset.negative_count
        < 2
    ):

        raise TrainingPipelineError(
            "A stratified train/validation split "
            "requires at least 2 positive and "
            "2 negative samples."
        )


    sample_count = (
        dataset.sample_count
    )


    validation_count = int(
        math.ceil(
            sample_count
            * VALIDATION_RATIO
        )
    )


    training_count = (
        sample_count
        - validation_count
    )


    class_count = 2


    if (
        validation_count
        < class_count
    ):

        raise TrainingPipelineError(
            "Not enough samples to create "
            "a stratified 80/20 split. "
            f"The validation set would contain "
            f"only {validation_count} sample(s)."
        )


    if (
        training_count
        < class_count
    ):

        raise TrainingPipelineError(
            "Not enough samples to create "
            "a stratified 80/20 split. "
            f"The training set would contain "
            f"only {training_count} sample(s)."
        )


# ============================================================
# CREATE TRAIN / VALIDATION SPLIT
# ============================================================

def create_train_validation_split(
    dataset: TrainingDataset,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Return dataset indices for a reproducible stratified
    training / validation split.
    """

    validate_split(
        dataset
    )


    indices = np.arange(
        dataset.sample_count,
        dtype=np.int64,
    )


    try:

        (
            train_indices,
            validation_indices,
        ) = train_test_split(
            indices,

            test_size=(
                VALIDATION_RATIO
            ),

            random_state=(
                RANDOM_STATE
            ),

            stratify=(
                dataset.y
            ),
        )

    except ValueError as exc:

        raise TrainingPipelineError(
            "Could not create the stratified "
            "training/validation split: "
            f"{exc}"
        ) from exc


    return (
        np.asarray(
            train_indices,
            dtype=np.int64,
        ),

        np.asarray(
            validation_indices,
            dtype=np.int64,
        ),
    )


# ============================================================
# EVALUATION
# ============================================================

def calculate_validation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[
    str,
    float | int
]:

    if (
        y_true.shape
        != y_pred.shape
    ):

        raise TrainingPipelineError(
            "Validation labels and predictions "
            "have different shapes."
        )


    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )


    precision = float(
        precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )
    )


    recall = float(
        recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )
    )


    f1 = float(
        f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )
    )


    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )


    (
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    ) = (
        matrix.ravel()
    )


    return {
        "accuracy": (
            accuracy
        ),

        "precision": (
            precision
        ),

        "recall": (
            recall
        ),

        "f1": (
            f1
        ),

        "tn": int(
            true_negative
        ),

        "fp": int(
            false_positive
        ),

        "fn": int(
            false_negative
        ),

        "tp": int(
            true_positive
        ),
    }


# ============================================================
# TRAIN GESTURE
# ============================================================

def train_gesture(
    gesture_id: str,
    *,
    model_name: str = DEFAULT_MODEL,
    representation_name: str = DEFAULT_REPRESENTATION,
) -> TrainingResult:
    """
    Train and evaluate one binary gesture classifier.

    Current evaluation strategy:

        complete dataset
              ↓
        stratified 80/20 split
              ↓
        80% training
              ↓
        fit model
              ↓
        20% validation
              ↓
        accuracy / precision / recall / F1
        confusion matrix

    The validation examples are never passed to model.fit().
    """

    # ========================================================
    # LOAD
    # ========================================================

    samples = (
        load_gesture_samples(
            gesture_id
        )
    )


    # ========================================================
    # REPRESENTATION
    # ========================================================

    representation = (
        create_representation(
            representation_name
        )
    )


    # ========================================================
    # DATASET
    # ========================================================

    dataset = (
        build_training_dataset(
            samples,
            representation,
        )
    )


    # ========================================================
    # BOTH CLASSES REQUIRED
    # ========================================================

    unique_classes = np.unique(
        dataset.y
    )


    if not np.array_equal(
        unique_classes,
        np.asarray(
            [
                0,
                1,
            ],
            dtype=np.int64,
        ),
    ):

        raise TrainingPipelineError(
            "Training requires both "
            "negative (0) and positive (1) "
            "samples."
        )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    (
        train_indices,
        validation_indices,
    ) = create_train_validation_split(
        dataset
    )


    X_train = (
        dataset.X[
            train_indices
        ]
    )


    y_train = (
        dataset.y[
            train_indices
        ]
    )


    X_validation = (
        dataset.X[
            validation_indices
        ]
    )


    y_validation = (
        dataset.y[
            validation_indices
        ]
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = create_model(
        model_name
    )


    model.fit(
        X_train,
        y_train,
    )


    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    validation_predictions = (
        model.predict(
            X_validation
        )
    )


    validation_probabilities = (
        model.predict_proba(
            X_validation
        )
    )


    # ========================================================
    # METRICS
    # ========================================================

    metrics = (
        calculate_validation_metrics(
            y_validation,
            validation_predictions,
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    return TrainingResult(
        dataset=dataset,

        model=model,

        train_indices=(
            train_indices
        ),

        validation_indices=(
            validation_indices
        ),

        validation_predictions=(
            np.asarray(
                validation_predictions,
                dtype=np.int64,
            )
        ),

        validation_probabilities=(
            np.asarray(
                validation_probabilities,
                dtype=np.float64,
            )
        ),

        accuracy=float(
            metrics[
                "accuracy"
            ]
        ),

        precision=float(
            metrics[
                "precision"
            ]
        ),

        recall=float(
            metrics[
                "recall"
            ]
        ),

        f1=float(
            metrics[
                "f1"
            ]
        ),

        true_negative=int(
            metrics[
                "tn"
            ]
        ),

        false_positive=int(
            metrics[
                "fp"
            ]
        ),

        false_negative=int(
            metrics[
                "fn"
            ]
        ),

        true_positive=int(
            metrics[
                "tp"
            ]
        ),
    )