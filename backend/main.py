# from __future__ import annotations

# import json
# import mimetypes
# import re
# import unicodedata
# import uuid

# from datetime import (
#     datetime,
#     timezone,
# )

# from pathlib import Path
# from typing import Literal

# from fastapi import (
#     FastAPI,
#     File,
#     Form,
#     HTTPException,
#     UploadFile,
# )

# from fastapi.responses import (
#     FileResponse,
# )

# from fastapi.staticfiles import (
#     StaticFiles,
# )

# from pydantic import BaseModel

# from backend.feature_extraction.pipeline import (
#     extract_sample,
# )


# # ============================================================
# # PATHS
# # ============================================================

# BASE_DIR = (
#     Path(__file__)
#     .resolve()
#     .parent
#     .parent
# )

# FRONTEND_DIR = (
#     BASE_DIR
#     / "frontend"
# )

# DATA_DIR = (
#     BASE_DIR
#     / "data"
# )

# GESTURES_DIR = (
#     DATA_DIR
#     / "gestures"
# )


# GESTURES_DIR.mkdir(
#     parents=True,
#     exist_ok=True,
# )


# # ============================================================
# # APP
# # ============================================================

# app = FastAPI(
#     title="Gesture Capture",
#     version="0.5.0",
# )


# # ============================================================
# # MODELS
# # ============================================================

# class GestureCreate(
#     BaseModel
# ):
#     name: str

#     target: Literal[
#         "hand",
#         "body",
#     ]


# # ============================================================
# # HELPERS
# # ============================================================

# def now_iso() -> str:

#     return (
#         datetime.now(
#             timezone.utc
#         )
#         .isoformat()
#     )


# def slugify(
#     value: str,
# ) -> str:

#     value = (
#         unicodedata.normalize(
#             "NFKD",
#             value,
#         )
#         .encode(
#             "ascii",
#             "ignore",
#         )
#         .decode(
#             "ascii"
#         )
#     )

#     value = (
#         value.strip().lower()
#     )

#     value = re.sub(
#         r"[^a-z0-9]+",
#         "-",
#         value,
#     )

#     value = value.strip(
#         "-"
#     )

#     if not value:

#         raise ValueError(
#             "Gesture name does not produce "
#             "a valid identifier."
#         )

#     return value


# def read_json(
#     path: Path,
# ) -> dict:

#     if not path.exists():

#         raise FileNotFoundError(
#             str(path)
#         )

#     with path.open(
#         "r",
#         encoding="utf-8",
#     ) as file:

#         return json.load(
#             file
#         )


# def write_json(
#     path: Path,
#     data: dict,
# ) -> None:

#     path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     with path.open(
#         "w",
#         encoding="utf-8",
#     ) as file:

#         json.dump(
#             data,
#             file,
#             indent=2,
#             ensure_ascii=False,
#         )


# # ============================================================
# # GESTURE HELPERS
# # ============================================================

# def get_gesture_dir(
#     gesture_id: str,
# ) -> Path:

#     return (
#         GESTURES_DIR
#         / gesture_id
#     )


# def get_gesture_metadata(
#     gesture_id: str,
# ) -> dict:

#     gesture_dir = (
#         get_gesture_dir(
#             gesture_id
#         )
#     )

#     gesture_file = (
#         gesture_dir
#         / "gesture.json"
#     )

#     if not gesture_file.exists():

#         raise HTTPException(
#             status_code=404,
#             detail=(
#                 f"Gesture '{gesture_id}' "
#                 "not found."
#             ),
#         )

#     return read_json(
#         gesture_file
#     )


# # ============================================================
# # SAMPLE HELPERS
# # ============================================================

# def get_sample_dir(
#     gesture_id: str,
#     sample_type: str,
#     sample_id: str,
# ) -> Path:
#     """
#     Obtém e valida a diretoria de um sample.
#     """

#     if sample_type not in {
#         "positive",
#         "negative",
#     }:

#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "sample_type must be "
#                 "'positive' or 'negative'."
#             ),
#         )

#     gesture_dir = (
#         get_gesture_dir(
#             gesture_id
#         )
#     )

#     sample_dir = (
#         gesture_dir
#         / sample_type
#         / sample_id
#     )

#     if (
#         not sample_dir.exists()
#         or not sample_dir.is_dir()
#     ):

#         raise HTTPException(
#             status_code=404,
#             detail=(
#                 f"Sample '{sample_id}' "
#                 "not found."
#             ),
#         )

#     return sample_dir


# def get_sample_metadata(
#     gesture_id: str,
#     sample_type: str,
#     sample_id: str,
# ) -> dict:
#     """
#     Obtém sample.json.
#     """

#     sample_dir = (
#         get_sample_dir(
#             gesture_id,
#             sample_type,
#             sample_id,
#         )
#     )

#     sample_file = (
#         sample_dir
#         / "sample.json"
#     )

#     if not sample_file.exists():

#         raise HTTPException(
#             status_code=404,
#             detail=(
#                 "sample.json not found."
#             ),
#         )

#     return read_json(
#         sample_file
#     )


# # ============================================================
# # COUNTS
# # ============================================================

# def count_samples(
#     gesture_dir: Path,
#     sample_type: str,
# ) -> int:

#     directory = (
#         gesture_dir
#         / sample_type
#     )

#     if not directory.exists():
#         return 0

#     return sum(
#         1
#         for path in directory.iterdir()
#         if (
#             path.is_dir()
#             and (
#                 path
#                 / "sample.json"
#             ).exists()
#         )
#     )


# def gesture_with_counts(
#     gesture: dict,
# ) -> dict:

#     gesture_dir = (
#         get_gesture_dir(
#             gesture["id"]
#         )
#     )

#     result = dict(
#         gesture
#     )

#     result["samples"] = {

#         "positive":
#             count_samples(
#                 gesture_dir,
#                 "positive",
#             ),

#         "negative":
#             count_samples(
#                 gesture_dir,
#                 "negative",
#             ),
#     }

#     return result


# # ============================================================
# # FEATURE SUMMARY
# # ============================================================

# def get_feature_summary(
#     sample_dir: Path,
# ) -> dict | None:

#     features_file = (
#         sample_dir
#         / "features.npz"
#     )

#     metadata_file = (
#         sample_dir
#         / "features.json"
#     )

#     if (
#         not features_file.exists()
#         or not metadata_file.exists()
#     ):

#         return None

#     try:

#         metadata = read_json(
#             metadata_file
#         )

#     except Exception:

#         return None

#     return {

#         "version":
#             metadata.get(
#                 "version"
#             ),

#         "landmarker":
#             metadata.get(
#                 "landmarker"
#             ),

#         "fps":
#             metadata.get(
#                 "fps"
#             ),

#         "frame_count":
#             metadata.get(
#                 "frame_count"
#             ),

#         "extracted_duration_seconds":
#             metadata.get(
#                 "extracted_duration_seconds"
#             ),

#         "landmark_count":
#             metadata.get(
#                 "landmark_count"
#             ),

#         "distance_count":
#             metadata.get(
#                 "distance_count"
#             ),

#         "valid_landmark_frames":
#             metadata.get(
#                 "valid_landmark_frames"
#             ),

#         "missing_landmark_frames":
#             metadata.get(
#                 "missing_landmark_frames"
#             ),

#         "features":
#             metadata.get(
#                 "features",
#                 {},
#             ),

#         "processing":
#             metadata.get(
#                 "processing",
#                 {},
#             ),
#     }


# # ============================================================
# # SAMPLE LIST
# # ============================================================

# def list_samples_for_type(
#     gesture_dir: Path,
#     sample_type: str,
# ) -> list[dict]:

#     directory = (
#         gesture_dir
#         / sample_type
#     )

#     if not directory.exists():
#         return []

#     samples: list[
#         dict
#     ] = []

#     for sample_dir in sorted(
#         directory.iterdir()
#     ):

#         if not sample_dir.is_dir():
#             continue

#         sample_file = (
#             sample_dir
#             / "sample.json"
#         )

#         if not sample_file.exists():
#             continue

#         try:

#             metadata = read_json(
#                 sample_file
#             )

#         except Exception:

#             continue

#         feature_summary = (
#             get_feature_summary(
#                 sample_dir
#             )
#         )

#         metadata[
#             "features_extracted"
#         ] = (
#             feature_summary
#             is not None
#         )

#         metadata[
#             "feature_summary"
#         ] = (
#             feature_summary
#         )

#         samples.append(
#             metadata
#         )

#     return samples


# # ============================================================
# # VIDEO EXTENSION
# # ============================================================

# def get_video_extension(
#     video: UploadFile,
# ) -> str:

#     filename = (
#         video.filename
#         or ""
#     )

#     suffix = (
#         Path(filename)
#         .suffix
#         .lower()
#     )

#     if suffix in {
#         ".webm",
#         ".mp4",
#     }:

#         return suffix

#     content_type = (
#         video.content_type
#         or ""
#     ).lower()

#     if "mp4" in content_type:

#         return ".mp4"

#     return ".webm"


# # ============================================================
# # CREATE GESTURE
# # ============================================================

# @app.post(
#     "/api/gestures",
#     status_code=201,
# )
# def create_gesture(
#     payload: GestureCreate,
# ):

#     name = (
#         payload.name.strip()
#     )

#     if not name:

#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "Gesture name cannot be empty."
#             ),
#         )

#     try:

#         gesture_id = slugify(
#             name
#         )

#     except ValueError as exc:

#         raise HTTPException(
#             status_code=400,
#             detail=str(exc),
#         )

#     gesture_dir = (
#         get_gesture_dir(
#             gesture_id
#         )
#     )

#     if gesture_dir.exists():

#         raise HTTPException(
#             status_code=409,
#             detail=(
#                 f"Gesture '{gesture_id}' "
#                 "already exists."
#             ),
#         )

#     (
#         gesture_dir
#         / "positive"
#     ).mkdir(
#         parents=True,
#         exist_ok=False,
#     )

#     (
#         gesture_dir
#         / "negative"
#     ).mkdir(
#         parents=True,
#         exist_ok=False,
#     )

#     metadata = {

#         "id":
#             gesture_id,

#         "name":
#             name,

#         "target":
#             payload.target,

#         "created_at":
#             now_iso(),
#     }

#     write_json(
#         gesture_dir
#         / "gesture.json",
#         metadata,
#     )

#     return gesture_with_counts(
#         metadata
#     )


# # ============================================================
# # LIST GESTURES
# # ============================================================

# @app.get(
#     "/api/gestures"
# )
# def list_gestures():

#     gestures: list[
#         dict
#     ] = []

#     for gesture_dir in sorted(
#         GESTURES_DIR.iterdir()
#     ):

#         if not gesture_dir.is_dir():
#             continue

#         metadata_file = (
#             gesture_dir
#             / "gesture.json"
#         )

#         if not metadata_file.exists():
#             continue

#         try:

#             metadata = read_json(
#                 metadata_file
#             )

#         except Exception:

#             continue

#         gestures.append(
#             gesture_with_counts(
#                 metadata
#             )
#         )

#     return gestures


# # ============================================================
# # GET GESTURE
# # ============================================================

# @app.get(
#     "/api/gestures/{gesture_id}"
# )
# def get_gesture(
#     gesture_id: str,
# ):

#     metadata = (
#         get_gesture_metadata(
#             gesture_id
#         )
#     )

#     return gesture_with_counts(
#         metadata
#     )


# # ============================================================
# # CREATE SAMPLE
# # ============================================================

# @app.post(
#     "/api/gestures/{gesture_id}/samples",
#     status_code=201,
# )
# async def create_sample(
#     gesture_id: str,

#     sample_type: Literal[
#         "positive",
#         "negative",
#     ] = Form(...),

#     trim_start_ms: int = Form(...),

#     trim_end_ms: int = Form(...),

#     video: UploadFile = File(...),
# ):

#     gesture = (
#         get_gesture_metadata(
#             gesture_id
#         )
#     )

#     gesture_dir = (
#         get_gesture_dir(
#             gesture_id
#         )
#     )

#     # --------------------------------------------------------
#     # Validate trim
#     # --------------------------------------------------------

#     if trim_start_ms < 0:

#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "trim_start_ms cannot "
#                 "be negative."
#             ),
#         )

#     if trim_end_ms <= trim_start_ms:

#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "trim_end_ms must be greater "
#                 "than trim_start_ms."
#             ),
#         )

#     # --------------------------------------------------------
#     # Sample
#     # --------------------------------------------------------

#     sample_id = str(
#         uuid.uuid4()
#     )

#     sample_dir = (
#         gesture_dir
#         / sample_type
#         / sample_id
#     )

#     sample_dir.mkdir(
#         parents=True,
#         exist_ok=False,
#     )

#     # --------------------------------------------------------
#     # Video
#     # --------------------------------------------------------

#     extension = (
#         get_video_extension(
#             video
#         )
#     )

#     video_filename = (
#         f"original{extension}"
#     )

#     video_path = (
#         sample_dir
#         / video_filename
#     )

#     try:

#         video_content = (
#             await video.read()
#         )

#         if not video_content:

#             raise HTTPException(
#                 status_code=400,
#                 detail=(
#                     "Uploaded video is empty."
#                 ),
#             )

#         with video_path.open(
#             "wb"
#         ) as file:

#             file.write(
#                 video_content
#             )

#     finally:

#         await video.close()

#     # --------------------------------------------------------
#     # sample.json
#     # --------------------------------------------------------

#     metadata = {

#         "id":
#             sample_id,

#         "gesture_id":
#             gesture["id"],

#         "gesture_name":
#             gesture["name"],

#         "target":
#             gesture["target"],

#         "sample_type":
#             sample_type,

#         "video":
#             video_filename,

#         "trim": {

#             "start_ms":
#                 int(
#                     trim_start_ms
#                 ),

#             "end_ms":
#                 int(
#                     trim_end_ms
#                 ),

#             "duration_ms":
#                 int(
#                     trim_end_ms
#                     - trim_start_ms
#                 ),
#         },

#         "created_at":
#             now_iso(),
#     }

#     write_json(
#         sample_dir
#         / "sample.json",
#         metadata,
#     )

#     # --------------------------------------------------------
#     # Feature Extraction
#     # --------------------------------------------------------

#     try:

#         extraction_result = (
#             extract_sample(
#                 gesture_dir=gesture_dir,
#                 sample_dir=sample_dir,
#                 label=sample_type,
#                 force=True,
#             )
#         )

#         metadata[
#             "feature_extraction"
#         ] = {

#             "status":
#                 extraction_result[
#                     "status"
#                 ],

#             "frame_count":
#                 extraction_result.get(
#                     "frame_count"
#                 ),

#             "fps":
#                 extraction_result.get(
#                     "fps"
#                 ),
#         }

#     except Exception as exc:

#         metadata[
#             "feature_extraction"
#         ] = {

#             "status":
#                 "error",

#             "error":
#                 str(exc),
#         }

#         print(
#             "[Feature Extraction Error] "
#             f"{gesture_id} / "
#             f"{sample_type} / "
#             f"{sample_id}: "
#             f"{exc}"
#         )

#     return metadata


# # ============================================================
# # LIST SAMPLES
# # ============================================================

# @app.get(
#     "/api/gestures/{gesture_id}/samples"
# )
# def list_gesture_samples(
#     gesture_id: str,
# ):

#     get_gesture_metadata(
#         gesture_id
#     )

#     gesture_dir = (
#         get_gesture_dir(
#             gesture_id
#         )
#     )

#     return {

#         "positive":
#             list_samples_for_type(
#                 gesture_dir,
#                 "positive",
#             ),

#         "negative":
#             list_samples_for_type(
#                 gesture_dir,
#                 "negative",
#             ),
#     }


# # ============================================================
# # GET SAMPLE
# # ============================================================

# @app.get(
#     "/api/gestures/"
#     "{gesture_id}/samples/"
#     "{sample_type}/{sample_id}"
# )
# def get_sample(
#     gesture_id: str,
#     sample_type: Literal[
#         "positive",
#         "negative",
#     ],
#     sample_id: str,
# ):
#     """
#     Devolve os metadados completos necessários para
#     inspecionar um sample no frontend.
#     """

#     get_gesture_metadata(
#         gesture_id
#     )

#     sample_dir = (
#         get_sample_dir(
#             gesture_id,
#             sample_type,
#             sample_id,
#         )
#     )

#     metadata = (
#         get_sample_metadata(
#             gesture_id,
#             sample_type,
#             sample_id,
#         )
#     )

#     feature_summary = (
#         get_feature_summary(
#             sample_dir
#         )
#     )

#     result = dict(
#         metadata
#     )

#     result[
#         "features_extracted"
#     ] = (
#         feature_summary
#         is not None
#     )

#     result[
#         "feature_summary"
#     ] = (
#         feature_summary
#     )

#     result[
#         "video_url"
#     ] = (
#         f"/api/gestures/"
#         f"{gesture_id}/samples/"
#         f"{sample_type}/"
#         f"{sample_id}/video"
#     )

#     return result


# # ============================================================
# # SAMPLE VIDEO
# # ============================================================

# @app.get(
#     "/api/gestures/"
#     "{gesture_id}/samples/"
#     "{sample_type}/{sample_id}/video"
# )
# def get_sample_video(
#     gesture_id: str,
#     sample_type: Literal[
#         "positive",
#         "negative",
#     ],
#     sample_id: str,
# ):
#     """
#     Serve o vídeo original associado ao sample.
#     """

#     get_gesture_metadata(
#         gesture_id
#     )

#     sample_dir = (
#         get_sample_dir(
#             gesture_id,
#             sample_type,
#             sample_id,
#         )
#     )

#     metadata = (
#         get_sample_metadata(
#             gesture_id,
#             sample_type,
#             sample_id,
#         )
#     )

#     video_filename = (
#         metadata.get(
#             "video"
#         )
#     )

#     if not video_filename:

#         raise HTTPException(
#             status_code=404,
#             detail=(
#                 "Video filename not found "
#                 "in sample metadata."
#             ),
#         )

#     video_path = (
#         sample_dir
#         / str(
#             video_filename
#         )
#     )

#     if not video_path.exists():

#         raise HTTPException(
#             status_code=404,
#             detail=(
#                 "Sample video not found."
#             ),
#         )

#     media_type, _ = (
#         mimetypes.guess_type(
#             str(
#                 video_path
#             )
#         )
#     )

#     if media_type is None:

#         media_type = (
#             "application/octet-stream"
#         )

#     return FileResponse(
#         path=str(
#             video_path
#         ),
#         media_type=media_type,
#         filename=None,
#     )


# # ============================================================
# # FRONTEND
# # ============================================================

# app.mount(
#     "/",
#     StaticFiles(
#         directory=str(
#             FRONTEND_DIR
#         ),
#         html=True,
#     ),
#     name="frontend",
# )



from __future__ import annotations

import json
import mimetypes
import re
import unicodedata
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import Literal

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.responses import (
    FileResponse,
)

from fastapi.staticfiles import (
    StaticFiles,
)

from pydantic import BaseModel

from backend.feature_extraction.pipeline import (
    extract_sample,
)

from backend.feature_extraction.storage.reader import (
    landmarks_to_json_safe,
    load_landmark_sequence,
    timestamps_to_json_safe,
)

#from sklearn.tree import export_text

from backend.training.dataset.loader import (
    DatasetLoadError,
    load_gesture_samples,
)

from backend.training.pipeline import (
    TrainingPipelineError,
    train_gesture,
)

import shutil

from backend.training.config import (
    GESTURES_DIR,
)


from backend.training.storage.model_store import (
    ModelStoreError,
    build_dataset_signature,
    load_training_info,
    save_training_result,
    training_exists,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

FRONTEND_DIR = (
    BASE_DIR
    / "frontend"
)

DATA_DIR = (
    BASE_DIR
    / "data"
)

GESTURES_DIR = (
    DATA_DIR
    / "gestures"
)


GESTURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Gesture Capture",
    version="0.6.0",
)


# ============================================================
# MODELS
# ============================================================

class GestureCreate(
    BaseModel
):
    name: str

    target: Literal[
        "hand",
        "body",
    ]


class NegativeLibrarySampleImport(
    BaseModel
):
    gesture_id: str
    sample_id: str


class NegativeLibraryImportRequest(
    BaseModel
):
    samples: list[
        NegativeLibrarySampleImport
    ]

# ============================================================
# HELPERS
# ============================================================

def now_iso() -> str:

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def slugify(
    value: str,
) -> str:

    value = (
        unicodedata.normalize(
            "NFKD",
            value,
        )
        .encode(
            "ascii",
            "ignore",
        )
        .decode(
            "ascii"
        )
    )

    value = (
        value.strip().lower()
    )

    value = re.sub(
        r"[^a-z0-9]+",
        "-",
        value,
    )

    value = value.strip(
        "-"
    )

    if not value:

        raise ValueError(
            "Gesture name does not produce "
            "a valid identifier."
        )

    return value


def read_json(
    path: Path,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            str(path)
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def write_json(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# GESTURE HELPERS
# ============================================================

def get_gesture_dir(
    gesture_id: str,
) -> Path:

    return (
        GESTURES_DIR
        / gesture_id
    )


def get_gesture_metadata(
    gesture_id: str,
) -> dict:

    gesture_dir = (
        get_gesture_dir(
            gesture_id
        )
    )

    gesture_file = (
        gesture_dir
        / "gesture.json"
    )

    if not gesture_file.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                f"Gesture '{gesture_id}' "
                "not found."
            ),
        )

    return read_json(
        gesture_file
    )


# ============================================================
# SAMPLE HELPERS
# ============================================================

def get_sample_dir(
    gesture_id: str,
    sample_type: str,
    sample_id: str,
) -> Path:

    if sample_type not in {
        "positive",
        "negative",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "sample_type must be "
                "'positive' or 'negative'."
            ),
        )

    gesture_dir = (
        get_gesture_dir(
            gesture_id
        )
    )

    sample_dir = (
        gesture_dir
        / sample_type
        / sample_id
    )

    if (
        not sample_dir.exists()
        or not sample_dir.is_dir()
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                f"Sample '{sample_id}' "
                "not found."
            ),
        )

    return sample_dir


def get_sample_metadata(
    gesture_id: str,
    sample_type: str,
    sample_id: str,
) -> dict:

    sample_dir = (
        get_sample_dir(
            gesture_id,
            sample_type,
            sample_id,
        )
    )

    sample_file = (
        sample_dir
        / "sample.json"
    )

    if not sample_file.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "sample.json not found."
            ),
        )

    return read_json(
        sample_file
    )


# ============================================================
# NEGATIVE LIBRARY HELPERS
# ============================================================

def is_complete_sample(
    sample_dir: Path,
) -> bool:
    """
    Verifica se um sample contém tudo o que é necessário
    para ser reutilizado como exemplo negativo.

    Um sample completo deve possuir:

        sample.json
        features.json
        features.npz
        original.webm / original.mp4
    """

    sample_file = (
        sample_dir
        / "sample.json"
    )

    features_metadata_file = (
        sample_dir
        / "features.json"
    )

    features_file = (
        sample_dir
        / "features.npz"
    )


    if not sample_file.is_file():
        return False


    if not features_metadata_file.is_file():
        return False


    if not features_file.is_file():
        return False


    try:

        sample_metadata = read_json(
            sample_file
        )

    except Exception:

        return False


    video_filename = (
        sample_metadata.get(
            "video"
        )
    )


    if not video_filename:
        return False


    video_file = (
        sample_dir
        / str(
            video_filename
        )
    )


    if not video_file.is_file():
        return False


    return True


def negative_library_source_exists(
    destination_gesture_id: str,
    source_gesture_id: str,
    source_sample_id: str,
) -> bool:
    """
    Verifica se um sample da biblioteca já foi importado
    anteriormente para os negativos do gesto de destino.
    """

    negative_dir = (
        get_gesture_dir(
            destination_gesture_id
        )
        / "negative"
    )


    if not negative_dir.exists():
        return False


    for sample_dir in (
        negative_dir.iterdir()
    ):

        if not sample_dir.is_dir():
            continue


        sample_file = (
            sample_dir
            / "sample.json"
        )


        if not sample_file.is_file():
            continue


        try:

            metadata = read_json(
                sample_file
            )

        except Exception:

            continue


        imported_from = (
            metadata.get(
                "imported_from"
            )
        )


        if not isinstance(
            imported_from,
            dict,
        ):
            continue


        if (
            imported_from.get(
                "gesture_id"
            )
            == source_gesture_id

            and

            imported_from.get(
                "sample_id"
            )
            == source_sample_id
        ):

            return True


    return False


def import_positive_as_negative(
    *,
    destination_gesture: dict,
    source_gesture: dict,
    source_sample_id: str,
) -> dict:
    """
    Faz hard-copy de um positive sample de outro gesto
    para a pasta negative do gesto atual.

    É criado um novo UUID.

    Os ficheiros binários são reutilizados sem novo
    processamento:

        original.*
        features.npz

    Os metadados são adaptados:

        sample.json
        features.json
    """

    destination_gesture_id = (
        destination_gesture[
            "id"
        ]
    )

    source_gesture_id = (
        source_gesture[
            "id"
        ]
    )


    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if (
        source_gesture.get(
            "target"
        )
        != destination_gesture.get(
            "target"
        )
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Source and destination gestures "
                "must use the same target."
            ),
        )


    # --------------------------------------------------------
    # Source sample
    # --------------------------------------------------------

    source_sample_dir = (
        get_sample_dir(
            source_gesture_id,
            "positive",
            source_sample_id,
        )
    )


    if not is_complete_sample(
        source_sample_dir
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Source sample "
                f"'{source_sample_id}' "
                "is incomplete."
            ),
        )


    source_sample_metadata = (
        get_sample_metadata(
            source_gesture_id,
            "positive",
            source_sample_id,
        )
    )


    # --------------------------------------------------------
    # Validate source metadata
    # --------------------------------------------------------

    if (
        source_sample_metadata.get(
            "gesture_id"
        )
        != source_gesture_id
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Source sample gesture_id "
                "is inconsistent."
            ),
        )


    if (
        source_sample_metadata.get(
            "sample_type"
        )
        != "positive"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only positive samples can "
                "be imported from the library."
            ),
        )


    if (
        source_sample_metadata.get(
            "target"
        )
        != destination_gesture.get(
            "target"
        )
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Source sample target does not "
                "match destination gesture."
            ),
        )


    # --------------------------------------------------------
    # Prevent duplicate imports
    # --------------------------------------------------------

    if negative_library_source_exists(
        destination_gesture_id,
        source_gesture_id,
        source_sample_id,
    ):

        return {
            "status":
                "already_imported",

            "source_gesture_id":
                source_gesture_id,

            "source_sample_id":
                source_sample_id,

            "sample_id":
                None,
        }


    # --------------------------------------------------------
    # New sample
    # --------------------------------------------------------

    new_sample_id = str(
        uuid.uuid4()
    )


    destination_sample_dir = (
        get_gesture_dir(
            destination_gesture_id
        )
        / "negative"
        / new_sample_id
    )


    imported_at = (
        now_iso()
    )


    try:

        # ----------------------------------------------------
        # Hard copy complete sample directory
        # ----------------------------------------------------

        shutil.copytree(
            source_sample_dir,
            destination_sample_dir,
        )


        # ----------------------------------------------------
        # Rewrite sample.json
        # ----------------------------------------------------

        copied_sample_metadata = (
            read_json(
                destination_sample_dir
                / "sample.json"
            )
        )


        original_created_at = (
            copied_sample_metadata.get(
                "created_at"
            )
        )


        copied_sample_metadata[
            "id"
        ] = (
            new_sample_id
        )


        copied_sample_metadata[
            "gesture_id"
        ] = (
            destination_gesture_id
        )


        copied_sample_metadata[
            "gesture_name"
        ] = (
            destination_gesture[
                "name"
            ]
        )


        copied_sample_metadata[
            "target"
        ] = (
            destination_gesture[
                "target"
            ]
        )


        copied_sample_metadata[
            "sample_type"
        ] = (
            "negative"
        )


        copied_sample_metadata[
            "created_at"
        ] = (
            imported_at
        )


        copied_sample_metadata[
            "imported_from"
        ] = {

            "gesture_id":
                source_gesture_id,

            "gesture_name":
                source_gesture.get(
                    "name",
                    source_gesture_id,
                ),

            "sample_id":
                source_sample_id,

            "sample_type":
                "positive",

            "original_created_at":
                original_created_at,

            "imported_at":
                imported_at,
        }


        write_json(
            destination_sample_dir
            / "sample.json",
            copied_sample_metadata,
        )


        # ----------------------------------------------------
        # Rewrite features.json
        # ----------------------------------------------------

        features_metadata_file = (
            destination_sample_dir
            / "features.json"
        )


        features_metadata = (
            read_json(
                features_metadata_file
            )
        )


        features_metadata[
            "gesture"
        ] = (
            destination_gesture[
                "name"
            ]
        )


        features_metadata[
            "gesture_type"
        ] = (
            destination_gesture[
                "target"
            ]
        )


        features_metadata[
            "label"
        ] = (
            "negative"
        )


        features_metadata[
            "sample_id"
        ] = (
            new_sample_id
        )


        features_metadata[
            "imported_from"
        ] = {

            "gesture_id":
                source_gesture_id,

            "gesture_name":
                source_gesture.get(
                    "name",
                    source_gesture_id,
                ),

            "sample_id":
                source_sample_id,

            "sample_type":
                "positive",

            "imported_at":
                imported_at,
        }


        write_json(
            features_metadata_file,
            features_metadata,
        )


    except Exception:

        # ----------------------------------------------------
        # Avoid leaving half-created samples
        # ----------------------------------------------------

        if destination_sample_dir.exists():

            shutil.rmtree(
                destination_sample_dir,
                ignore_errors=True,
            )


        raise


    return {

        "status":
            "imported",

        "sample_id":
            new_sample_id,

        "source_gesture_id":
            source_gesture_id,

        "source_sample_id":
            source_sample_id,

        "sample":
            copied_sample_metadata,
    }

# ============================================================
# COUNTS
# ============================================================

def count_samples(
    gesture_dir: Path,
    sample_type: str,
) -> int:

    directory = (
        gesture_dir
        / sample_type
    )

    if not directory.exists():
        return 0

    return sum(
        1
        for path in directory.iterdir()
        if (
            path.is_dir()
            and (
                path
                / "sample.json"
            ).exists()
        )
    )


def gesture_with_counts(
    gesture: dict,
) -> dict:

    gesture_dir = (
        get_gesture_dir(
            gesture["id"]
        )
    )

    result = dict(
        gesture
    )

    result["samples"] = {

        "positive":
            count_samples(
                gesture_dir,
                "positive",
            ),

        "negative":
            count_samples(
                gesture_dir,
                "negative",
            ),
    }

    return result


# ============================================================
# FEATURE SUMMARY
# ============================================================

def get_feature_summary(
    sample_dir: Path,
) -> dict | None:

    features_file = (
        sample_dir
        / "features.npz"
    )

    metadata_file = (
        sample_dir
        / "features.json"
    )

    if (
        not features_file.exists()
        or not metadata_file.exists()
    ):

        return None

    try:

        metadata = read_json(
            metadata_file
        )

    except Exception:

        return None

    return {

        "version":
            metadata.get(
                "version"
            ),

        "landmarker":
            metadata.get(
                "landmarker"
            ),

        "fps":
            metadata.get(
                "fps"
            ),

        "frame_count":
            metadata.get(
                "frame_count"
            ),

        "extracted_duration_seconds":
            metadata.get(
                "extracted_duration_seconds"
            ),

        "landmark_count":
            metadata.get(
                "landmark_count"
            ),

        "distance_count":
            metadata.get(
                "distance_count"
            ),

        "valid_landmark_frames":
            metadata.get(
                "valid_landmark_frames"
            ),

        "missing_landmark_frames":
            metadata.get(
                "missing_landmark_frames"
            ),

        "features":
            metadata.get(
                "features",
                {},
            ),

        "processing":
            metadata.get(
                "processing",
                {},
            ),
    }


# ============================================================
# SAMPLE LIST
# ============================================================

def list_samples_for_type(
    gesture_dir: Path,
    sample_type: str,
) -> list[dict]:

    directory = (
        gesture_dir
        / sample_type
    )

    if not directory.exists():
        return []

    samples: list[
        dict
    ] = []

    for sample_dir in sorted(
        directory.iterdir()
    ):

        if not sample_dir.is_dir():
            continue

        sample_file = (
            sample_dir
            / "sample.json"
        )

        if not sample_file.exists():
            continue

        try:

            metadata = read_json(
                sample_file
            )

        except Exception:

            continue

        feature_summary = (
            get_feature_summary(
                sample_dir
            )
        )

        metadata[
            "features_extracted"
        ] = (
            feature_summary
            is not None
        )

        metadata[
            "feature_summary"
        ] = (
            feature_summary
        )

        samples.append(
            metadata
        )

    return samples


# ============================================================
# VIDEO EXTENSION
# ============================================================

def get_video_extension(
    video: UploadFile,
) -> str:

    filename = (
        video.filename
        or ""
    )

    suffix = (
        Path(filename)
        .suffix
        .lower()
    )

    if suffix in {
        ".webm",
        ".mp4",
    }:

        return suffix

    content_type = (
        video.content_type
        or ""
    ).lower()

    if "mp4" in content_type:

        return ".mp4"

    return ".webm"


# ============================================================
# CREATE GESTURE
# ============================================================

@app.post(
    "/api/gestures",
    status_code=201,
)
def create_gesture(
    payload: GestureCreate,
):

    name = (
        payload.name.strip()
    )

    if not name:

        raise HTTPException(
            status_code=400,
            detail=(
                "Gesture name cannot be empty."
            ),
        )

    try:

        gesture_id = slugify(
            name
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    gesture_dir = (
        get_gesture_dir(
            gesture_id
        )
    )

    if gesture_dir.exists():

        raise HTTPException(
            status_code=409,
            detail=(
                f"Gesture '{gesture_id}' "
                "already exists."
            ),
        )

    (
        gesture_dir
        / "positive"
    ).mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        gesture_dir
        / "negative"
    ).mkdir(
        parents=True,
        exist_ok=False,
    )

    metadata = {

        "id":
            gesture_id,

        "name":
            name,

        "target":
            payload.target,

        "created_at":
            now_iso(),
    }

    write_json(
        gesture_dir
        / "gesture.json",
        metadata,
    )

    return gesture_with_counts(
        metadata
    )


# ============================================================
# LIST GESTURES
# ============================================================

@app.get(
    "/api/gestures"
)
def list_gestures():

    gestures: list[
        dict
    ] = []

    for gesture_dir in sorted(
        GESTURES_DIR.iterdir()
    ):

        if not gesture_dir.is_dir():
            continue

        metadata_file = (
            gesture_dir
            / "gesture.json"
        )

        if not metadata_file.exists():
            continue

        try:

            metadata = read_json(
                metadata_file
            )

        except Exception:

            continue

        gestures.append(
            gesture_with_counts(
                metadata
            )
        )

    return gestures


# ============================================================
# DELETE SAMPLE
# ============================================================

@app.delete(
    "/api/gestures/{gesture_id}/samples/{sample_type}/{sample_id}"
)
def delete_gesture_sample(
    gesture_id: str,
    sample_type: str,
    sample_id: str,
):
    """
    Permanently remove one recorded gesture sample.

    The complete sample directory is deleted, including:

        original.webm
        sample.json
        features.json
        features.npz
    """

    if sample_type not in {
        "positive",
        "negative",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "sample_type must be "
                "'positive' or 'negative'."
            ),
        )


    gesture_dir = (
        GESTURES_DIR
        / gesture_id
    )


    if not gesture_dir.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                f"Gesture '{gesture_id}' "
                "does not exist."
            ),
        )


    sample_dir = (
        gesture_dir
        / sample_type
        / sample_id
    )


    if not sample_dir.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                f"Sample '{sample_id}' "
                "does not exist."
            ),
        )


    if not sample_dir.is_dir():

        raise HTTPException(
            status_code=400,
            detail=(
                "Sample path is not "
                "a directory."
            ),
        )


    try:

        shutil.rmtree(
            sample_dir
        )

    except OSError as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not delete sample."
            ),
        ) from exc


    return {
        "status": "deleted",
        "gesture_id": gesture_id,
        "sample_type": sample_type,
        "sample_id": sample_id,
    }


# ============================================================
# TRAIN GESTURE MODEL
# ============================================================

@app.post(
    "/api/gestures/{gesture_id}/train"
)
def train_gesture_model(
    gesture_id: str,
):
    """
    Train and evaluate the Decision Tree model for one gesture.

    Current strategy:

        complete dataset
              ↓
        stratified 80/20 split
              ↓
        80% training
              ↓
        model.fit()
              ↓
        20% holdout validation
              ↓
        accuracy
        precision
        recall
        F1
        confusion matrix

    The trained model and all associated metadata are persisted
    under data/models/<gesture_id>/decision_tree/.
    """

    try:

        # ====================================================
        # TRAIN + VALIDATE
        # ====================================================

        result = train_gesture(
            gesture_id,
            model_name=(
                "decision_tree"
            ),
            representation_name=(
                "temporal_pyramid"
            ),
        )


        # ====================================================
        # SAVE MODEL + EVALUATION
        # ====================================================

        stored_training = (
            save_training_result(
                result
            )
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        return {
            "status": "trained",

            "gesture": (
                stored_training.get(
                    "gesture",
                    {},
                )
            ),

            "model": (
                stored_training.get(
                    "model",
                    {},
                )
            ),

            "representation": (
                stored_training.get(
                    "representation",
                    {},
                )
            ),

            "dataset": (
                stored_training.get(
                    "dataset",
                    {},
                )
            ),

            "evaluation": (
                stored_training.get(
                    "evaluation",
                    {},
                )
            ),

            "decision_tree": (
                stored_training.get(
                    "decision_tree",
                    {},
                )
            ),

            "validation_samples": (
                stored_training.get(
                    "validation_samples",
                    [],
                )
            ),

            "trained_at": (
                stored_training.get(
                    "trained_at"
                )
            ),
        }


    # ========================================================
    # DATASET
    # ========================================================

    except DatasetLoadError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(
                exc
            ),
        ) from exc


    # ========================================================
    # TRAINING / SPLIT / REPRESENTATION
    # ========================================================

    except TrainingPipelineError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(
                exc
            ),
        ) from exc


    # ========================================================
    # MODEL STORAGE
    # ========================================================

    except ModelStoreError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(
                exc
            ),
        ) from exc


    # ========================================================
    # OTHER VALIDATION ERRORS
    # ========================================================

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(
                exc
            ),
        ) from exc

# ============================================================
# GET LAST TRAINING
# ============================================================

@app.get(
    "/api/gestures/{gesture_id}/training"
)
def get_gesture_training(
    gesture_id: str,
):
    """
    Return the latest stored training information for one
    gesture.

    The endpoint also compares the current gesture dataset
    against the dataset used during the last training run.

    Possible states:

        no_training
            No persisted model exists yet.

        current
            Current samples match the samples used for training.

        outdated
            Samples were added or removed since training.

        unknown
            The current dataset could not be inspected safely.
    """

    model_name = "decision_tree"


    # ========================================================
    # NO STORED TRAINING
    # ========================================================

    if not training_exists(
        gesture_id,
        model_name,
    ):

        return {
            "status": "no_training",
            "gesture_id": gesture_id,
            "model_name": model_name,
            "training": None,
        }


    # ========================================================
    # LOAD STORED TRAINING
    # ========================================================

    try:

        stored_training = (
            load_training_info(
                gesture_id,
                model_name,
            )
        )

    except ModelStoreError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


    # ========================================================
    # CURRENT DATASET STATUS
    # ========================================================

    dataset_status = {
        "status": "unknown",

        "changed": None,

        "stored_signature": (
            stored_training
            .get(
                "dataset",
                {},
            )
            .get(
                "signature"
            )
        ),

        "current_signature": None,

        "stored_samples": (
            stored_training
            .get(
                "dataset",
                {},
            )
            .get(
                "samples"
            )
        ),

        "current_samples": None,

        "stored_positive": (
            stored_training
            .get(
                "dataset",
                {},
            )
            .get(
                "positive"
            )
        ),

        "current_positive": None,

        "stored_negative": (
            stored_training
            .get(
                "dataset",
                {},
            )
            .get(
                "negative"
            )
        ),

        "current_negative": None,
    }


    try:

        current_samples = (
            load_gesture_samples(
                gesture_id
            )
        )


        current_sample_ids = [
            sample.sample_id
            for sample in current_samples
        ]


        current_sample_types = [
            sample.sample_type
            for sample in current_samples
        ]


        current_signature = (
            build_dataset_signature(
                current_sample_ids,
                current_sample_types,
            )
        )


        current_positive = sum(
            1
            for sample in current_samples
            if sample.label == 1
        )


        current_negative = sum(
            1
            for sample in current_samples
            if sample.label == 0
        )


        stored_signature = (
            dataset_status[
                "stored_signature"
            ]
        )


        changed = (
            current_signature
            != stored_signature
        )


        dataset_status.update(
            {
                "status": (
                    "outdated"
                    if changed
                    else "current"
                ),

                "changed": changed,

                "current_signature": (
                    current_signature
                ),

                "current_samples": len(
                    current_samples
                ),

                "current_positive": (
                    current_positive
                ),

                "current_negative": (
                    current_negative
                ),
            }
        )

    except DatasetLoadError as exc:

        dataset_status.update(
            {
                "status": "unknown",

                "changed": None,

                "error": str(
                    exc
                ),
            }
        )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "status": "trained",

        "gesture_id": gesture_id,

        "model_name": model_name,

        "dataset_status": (
            dataset_status
        ),

        "training": (
            stored_training
        ),
    }

# ============================================================
# GET GESTURE
# ============================================================

@app.get(
    "/api/gestures/{gesture_id}"
)
def get_gesture(
    gesture_id: str,
):

    metadata = (
        get_gesture_metadata(
            gesture_id
        )
    )

    return gesture_with_counts(
        metadata
    )


# ============================================================
# CREATE SAMPLE
# ============================================================

@app.post(
    "/api/gestures/{gesture_id}/samples",
    status_code=201,
)
async def create_sample(
    gesture_id: str,

    sample_type: Literal[
        "positive",
        "negative",
    ] = Form(...),

    trim_start_ms: int = Form(...),

    trim_end_ms: int = Form(...),

    video: UploadFile = File(...),
):

    gesture = (
        get_gesture_metadata(
            gesture_id
        )
    )

    gesture_dir = (
        get_gesture_dir(
            gesture_id
        )
    )

    # --------------------------------------------------------
    # Validate trim
    # --------------------------------------------------------

    if trim_start_ms < 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "trim_start_ms cannot "
                "be negative."
            ),
        )

    if trim_end_ms <= trim_start_ms:

        raise HTTPException(
            status_code=400,
            detail=(
                "trim_end_ms must be greater "
                "than trim_start_ms."
            ),
        )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    sample_id = str(
        uuid.uuid4()
    )

    sample_dir = (
        gesture_dir
        / sample_type
        / sample_id
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    extension = (
        get_video_extension(
            video
        )
    )

    video_filename = (
        f"original{extension}"
    )

    video_path = (
        sample_dir
        / video_filename
    )

    try:

        video_content = (
            await video.read()
        )

        if not video_content:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded video is empty."
                ),
            )

        with video_path.open(
            "wb"
        ) as file:

            file.write(
                video_content
            )

    finally:

        await video.close()

    # --------------------------------------------------------
    # sample.json
    # --------------------------------------------------------

    metadata = {

        "id":
            sample_id,

        "gesture_id":
            gesture["id"],

        "gesture_name":
            gesture["name"],

        "target":
            gesture["target"],

        "sample_type":
            sample_type,

        "video":
            video_filename,

        "trim": {

            "start_ms":
                int(
                    trim_start_ms
                ),

            "end_ms":
                int(
                    trim_end_ms
                ),

            "duration_ms":
                int(
                    trim_end_ms
                    - trim_start_ms
                ),
        },

        "created_at":
            now_iso(),
    }

    write_json(
        sample_dir
        / "sample.json",
        metadata,
    )

    # --------------------------------------------------------
    # Feature Extraction
    # --------------------------------------------------------

    try:

        extraction_result = (
            extract_sample(
                gesture_dir=gesture_dir,
                sample_dir=sample_dir,
                label=sample_type,
                force=True,
            )
        )

        metadata[
            "feature_extraction"
        ] = {

            "status":
                extraction_result[
                    "status"
                ],

            "frame_count":
                extraction_result.get(
                    "frame_count"
                ),

            "fps":
                extraction_result.get(
                    "fps"
                ),
        }

    except Exception as exc:

        metadata[
            "feature_extraction"
        ] = {

            "status":
                "error",

            "error":
                str(exc),
        }

        print(
            "[Feature Extraction Error] "
            f"{gesture_id} / "
            f"{sample_type} / "
            f"{sample_id}: "
            f"{exc}"
        )

    return metadata


# ============================================================
# LIST SAMPLES
# ============================================================

@app.get(
    "/api/gestures/{gesture_id}/samples"
)
def list_gesture_samples(
    gesture_id: str,
):

    get_gesture_metadata(
        gesture_id
    )

    gesture_dir = (
        get_gesture_dir(
            gesture_id
        )
    )

    return {

        "positive":
            list_samples_for_type(
                gesture_dir,
                "positive",
            ),

        "negative":
            list_samples_for_type(
                gesture_dir,
                "negative",
            ),
    }


# ============================================================
# NEGATIVE LIBRARY
# ============================================================

@app.get(
    "/api/gestures/"
    "{gesture_id}/negative-library"
)
def get_negative_library(
    gesture_id: str,
):
    """
    Lista positive samples de outros gestos que podem
    ser utilizados como negativos do gesto atual.

    Apenas são apresentados gestos com o mesmo target:

        hand -> hand
        body -> body

    E apenas samples completos são disponibilizados.
    """

    destination_gesture = (
        get_gesture_metadata(
            gesture_id
        )
    )


    destination_target = (
        destination_gesture[
            "target"
        ]
    )


    library_gestures: list[
        dict
    ] = []


    for source_gesture_dir in sorted(
        GESTURES_DIR.iterdir()
    ):

        if not source_gesture_dir.is_dir():
            continue


        source_gesture_file = (
            source_gesture_dir
            / "gesture.json"
        )


        if not source_gesture_file.is_file():
            continue


        try:

            source_gesture = (
                read_json(
                    source_gesture_file
                )
            )

        except Exception:

            continue


        source_gesture_id = (
            source_gesture.get(
                "id"
            )
        )


        # ----------------------------------------------------
        # Cannot import current gesture into itself
        # ----------------------------------------------------

        if (
            source_gesture_id
            == gesture_id
        ):
            continue


        # ----------------------------------------------------
        # Only same target
        # ----------------------------------------------------

        if (
            source_gesture.get(
                "target"
            )
            != destination_target
        ):
            continue


        positive_dir = (
            source_gesture_dir
            / "positive"
        )


        if not positive_dir.is_dir():
            continue


        samples: list[
            dict
        ] = []


        for sample_dir in sorted(
            positive_dir.iterdir()
        ):

            if not sample_dir.is_dir():
                continue


            if not is_complete_sample(
                sample_dir
            ):
                continue


            try:

                sample_metadata = (
                    read_json(
                        sample_dir
                        / "sample.json"
                    )
                )

            except Exception:

                continue


            source_sample_id = (
                sample_dir.name
            )


            trim = (
                sample_metadata.get(
                    "trim"
                )
                or {}
            )


            already_imported = (
                negative_library_source_exists(
                    gesture_id,
                    source_gesture_id,
                    source_sample_id,
                )
            )


            samples.append({

                "id":
                    source_sample_id,

                "gesture_id":
                    source_gesture_id,

                "gesture_name":
                    source_gesture.get(
                        "name",
                        source_gesture_id,
                    ),

                "target":
                    source_gesture.get(
                        "target"
                    ),

                "duration_ms":
                    trim.get(
                        "duration_ms"
                    ),

                "created_at":
                    sample_metadata.get(
                        "created_at"
                    ),

                "already_imported":
                    already_imported,
            })


        if not samples:
            continue


        library_gestures.append({

            "id":
                source_gesture_id,

            "name":
                source_gesture.get(
                    "name",
                    source_gesture_id,
                ),

            "target":
                source_gesture.get(
                    "target"
                ),

            "sample_count":
                len(
                    samples
                ),

            "available_count":
                sum(
                    1
                    for sample
                    in samples
                    if not sample[
                        "already_imported"
                    ]
                ),

            "samples":
                samples,
        })


    return {

        "gesture": {
            "id":
                destination_gesture[
                    "id"
                ],

            "name":
                destination_gesture[
                    "name"
                ],

            "target":
                destination_target,
        },

        "gestures":
            library_gestures,
    }


# ============================================================
# IMPORT NEGATIVES FROM LIBRARY
# ============================================================

@app.post(
    "/api/gestures/"
    "{gesture_id}/negative-library/import"
)
def import_negative_library_samples(
    gesture_id: str,
    payload: NegativeLibraryImportRequest,
):
    """
    Hard-copy selected positive samples from other gestures
    into the negative dataset of the current gesture.
    """

    destination_gesture = (
        get_gesture_metadata(
            gesture_id
        )
    )


    if not payload.samples:

        raise HTTPException(
            status_code=400,
            detail=(
                "No samples were selected."
            ),
        )


    # --------------------------------------------------------
    # Prevent duplicate entries inside the request itself
    # --------------------------------------------------------

    requested_sources: set[
        tuple[
            str,
            str,
        ]
    ] = set()


    for requested_sample in (
        payload.samples
    ):

        source_key = (
            requested_sample.gesture_id,
            requested_sample.sample_id,
        )


        if (
            source_key
            in requested_sources
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "The same library sample "
                    "was selected more than once."
                ),
            )


        requested_sources.add(
            source_key
        )


    # --------------------------------------------------------
    # Validate all samples before importing anything
    # --------------------------------------------------------

    validated_samples: list[
        tuple[
            dict,
            str,
        ]
    ] = []


    for requested_sample in (
        payload.samples
    ):

        source_gesture_id = (
            requested_sample
            .gesture_id
        )


        source_sample_id = (
            requested_sample
            .sample_id
        )


        if (
            source_gesture_id
            == gesture_id
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "A gesture cannot import "
                    "its own positive samples "
                    "as negatives."
                ),
            )


        source_gesture = (
            get_gesture_metadata(
                source_gesture_id
            )
        )


        if (
            source_gesture.get(
                "target"
            )
            != destination_gesture.get(
                "target"
            )
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Gesture '{source_gesture_id}' "
                    "has a different target."
                ),
            )


        source_sample_dir = (
            get_sample_dir(
                source_gesture_id,
                "positive",
                source_sample_id,
            )
        )


        if not is_complete_sample(
            source_sample_dir
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Sample '{source_sample_id}' "
                    "is incomplete and cannot "
                    "be imported."
                ),
            )


        validated_samples.append(
            (
                source_gesture,
                source_sample_id,
            )
        )


    # --------------------------------------------------------
    # Import
    # --------------------------------------------------------

    imported: list[
        dict
    ] = []


    skipped: list[
        dict
    ] = []


    for (
        source_gesture,
        source_sample_id,
    ) in validated_samples:

        result = (
            import_positive_as_negative(
                destination_gesture=(
                    destination_gesture
                ),
                source_gesture=(
                    source_gesture
                ),
                source_sample_id=(
                    source_sample_id
                ),
            )
        )


        if (
            result[
                "status"
            ]
            == "already_imported"
        ):

            skipped.append(
                result
            )

        else:

            imported.append(
                result
            )


    return {

        "status":
            "completed",

        "gesture_id":
            gesture_id,

        "imported_count":
            len(
                imported
            ),

        "skipped_count":
            len(
                skipped
            ),

        "imported":
            imported,

        "skipped":
            skipped,

        "gesture":
            gesture_with_counts(
                destination_gesture
            ),
    }

# ============================================================
# GET SAMPLE
# ============================================================

@app.get(
    "/api/gestures/"
    "{gesture_id}/samples/"
    "{sample_type}/{sample_id}"
)
def get_sample(
    gesture_id: str,
    sample_type: Literal[
        "positive",
        "negative",
    ],
    sample_id: str,
):

    get_gesture_metadata(
        gesture_id
    )

    sample_dir = (
        get_sample_dir(
            gesture_id,
            sample_type,
            sample_id,
        )
    )

    metadata = (
        get_sample_metadata(
            gesture_id,
            sample_type,
            sample_id,
        )
    )

    feature_summary = (
        get_feature_summary(
            sample_dir
        )
    )

    result = dict(
        metadata
    )

    result[
        "features_extracted"
    ] = (
        feature_summary
        is not None
    )

    result[
        "feature_summary"
    ] = (
        feature_summary
    )

    result[
        "video_url"
    ] = (
        f"/api/gestures/"
        f"{gesture_id}/samples/"
        f"{sample_type}/"
        f"{sample_id}/video"
    )

    result[
        "landmarks_url"
    ] = (
        f"/api/gestures/"
        f"{gesture_id}/samples/"
        f"{sample_type}/"
        f"{sample_id}/landmarks"
    )

    return result


# ============================================================
# SAMPLE VIDEO
# ============================================================

@app.get(
    "/api/gestures/"
    "{gesture_id}/samples/"
    "{sample_type}/{sample_id}/video"
)
def get_sample_video(
    gesture_id: str,
    sample_type: Literal[
        "positive",
        "negative",
    ],
    sample_id: str,
):

    get_gesture_metadata(
        gesture_id
    )

    sample_dir = (
        get_sample_dir(
            gesture_id,
            sample_type,
            sample_id,
        )
    )

    metadata = (
        get_sample_metadata(
            gesture_id,
            sample_type,
            sample_id,
        )
    )

    video_filename = (
        metadata.get(
            "video"
        )
    )

    if not video_filename:

        raise HTTPException(
            status_code=404,
            detail=(
                "Video filename not found "
                "in sample metadata."
            ),
        )

    video_path = (
        sample_dir
        / str(
            video_filename
        )
    )

    if not video_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Sample video not found."
            ),
        )

    media_type, _ = (
        mimetypes.guess_type(
            str(
                video_path
            )
        )
    )

    if media_type is None:

        media_type = (
            "application/octet-stream"
        )

    return FileResponse(
        path=str(
            video_path
        ),
        media_type=media_type,
        filename=None,
    )


# ============================================================
# SAMPLE LANDMARKS
# ============================================================

@app.get(
    "/api/gestures/"
    "{gesture_id}/samples/"
    "{sample_type}/{sample_id}/landmarks"
)
def get_sample_landmarks(
    gesture_id: str,
    sample_type: Literal[
        "positive",
        "negative",
    ],
    sample_id: str,
):
    """
    Devolve apenas timestamps + landmarks utilizados
    na Feature Extraction.

    Os timestamps começam em 0 relativamente ao início
    do clip selecionado, não relativamente ao início
    do vídeo original.
    """

    get_gesture_metadata(
        gesture_id
    )

    sample_dir = (
        get_sample_dir(
            gesture_id,
            sample_type,
            sample_id,
        )
    )

    metadata = (
        get_sample_metadata(
            gesture_id,
            sample_type,
            sample_id,
        )
    )

    try:

        timestamps, landmarks = (
            load_landmark_sequence(
                sample_dir
            )
        )

    except FileNotFoundError:

        raise HTTPException(
            status_code=404,
            detail=(
                "Feature extraction is not "
                "available for this sample."
            ),
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Invalid feature data: "
                f"{exc}"
            ),
        )

    frame_count = int(
        landmarks.shape[0]
    )

    landmark_count = int(
        landmarks.shape[1]
    )

    return {

        "gesture_id":
            gesture_id,

        "sample_id":
            sample_id,

        "sample_type":
            sample_type,

        "target":
            metadata.get(
                "target"
            ),

        "trim":
            metadata.get(
                "trim",
                {},
            ),

        "timestamp_reference":
            "clip_start",

        "frame_count":
            frame_count,

        "landmark_count":
            landmark_count,

        "timestamps":
            timestamps_to_json_safe(
                timestamps
            ),

        "landmarks":
            landmarks_to_json_safe(
                landmarks
            ),
    }


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory=str(
            FRONTEND_DIR
        ),
        html=True,
    ),
    name="frontend",
)