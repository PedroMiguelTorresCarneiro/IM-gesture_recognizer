# IM-Gesture_Recognizer


System for capturing gesture samples, extracting motion features from video, and training gesture classifiers.

The current pipeline supports **hand** and **body** gestures, using MediaPipe landmarks and a feature-based machine learning pipeline.

---

# How to run

## 1. Create and activate the virtual environment

Python **3.11** is recommended.

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Upgrade the packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install the project

From the project root:

```bash
pip install -e .
```

This installs the project dependencies and exposes the `gesture-capture` CLI command.

## 3. Start the application

```bash
gesture-capture serve
```

By default, the application is available at:

```text
http://127.0.0.1:8000
```

The server can also be configured explicitly:

```bash
gesture-capture serve --host 0.0.0.0 --port 8000
```

To disable automatic reload:

```bash
gesture-capture serve --no-reload
```

Alternatively, the FastAPI application can be started directly:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Feature extraction from the CLI

Feature extraction normally runs automatically when a sample is confirmed in the interface.

It can also be executed manually.

Extract all samples from one gesture:

```bash
gesture-capture extract --gesture swipe-left
```

Force regeneration of existing features:

```bash
gesture-capture extract --gesture swipe-left --force
```

Extract all gestures:

```bash
gesture-capture extract --all
```

Force extraction for all gestures:

```bash
gesture-capture extract --all --force
```

---

# Project overview

The project is divided into four main areas:

```text
backend/
├── cli.py
├── main.py
├── feature_extraction/
└── training/

frontend/
├── index.html
├── app.js
└── styles.css

data/
├── gestures/
└── models/
```

The overall processing pipeline is:

```text
Gesture Creation
      ↓
Video Capture
      ↓
Trim Selection
      ↓
Sample Confirmation
      ↓
Feature Extraction
      ↓
Temporal Representation
      ↓
Classifier Training
      ↓
Model + Metrics + Decision Rules
```

---

# Backend

## `backend/main.py`

Main FastAPI application.

It is responsible for exposing the API used by the frontend and coordinating the main application operations.

The API handles operations such as:

- creating gestures;
- listing existing gestures;
- storing gesture metadata;
- storing positive and negative samples;
- receiving recorded video clips;
- storing trim information;
- triggering feature extraction after a clip is confirmed;
- listing and retrieving gesture samples;
- importing compatible samples from the gesture library;
- triggering classifier training;
- retrieving the latest training results and training status.

It also serves the frontend application.

---

## `backend/cli.py`

Command-line interface for the project.

It exposes the `gesture-capture` command and currently provides the main commands:

```text
gesture-capture serve
gesture-capture extract
```

`serve` starts the FastAPI application.

`extract` executes the feature extraction pipeline manually for one gesture or for all gestures.

---

# Feature Extraction

The feature extraction pipeline is located in:

```text
backend/feature_extraction/
```

Its objective is to transform each recorded gesture sample into numerical time-series features that can later be used for training.

The current V1 intentionally preserves the original gesture duration and does **not** apply additional smoothing, temporal resampling, or normalization.

---

<!-- ## `feature_extraction/config.py`

Contains configuration used by the feature extraction pipeline.

It centralizes parameters and constants required during video processing and feature generation.

--- -->

## `feature_extraction/pipeline.py`

Main feature extraction pipeline.

For each sample, it:

1. loads the recorded video;
2. reads the trim interval from `sample.json`;
3. processes only the selected temporal interval;
4. decodes frames using PyAV;
5. preserves real frame timestamps using video PTS;
6. extracts MediaPipe landmarks;
7. computes geometric and kinematic features;
8. stores the resulting feature sequence.

Feature extraction is automatically triggered when a recorded clip is confirmed.

---

# Landmark Extractors

Located in:

```text
backend/feature_extraction/extractors/
```

## `extractors/hand.py`

Processes gestures whose target is:

```text
hand
```

Uses MediaPipe Hand landmarks.

Each detected hand contains **21 landmarks**.

The extractor converts MediaPipe output into the internal representation used by the feature pipeline.

---

## `extractors/pose.py`

Processes gestures whose target is:

```text
body
```

Uses MediaPipe Pose landmarks.

Each detected pose contains **33 landmarks**.

The resulting landmark sequence is used to calculate body movement features.

---

# Feature Computation

Located in:

```text
backend/feature_extraction/features/
```

## `features/distances.py`

Calculates geometric distances between selected landmark pairs.

These distances describe the spatial configuration of the hand or body at each frame.

---

## `features/kinematics.py`

Calculates motion-based features from the landmark trajectories.

The current pipeline includes:

- velocity;
- speed;
- acceleration;
- acceleration magnitude.

These values are calculated over time using the real timestamps extracted from the video.

---

# Feature Storage

Located in:

```text
backend/feature_extraction/storage/
```

<!-- ## `storage/reader.py`

Contains utilities for loading sample metadata and previously generated feature data.

--- -->

## `storage/writer.py`

Responsible for persisting extracted features.

Each processed sample contains:

```text
features.npz
features.json
```

### `features.npz`

Stores the numerical arrays used by the training pipeline.

It currently contains data such as:

```text
timestamps
landmarks
distances
distance_pairs
velocities
speeds
accelerations
acceleration_magnitudes
```

### `features.json`

Stores readable metadata describing the extracted features and sample processing information.

---

# Training

The training pipeline is located in:

```text
backend/training/
```

Its objective is to convert variable-length gesture sequences into fixed-size vectors and train a classifier for each gesture.

The current default training pipeline is:

```text
Extracted Features
       ↓
Temporal Pyramid
       ↓
Fixed-size Feature Vector
       ↓
Decision Tree
       ↓
Model + Metrics + Rules
```

---

<!-- ## `training/config.py`

Contains configuration used during classifier training.

This includes parameters shared by the dataset, representation, and model training stages.

The training pipeline uses a fixed `RANDOM_STATE` so that operations such as dataset splitting and model training remain reproducible.

--- -->

## `training/pipeline.py`

Main training orchestrator.

It coordinates:

1. loading the gesture dataset;
2. validating available samples;
3. converting each feature sequence into a fixed-size representation;
4. creating the training and validation sets;
5. training the selected model;
6. evaluating the classifier;
7. extracting model information;
8. persisting all training artefacts.

The current default combination is:

```text
Temporal Pyramid + Decision Tree
```

---

# Training Dataset

Located in:

```text
backend/training/dataset/
```

## `dataset/loader.py`

Loads samples from:

```text
data/gestures/<gesture_id>/positive/
data/gestures/<gesture_id>/negative/
```

Labels are assigned according to the directory:

```text
positive = 1
negative = 0
```

The loader also validates sample metadata, including:

- sample identifier;
- gesture identifier;
- sample type;
- gesture target.

Only samples compatible with the gesture target are used.

For example:

```text
hand → hand
body → body
```

---

# Temporal Representation

Located in:

```text
backend/training/representation/
```

## `representation/temporal_pyramid.py`

Gesture samples have variable durations, so their extracted feature sequences do not all contain the same number of frames.

The Temporal Pyramid converts each variable-length sequence into a fixed-size feature vector.

The current pyramid uses:

```text
1 + 2 + 4
```

temporal regions, for a total of:

```text
7 regions
```

For each region, statistical information is computed from the original temporal features.

This preserves information about both the complete gesture and different stages of its execution.

With the current feature configuration, the representation produces:

```text
HAND → 3087 features
BODY → 6237 features
```

The generated feature names are preserved so trained models can later identify exactly which original feature contributed to a decision.

---

# Models

Located in:

```text
backend/training/models/
```

## `models/base.py`

Defines the common interface used by training models.

It provides the abstraction required so different classifiers can be integrated into the same training pipeline.

---

## `models/decision_tree.py`

Decision Tree classifier implementation.

This is the current default classifier.

Besides predictions, a Decision Tree provides useful interpretability information:

- learned decision rules;
- features used by the tree;
- `feature_importances_`;
- decision thresholds.

Because the Temporal Pyramid preserves feature names, each rule can be mapped back to the exact generated feature.

Example:

```text
R1_acceleration_5_x_mean <= 0.95
```

This makes the classifier particularly useful during development and analysis of gesture features.

---

## `models/random_forest.py`

Random Forest model implementation.

> TO BE IMPLEMENTED

---

# Model Storage

Located in:

```text
backend/training/storage/
```

## `storage/model_store.py`

Persists the trained model and the information generated during training.

Models are stored under:

```text
data/models/<gesture_id>/<model_name>/
```

For the current Decision Tree classifier:

```text
data/models/<gesture_id>/decision_tree/
├── model.joblib
├── metadata.json
├── feature_names.json
├── metrics.json
├── feature_importance.json
├── decision_rules.txt
└── validation_samples.json
```

### Main artefacts

`model.joblib`

Serialized trained classifier.

`metadata.json`

General information about the training process, representation, dataset, and model.

`feature_names.json`

Ordered list of features used to train the classifier.

`metrics.json`

Evaluation results, including metrics such as:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix.

`feature_importance.json`

Feature importance values generated by the model.

`decision_rules.txt`

Human-readable rules learned by the Decision Tree.

`validation_samples.json`

Stores the validation samples and their predictions, allowing the interface to show how the model behaved on unseen samples.

---

<!-- # Frontend

The frontend is located in:

```text
frontend/
```

It provides the browser-based interface used to create gestures, record samples, inspect extracted landmarks, and train classifiers.

---

## `frontend/index.html`

Defines the main structure of the web interface.

---

## `frontend/app.js`

Contains the main frontend logic.

It handles operations such as:

- loading gestures from the backend;
- creating gestures;
- selecting `hand` or `body` targets;
- accessing the webcam;
- recording videos with `MediaRecorder`;
- creating positive and negative samples;
- previewing recorded clips;
- selecting the trim interval;
- confirming and uploading samples;
- displaying sample lists;
- visualizing original video and extracted landmarks;
- importing compatible samples from the gesture library;
- starting training;
- displaying training metrics, decision rules, feature importance, and validation predictions.

---

## `frontend/styles.css`

Contains the visual layout and styling of the application.

The sample visualization interface displays the original video and the landmark visualization side by side.

--- -->

# Data

Application data is stored locally under:

```text
data/
```

---

## Gestures

Each gesture has its own directory:

```text
data/gestures/<gesture_id>/
```

Example:

```text
data/gestures/swipe-left/
├── gesture.json
├── positive/
└── negative/
```

`gesture.json` stores the gesture configuration and metadata.

The gesture target is currently either:

```text
hand
body
```

---

## Samples

Each positive or negative sample has its own directory.

Example:

```text
data/gestures/swipe-left/positive/<sample_id>/
├── original.webm
├── sample.json
├── features.json
└── features.npz
```

### `original.webm`

Original video recorded by the browser.

### `sample.json`

Sample metadata, including the gesture, sample type, target, and trim interval.

### `features.json`

Readable feature extraction metadata.

### `features.npz`

Numerical feature arrays used by the training pipeline.

---

# Sample Library

Existing compatible samples can be reused between gestures.

The library only exposes samples with the same gesture target.

For example:

```text
hand gesture → hand samples only
body gesture → body samples only
```

Imported samples are physically copied into the destination gesture.

The imported `sample.json` and `features.json` metadata are updated for the new gesture, while the already extracted numerical data in `features.npz` can be reused unchanged.

---

<!-- # Training from the API

Training can be triggered through the frontend or directly through the API.

Example:

```bash
curl -X POST \
  http://127.0.0.1:8000/api/gestures/swipe-left/train
```

The latest training information can be retrieved with:

```bash
curl \
  http://127.0.0.1:8000/api/gestures/swipe-left/training
```

The training state indicates whether the stored model is still current or whether the dataset has changed since the last training.

--- -->

# Current pipeline

At the current stage, the project supports:

```text
Webcam Capture
    ↓
Positive / Negative Samples
    ↓
Manual Trim
    ↓
MediaPipe Hand / Pose
    ↓
Landmarks
    ↓
Distances
    ↓
Velocity / Speed
    ↓
Acceleration
    ↓
Temporal Pyramid
    ↓
Decision Tree
    ↓
Classification Rules + Metrics
```
