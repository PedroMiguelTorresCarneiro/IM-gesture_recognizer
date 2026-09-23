const state = {

    gesture: null,

    sampleType: null,

    stream: null,

    recorder: null,

    chunks: [],

    videoBlob: null,

    videoURL: null,

    samples: {
        positive: [],
        negative: [],
    },

    selectedSample: null,

    landmarkData: null,

    landmarkAnimationFrame: null,

    showLandmarks: true,

    trainingResult: null,

    trainingDatasetStatus: null,

    trainingInProgress: false,

};


// ==========================================================
// SCREENS
// ==========================================================

const screens = {

    gesture:
        document.getElementById(
            "screen-gesture"
        ),

    samples:
        document.getElementById(
            "screen-samples"
        ),

    capture:
        document.getElementById(
            "screen-capture"
        ),

    trim:
        document.getElementById(
            "screen-trim"
        ),

};


// ==========================================================
// VIDEO ELEMENTS
// ==========================================================

const cameraPreview =
    document.getElementById(
        "camera-preview"
    );


const reviewVideo =
    document.getElementById(
        "review-video"
    );


const trimStart =
    document.getElementById(
        "trim-start"
    );


const trimEnd =
    document.getElementById(
        "trim-end"
    );


const trimStartLabel =
    document.getElementById(
        "trim-start-label"
    );


const trimEndLabel =
    document.getElementById(
        "trim-end-label"
    );


const selectionInfo =
    document.getElementById(
        "selection-info"
    );



// ==========================================================
// TRAINING ELEMENTS
// ==========================================================

const trainModelButton =
    document.getElementById(
        "train-model"
    );


const trainingStatus =
    document.getElementById(
        "training-status"
    );


const trainingResults =
    document.getElementById(
        "training-results"
    );


const trainingSampleCount =
    document.getElementById(
        "training-sample-count"
    );


const trainingPositiveCount =
    document.getElementById(
        "training-positive-count"
    );


const trainingNegativeCount =
    document.getElementById(
        "training-negative-count"
    );


const trainingFeatureCount =
    document.getElementById(
        "training-feature-count"
    );


const trainingTreeDepth =
    document.getElementById(
        "training-tree-depth"
    );


const trainingLeafCount =
    document.getElementById(
        "training-leaf-count"
    );


const trainingUsedFeatureCount =
    document.getElementById(
        "training-used-feature-count"
    );


const trainingAccuracy =
    document.getElementById(
        "training-accuracy"
    );


const trainingWarning =
    document.getElementById(
        "training-warning"
    );


const trainingDecisionRules =
    document.getElementById(
        "training-decision-rules"
    );


const trainingFeaturesBody =
    document.getElementById(
        "training-features-body"
    );


const trainingSamplesBody =
    document.getElementById(
        "training-samples-body"
    );

// ==========================================================
// SCREEN MANAGEMENT
// ==========================================================

function showScreen(
    name
) {

    Object.values(
        screens
    ).forEach(
        screen => {

            screen.classList.add(
                "hidden"
            );

        }
    );


    screens[
        name
    ].classList.remove(
        "hidden"
    );

}


// ==========================================================
// GENERIC HELPERS
// ==========================================================

function escapeHTML(
    value
) {

    return String(
        value ?? ""
    )
        .replaceAll(
            "&",
            "&amp;"
        )
        .replaceAll(
            "<",
            "&lt;"
        )
        .replaceAll(
            ">",
            "&gt;"
        )
        .replaceAll(
            '"',
            "&quot;"
        )
        .replaceAll(
            "'",
            "&#039;"
        );

}


function formatDurationMs(
    milliseconds
) {

    if (
        milliseconds === null
        ||
        milliseconds === undefined
    ) {

        return "—";

    }


    const seconds =
        Number(
            milliseconds
        ) / 1000;


    if (
        !Number.isFinite(
            seconds
        )
    ) {

        return "—";

    }


    return (
        `${seconds.toFixed(2)} s`
    );

}


function formatFPS(
    fps
) {

    const value =
        Number(
            fps
        );


    if (
        !Number.isFinite(
            value
        )
    ) {

        return "—";

    }


    return (
        value.toFixed(
            2
        )
    );

}


function formatDate(
    value
) {

    if (!value) {
        return "—";
    }


    const date =
        new Date(
            value
        );


    if (
        Number.isNaN(
            date.getTime()
        )
    ) {

        return value;

    }


    return date.toLocaleString();

}


function shortSampleId(
    sampleId
) {

    if (!sampleId) {
        return "Unknown";
    }


    return (
        sampleId.length > 16
            ? `${sampleId.slice(0, 8)}…`
            : sampleId
    );

}


// ==========================================================
// MEDIAPIPE CONNECTIONS
// ==========================================================

const HAND_CONNECTIONS = [

    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],

    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],

    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],

    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],

    [13, 17],
    [17, 18],
    [18, 19],
    [19, 20],

    [0, 17],

];


const POSE_CONNECTIONS = [

    // Face

    [0, 1],
    [1, 2],
    [2, 3],
    [3, 7],

    [0, 4],
    [4, 5],
    [5, 6],
    [6, 8],

    [9, 10],


    // Shoulders

    [11, 12],


    // Left arm

    [11, 13],
    [13, 15],

    [15, 17],
    [15, 19],
    [15, 21],

    [17, 19],


    // Right arm

    [12, 14],
    [14, 16],

    [16, 18],
    [16, 20],
    [16, 22],

    [18, 20],


    // Torso

    [11, 23],
    [12, 24],
    [23, 24],


    // Left leg

    [23, 25],
    [25, 27],
    [27, 29],
    [29, 31],
    [27, 31],


    // Right leg

    [24, 26],
    [26, 28],
    [28, 30],
    [30, 32],
    [28, 32],

];


// ==========================================================
// SAMPLE LIST MODAL
// ==========================================================

function ensureSamplesModal() {

    let modal =
        document.getElementById(
            "samples-modal"
        );


    if (modal) {
        return modal;
    }


    modal =
        document.createElement(
            "div"
        );


    modal.id =
        "samples-modal";


    modal.className =
        "samples-modal hidden";


    modal.innerHTML = `

        <div
            class="samples-modal-backdrop"
            data-close-samples-modal
        ></div>

        <div
            class="samples-modal-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="samples-modal-title"
        >

            <div
                class="samples-modal-header"
            >

                <div>

                    <h2
                        id="samples-modal-title"
                    >
                        Examples
                    </h2>

                    <p
                        id="samples-modal-subtitle"
                    >
                    </p>

                </div>

                <button
                    type="button"
                    id="close-samples-modal"
                    class="samples-modal-close"
                    aria-label="Close"
                >
                    ×
                </button>

            </div>


            <div
                id="samples-modal-content"
                class="samples-modal-content"
            >

                <div
                    class="empty-state"
                >
                    Loading...
                </div>

            </div>

        </div>
    `;


    document.body.appendChild(
        modal
    );


    modal
        .querySelector(
            "#close-samples-modal"
        )
        .addEventListener(
            "click",
            closeSamplesModal
        );


    modal
        .querySelector(
            "[data-close-samples-modal]"
        )
        .addEventListener(
            "click",
            closeSamplesModal
        );


    return modal;

}


function closeSamplesModal() {

    const modal =
        document.getElementById(
            "samples-modal"
        );


    if (!modal) {
        return;
    }


    modal.classList.add(
        "hidden"
    );

}


// ==========================================================
// SAMPLE VIEWER MODAL
// ==========================================================

function ensureSampleViewerModal() {

    let modal =
        document.getElementById(
            "sample-viewer-modal"
        );


    if (modal) {
        return modal;
    }


    modal =
        document.createElement(
            "div"
        );


    modal.id =
        "sample-viewer-modal";


    modal.className =
        "samples-modal hidden";


    modal.innerHTML = `

        <div
            class="samples-modal-backdrop"
            data-close-sample-viewer
        ></div>


        <div
            class="samples-modal-panel sample-viewer-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="sample-viewer-title"
        >

            <div
                class="samples-modal-header"
            >

                <div>

                    <h2
                        id="sample-viewer-title"
                    >
                        Sample
                    </h2>

                    <p
                        id="sample-viewer-subtitle"
                    >
                    </p>

                </div>


                <button
                    type="button"
                    id="close-sample-viewer"
                    class="samples-modal-close"
                    aria-label="Close"
                >
                    ×
                </button>

            </div>


            <div
                class="samples-modal-content"
            >

                <div
                    id="sample-viewer-loading"
                    class="empty-state"
                >
                    Loading sample...
                </div>


                <div
                    id="sample-viewer-content"
                    class="hidden"
                >

                    <div
                        class="sample-comparison"
                    >

                        <div
                            class="sample-view-column"
                        >

                            <div
                                class="sample-view-title"
                            >
                                Original video
                            </div>

                            <div
                                class="video-container sample-original-stage"
                            >

                                <video
                                    id="sample-viewer-video"
                                    controls
                                    preload="metadata"
                                ></video>

                            </div>

                        </div>


                        <div
                            class="sample-view-column"
                        >

                            <div
                                class="sample-view-title"
                            >
                                MediaPipe landmarks
                            </div>

                            <div
                                id="sample-landmark-stage"
                                class="landmark-only-stage"
                            >

                                <canvas
                                    id="sample-landmark-canvas"
                                    aria-hidden="true"
                                ></canvas>

                            </div>

                        </div>

                    </div>


                    <div
                        class="sample-viewer-controls"
                    >

                        <button
                            type="button"
                            id="play-sample-clip"
                            class="primary"
                        >
                            Play clip
                        </button>


                        <label
                            class="landmark-toggle"
                        >

                            <input
                                type="checkbox"
                                id="show-sample-landmarks"
                                checked
                            >

                            Show landmarks

                        </label>


                        <span
                            id="sample-viewer-range"
                        >
                        </span>

                    </div>


                    <div
                        id="sample-viewer-details"
                    >
                    </div>

                </div>

            </div>

        </div>
    `;


    document.body.appendChild(
        modal
    );


    modal
        .querySelector(
            "#close-sample-viewer"
        )
        .addEventListener(
            "click",
            closeSampleViewer
        );


    modal
        .querySelector(
            "[data-close-sample-viewer]"
        )
        .addEventListener(
            "click",
            closeSampleViewer
        );


    modal
        .querySelector(
            "#play-sample-clip"
        )
        .addEventListener(
            "click",
            playSelectedSampleClip
        );


    const video =
        modal.querySelector(
            "#sample-viewer-video"
        );


    video.addEventListener(
        "loadedmetadata",
        () => {

            seekSampleVideoToStart();


            const landmarkStage =
                document.getElementById(
                    "sample-landmark-stage"
                );


            if (
                landmarkStage
                &&
                video.videoWidth
                &&
                video.videoHeight
            ) {

                landmarkStage.style.aspectRatio =
                    `${video.videoWidth} / `
                    + `${video.videoHeight}`;

            }


            resizeLandmarkCanvas();

            drawCurrentLandmarks();

        }
    );


    video.addEventListener(
        "play",
        () => {

            enforceSamplePlaybackStart();

            startLandmarkAnimation();

        }
    );

    video.addEventListener(
        "pause",
        () => {

            stopLandmarkAnimation();

            drawCurrentLandmarks();

        }
    );


    video.addEventListener(
        "seeked",
        () => {

            drawCurrentLandmarks();

        }
    );

    video.addEventListener(
        "timeupdate",
        () => {

            enforceSamplePlaybackEnd();

            drawCurrentLandmarks();

        }
    );


    modal
        .querySelector(
            "#show-sample-landmarks"
        )
        .addEventListener(
            "change",
            event => {

                state.showLandmarks =
                    event.target.checked;


                drawCurrentLandmarks();

            }
        );


    return modal;

}


function closeSampleViewer() {

    const modal =
        document.getElementById(
            "sample-viewer-modal"
        );


    if (!modal) {
        return;
    }


    const video =
        modal.querySelector(
            "#sample-viewer-video"
        );


    if (video) {

        video.pause();

        video.removeAttribute(
            "src"
        );

        video.load();

    }


    modal.classList.add(
        "hidden"
    );

    stopLandmarkAnimation();


    clearLandmarkCanvas();


    state.landmarkData =
        null;

    state.selectedSample =
        null;

}


// ==========================================================
// SAMPLE VIEWER TIMING
// ==========================================================

function getSelectedSampleTrim() {

    const trim =
        state.selectedSample
            ?.trim;


    if (!trim) {

        return {
            start: 0,
            end: Infinity,
        };

    }


    return {

        start:
            Number(
                trim.start_ms
                ?? 0
            ) / 1000,

        end:
            Number(
                trim.end_ms
                ?? Infinity
            ) / 1000,

    };

}


function seekSampleVideoToStart() {

    const video =
        document.getElementById(
            "sample-viewer-video"
        );


    if (!video) {
        return;
    }


    const trim =
        getSelectedSampleTrim();


    if (
        Number.isFinite(
            trim.start
        )
    ) {

        video.currentTime =
            trim.start;

    }

}


function enforceSamplePlaybackStart() {

    const video =
        document.getElementById(
            "sample-viewer-video"
        );


    if (!video) {
        return;
    }


    const trim =
        getSelectedSampleTrim();


    if (
        video.currentTime
        < trim.start
        ||
        video.currentTime
        >= trim.end
    ) {

        video.currentTime =
            trim.start;

    }

}


function enforceSamplePlaybackEnd() {

    const video =
        document.getElementById(
            "sample-viewer-video"
        );


    if (!video) {
        return;
    }


    const trim =
        getSelectedSampleTrim();


    if (
        video.currentTime
        >= trim.end
    ) {

        video.pause();

        video.currentTime =
            trim.start;

    }

}


async function playSelectedSampleClip() {

    const video =
        document.getElementById(
            "sample-viewer-video"
        );


    if (!video) {
        return;
    }


    const trim =
        getSelectedSampleTrim();


    video.pause();


    video.currentTime =
        trim.start;


    try {

        await video.play();

    }
    catch (error) {

        console.error(
            error
        );

    }

}


// ==========================================================
// ESCAPE KEY
// ==========================================================

document.addEventListener(
    "keydown",
    event => {

        if (
            event.key !== "Escape"
        ) {
            return;
        }


        const viewer =
            document.getElementById(
                "sample-viewer-modal"
            );


        if (
            viewer
            &&
            !viewer.classList.contains(
                "hidden"
            )
        ) {

            closeSampleViewer();

            return;

        }

        const negativeLibrary =
            document.getElementById(
                "negative-library-modal"
            );


        if (
            negativeLibrary
            &&
            !negativeLibrary
                .classList
                .contains(
                    "hidden"
                )
        ) {

            closeNegativeLibraryModal();

            return;

        }


        const negativeSource =
            document.getElementById(
                "negative-source-modal"
            );


        if (
            negativeSource
            &&
            !negativeSource
                .classList
                .contains(
                    "hidden"
                )
        ) {

            closeNegativeSourceModal();

            return;

        }


        closeSamplesModal();

    }
);


// ==========================================================
// LOAD SAMPLES
// ==========================================================

async function loadSamples() {

    if (!state.gesture) {

        throw new Error(
            "No gesture selected."
        );

    }


    const response =
        await fetch(
            `/api/gestures/${state.gesture.id}/samples`
        );


    if (!response.ok) {

        throw new Error(
            "Could not load samples."
        );

    }


    const data =
        await response.json();


    const sortByNewest =
        samples => {

            return [
                ...(samples ?? [])
            ].sort(
                (a, b) => {

                    const dateA =
                        new Date(
                            a.created_at
                            ?? 0
                        ).getTime();


                    const dateB =
                        new Date(
                            b.created_at
                            ?? 0
                        ).getTime();


                    return (
                        dateB
                        - dateA
                    );

                }
            );

        };


    state.samples = {

        positive:
            sortByNewest(
                data.positive
            ),

        negative:
            sortByNewest(
                data.negative
            ),

    };

    return state.samples;

}


// ==========================================================
// LOAD INDIVIDUAL SAMPLE
// ==========================================================

async function loadSample(
    sampleType,
    sampleId
) {

    if (!state.gesture) {

        throw new Error(
            "No gesture selected."
        );

    }


    const response =
        await fetch(
            `/api/gestures/`
            + `${state.gesture.id}`
            + `/samples/`
            + `${sampleType}/`
            + `${sampleId}`
        );


    if (!response.ok) {

        throw new Error(
            "Could not load sample."
        );

    }


    return (
        await response.json()
    );

}

// ==========================================================
// DELETE SAMPLE
// ==========================================================

async function deleteSample(
    sampleType,
    sampleId
) {

    if (!state.gesture) {

        throw new Error(
            "No gesture selected."
        );

    }


    const response =
        await fetch(
            `/api/gestures/`
            + `${state.gesture.id}`
            + `/samples/`
            + `${sampleType}/`
            + `${sampleId}`,
            {
                method: "DELETE",
            }
        );


    if (!response.ok) {

        let message =
            "Could not delete sample.";


        try {

            const errorData =
                await response.json();


            if (
                errorData.detail
            ) {

                message =
                    errorData.detail;

            }

        }
        catch {
            // Ignore JSON parsing errors
        }


        throw new Error(
            message
        );

    }


    return (
        await response.json()
    );

}

// ==========================================================
// DELETE SAMPLE ACTION
// ==========================================================

async function handleDeleteSample(
    sample,
    button
) {

    const typeLabel =
        sample.sample_type === "positive"
            ? "positive"
            : "negative";


    const confirmed =
        window.confirm(
            `Delete this ${typeLabel} example?\n\n`
            + `Sample: ${sample.id}\n\n`
            + `This action cannot be undone.`
        );


    if (!confirmed) {
        return;
    }


    const originalText =
        button.textContent;


    button.disabled =
        true;


    button.textContent =
        "Deleting...";


    try {

        await deleteSample(
            sample.sample_type,
            sample.id
        );


        console.log(
            "Sample deleted:",
            sample.id
        );


        // --------------------------------------------------
        // Any existing training result is now outdated.
        // --------------------------------------------------

        resetTrainingUI();


        // --------------------------------------------------
        // Refresh gesture metadata / counters.
        // --------------------------------------------------

        await refreshCurrentGesture();


        updateCurrentGestureCounters();


        // --------------------------------------------------
        // Reload sample lists.
        // --------------------------------------------------

        const samples =
            await loadSamples();


        // --------------------------------------------------
        // Refresh the currently visible sample list.
        // --------------------------------------------------

        renderSampleList(
            sample.sample_type,
            samples[
                sample.sample_type
            ]
        );

        await loadStoredTraining(); // Reload training data after sample deletion

    }
    catch (error) {

        console.error(
            error
        );


        alert(
            error.message
            ?? "Could not delete sample."
        );


        button.disabled =
            false;


        button.textContent =
            originalText;

    }

}

// ==========================================================
// SAMPLE CARD
// ==========================================================

function createSampleCard(
    sample
) {

    const card =
        document.createElement(
            "article"
        );


    card.className =
        "sample-browser-card";


    const summary =
        sample.feature_summary;


    const trim =
        sample.trim
        ?? {};


    const featureStatus =
        sample.features_extracted
            ? `
                <span
                    class="feature-status feature-status-ok"
                >
                    Features ✓
                </span>
            `
            : `
                <span
                    class="feature-status feature-status-missing"
                >
                    Features ✕
                </span>
            `;


    let featureDetails = `

        <div
            class="sample-feature-empty"
        >
            No feature extraction available.
        </div>
    `;


    if (summary) {

        featureDetails = `

            <div
                class="sample-details-grid"
            >

                <div>
                    <span>
                        Frames
                    </span>

                    <strong>
                        ${escapeHTML(
                            summary.frame_count
                            ?? "—"
                        )}
                    </strong>
                </div>


                <div>
                    <span>
                        FPS
                    </span>

                    <strong>
                        ${escapeHTML(
                            formatFPS(
                                summary.fps
                            )
                        )}
                    </strong>
                </div>


                <div>
                    <span>
                        Landmarks
                    </span>

                    <strong>
                        ${escapeHTML(
                            summary.landmark_count
                            ?? "—"
                        )}
                    </strong>
                </div>


                <div>
                    <span>
                        Distances
                    </span>

                    <strong>
                        ${escapeHTML(
                            summary.distance_count
                            ?? "—"
                        )}
                    </strong>
                </div>


                <div>
                    <span>
                        Valid frames
                    </span>

                    <strong>
                        ${escapeHTML(
                            summary.valid_landmark_frames
                            ?? "—"
                        )}
                    </strong>
                </div>


                <div>
                    <span>
                        Missing frames
                    </span>

                    <strong>
                        ${escapeHTML(
                            summary.missing_landmark_frames
                            ?? "—"
                        )}
                    </strong>
                </div>

            </div>
        `;

    }


    card.innerHTML = `

        <div
            class="sample-browser-card-header"
        >

            <div>

                <h3>
                    ${escapeHTML(
                        shortSampleId(
                            sample.id
                        )
                    )}
                </h3>

                <code>
                    ${escapeHTML(
                        sample.id
                    )}
                </code>

            </div>

            ${featureStatus}

        </div>


        <div
            class="sample-browser-meta"
        >

            <div>

                <span>
                    Clip
                </span>

                <strong>
                    ${escapeHTML(
                        formatDurationMs(
                            trim.duration_ms
                        )
                    )}
                </strong>

            </div>


            <div>

                <span>
                    Start
                </span>

                <strong>
                    ${escapeHTML(
                        trim.start_ms
                        ?? "—"
                    )}
                    ms
                </strong>

            </div>


            <div>

                <span>
                    End
                </span>

                <strong>
                    ${escapeHTML(
                        trim.end_ms
                        ?? "—"
                    )}
                    ms
                </strong>

            </div>

        </div>


        ${featureDetails}


        <div
            class="sample-browser-footer"
        >

            <span>
                Created:
                ${escapeHTML(
                    formatDate(
                        sample.created_at
                    )
                )}
            </span>


            ${
                summary
                    ? `
                        <span>
                            ${escapeHTML(
                                summary.landmarker
                                ?? ""
                            )}
                        </span>
                    `
                    : ""
            }

        </div>


        <div
            class="sample-browser-actions"
        >

            <button
                type="button"
                class="view-sample-button primary"
            >
                View sample
            </button>


            <button
                type="button"
                class="delete-sample-button danger"
            >
                Delete
            </button>

        </div>
    `;


    card
        .querySelector(
            ".view-sample-button"
        )
        .addEventListener(
            "click",
            () => {

                openSampleViewer(
                    sample.sample_type,
                    sample.id
                );

            }
        );

    const deleteButton =
        card.querySelector(
            ".delete-sample-button"
        );


    deleteButton.addEventListener(
        "click",
        () => {

            handleDeleteSample(
                sample,
                deleteButton
            );

        }
    );

    return card;

}


// ==========================================================
// SAMPLE DETAILS
// ==========================================================

function renderSampleViewerDetails(
    sample
) {

    const details =
        document.getElementById(
            "sample-viewer-details"
        );


    if (!details) {
        return;
    }


    const summary =
        sample.feature_summary;


    const trim =
        sample.trim
        ?? {};


    details.innerHTML = `

        <div
            class="sample-browser-meta"
        >

            <div>
                <span>
                    Type
                </span>

                <strong>
                    ${escapeHTML(
                        sample.sample_type
                        ?? "—"
                    )}
                </strong>
            </div>


            <div>
                <span>
                    Duration
                </span>

                <strong>
                    ${escapeHTML(
                        formatDurationMs(
                            trim.duration_ms
                        )
                    )}
                </strong>
            </div>


            <div>
                <span>
                    Target
                </span>

                <strong>
                    ${escapeHTML(
                        sample.target
                        ?? "—"
                    )}
                </strong>
            </div>

        </div>


        ${
            summary
                ? `

                    <div
                        class="sample-details-grid"
                    >

                        <div>
                            <span>
                                Frames
                            </span>

                            <strong>
                                ${escapeHTML(
                                    summary.frame_count
                                    ?? "—"
                                )}
                            </strong>
                        </div>


                        <div>
                            <span>
                                FPS
                            </span>

                            <strong>
                                ${escapeHTML(
                                    formatFPS(
                                        summary.fps
                                    )
                                )}
                            </strong>
                        </div>


                        <div>
                            <span>
                                Landmarks
                            </span>

                            <strong>
                                ${escapeHTML(
                                    summary.landmark_count
                                    ?? "—"
                                )}
                            </strong>
                        </div>


                        <div>
                            <span>
                                Distances
                            </span>

                            <strong>
                                ${escapeHTML(
                                    summary.distance_count
                                    ?? "—"
                                )}
                            </strong>
                        </div>


                        <div>
                            <span>
                                Valid frames
                            </span>

                            <strong>
                                ${escapeHTML(
                                    summary.valid_landmark_frames
                                    ?? "—"
                                )}
                            </strong>
                        </div>


                        <div>
                            <span>
                                Missing frames
                            </span>

                            <strong>
                                ${escapeHTML(
                                    summary.missing_landmark_frames
                                    ?? "—"
                                )}
                            </strong>
                        </div>

                    </div>

                `
                : `

                    <div
                        class="sample-feature-empty"
                    >
                        Feature extraction is not available.
                    </div>

                `
        }
    `;

}


// ==========================================================
// LOAD LANDMARK DATA
// ==========================================================

async function loadSampleLandmarks(
    landmarksURL
) {

    const response =
        await fetch(
            landmarksURL
        );


    if (!response.ok) {

        throw new Error(
            "Could not load landmark data."
        );

    }


    return (
        await response.json()
    );

}


// ==========================================================
// LANDMARK CANVAS
// ==========================================================

function getLandmarkCanvas() {

    return (
        document.getElementById(
            "sample-landmark-canvas"
        )
    );

}


function clearLandmarkCanvas() {

    const canvas =
        getLandmarkCanvas();


    if (!canvas) {
        return;
    }


    const context =
        canvas.getContext(
            "2d"
        );


    context.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

}


function resizeLandmarkCanvas() {

    const canvas =
        getLandmarkCanvas();


    const stage =
        document.getElementById(
            "sample-landmark-stage"
        );


    if (
        !canvas
        ||
        !stage
    ) {

        return;

    }


    const width =
        stage.clientWidth;


    const height =
        stage.clientHeight;


    if (
        width <= 0
        ||
        height <= 0
    ) {

        return;

    }


    const pixelRatio =
        window.devicePixelRatio
        || 1;


    const pixelWidth =
        Math.round(
            width
            * pixelRatio
        );


    const pixelHeight =
        Math.round(
            height
            * pixelRatio
        );


    /*
     * Só redimensionar se necessário.
     * Evita limpar/recriar o canvas a cada frame.
     */

    if (
        canvas.width
        !== pixelWidth
    ) {

        canvas.width =
            pixelWidth;

    }


    if (
        canvas.height
        !== pixelHeight
    ) {

        canvas.height =
            pixelHeight;

    }


    canvas.style.width =
        `${width}px`;


    canvas.style.height =
        `${height}px`;

}


// ==========================================================
// VIDEO CONTENT RECT
// ==========================================================

function getDisplayedVideoRect(
    video
) {

    const containerWidth =
        video.clientWidth;


    const containerHeight =
        video.clientHeight;


    const sourceWidth =
        video.videoWidth;


    const sourceHeight =
        video.videoHeight;


    if (
        !containerWidth
        ||
        !containerHeight
        ||
        !sourceWidth
        ||
        !sourceHeight
    ) {

        return null;

    }


    /*
     * O vídeo usa object-fit: contain.
     *
     * Portanto pode haver barras horizontais
     * ou verticais. Temos de desenhar as landmarks
     * apenas na região real da imagem.
     */

    const scale =
        Math.min(
            containerWidth
            / sourceWidth,

            containerHeight
            / sourceHeight
        );


    const width =
        sourceWidth
        * scale;


    const height =
        sourceHeight
        * scale;


    return {

        x:
            (
                containerWidth
                - width
            )
            / 2,

        y:
            (
                containerHeight
                - height
            )
            / 2,

        width,

        height,

    };

}


// ==========================================================
// LANDMARK FRAME LOOKUP
// ==========================================================

function findNearestLandmarkFrame(
    timestamp
) {

    const timestamps =
        state.landmarkData
            ?.timestamps;


    if (
        !timestamps
        ||
        timestamps.length === 0
    ) {

        return -1;

    }


    if (
        timestamp <= timestamps[0]
    ) {

        return 0;

    }


    const lastIndex =
        timestamps.length
        - 1;


    if (
        timestamp
        >= timestamps[
            lastIndex
        ]
    ) {

        return lastIndex;

    }


    /*
     * Binary search.
     */

    let low = 0;

    let high =
        lastIndex;


    while (
        low <= high
    ) {

        const middle =
            Math.floor(
                (
                    low
                    + high
                )
                / 2
            );


        const value =
            timestamps[
                middle
            ];


        if (
            value === timestamp
        ) {

            return middle;

        }


        if (
            value < timestamp
        ) {

            low =
                middle + 1;

        }
        else {

            high =
                middle - 1;

        }

    }


    const previousIndex =
        Math.max(
            0,
            high
        );


    const nextIndex =
        Math.min(
            lastIndex,
            low
        );


    const previousDifference =
        Math.abs(
            timestamp
            - timestamps[
                previousIndex
            ]
        );


    const nextDifference =
        Math.abs(
            timestamps[
                nextIndex
            ]
            - timestamp
        );


    return (
        previousDifference
        <= nextDifference
            ? previousIndex
            : nextIndex
    );

}


// ==========================================================
// LANDMARK VALIDATION
// ==========================================================

function isValidLandmark(
    landmark
) {

    if (
        !Array.isArray(
            landmark
        )
        ||
        landmark.length < 2
    ) {

        return false;

    }


    return (

        Number.isFinite(
            landmark[0]
        )

        &&

        Number.isFinite(
            landmark[1]
        )

    );

}


// ==========================================================
// LANDMARK -> SCREEN POSITION
// ==========================================================

function landmarkToScreen(
    landmark,
    videoRect
) {

    return {

        x:
            videoRect.x
            + landmark[0]
            * videoRect.width,

        y:
            videoRect.y
            + landmark[1]
            * videoRect.height,

    };

}


// ==========================================================
// DRAW FRAME
// ==========================================================

function drawLandmarkFrame(
    landmarks
) {

    const canvas =
        getLandmarkCanvas();


    const video =
        document.getElementById(
            "sample-viewer-video"
        );


    if (
        !canvas
        ||
        !video
    ) {

        return;

    }


    resizeLandmarkCanvas();


    const context =
        canvas.getContext(
            "2d"
        );


    const pixelRatio =
        window.devicePixelRatio
        || 1;


    context.setTransform(
        pixelRatio,
        0,
        0,
        pixelRatio,
        0,
        0
    );


    context.clearRect(
        0,
        0,
        video.clientWidth,
        video.clientHeight
    );


    if (
        !state.showLandmarks
        ||
        !Array.isArray(
            landmarks
        )
    ) {

        return;

    }


    const videoRect = {

        x: 0,

        y: 0,

        width:
            canvas.clientWidth,

        height:
            canvas.clientHeight,

    };


    const target =
        state.landmarkData
            ?.target;


    const connections =
        target === "body"
            ? POSE_CONNECTIONS
            : HAND_CONNECTIONS;


    // ------------------------------------------------------
    // Connections
    // ------------------------------------------------------

    context.lineWidth =
        target === "body"
            ? 2
            : 2.5;


    context.lineCap =
        "round";


    context.lineJoin =
        "round";


    context.strokeStyle =
        "rgba(0, 229, 255, 0.9)";


    for (
        const connection
        of connections
    ) {

        const [
            startIndex,
            endIndex,
        ] = connection;


        const startLandmark =
            landmarks[
                startIndex
            ];


        const endLandmark =
            landmarks[
                endIndex
            ];


        if (
            !isValidLandmark(
                startLandmark
            )
            ||
            !isValidLandmark(
                endLandmark
            )
        ) {

            continue;

        }


        const start =
            landmarkToScreen(
                startLandmark,
                videoRect
            );


        const end =
            landmarkToScreen(
                endLandmark,
                videoRect
            );


        context.beginPath();

        context.moveTo(
            start.x,
            start.y
        );

        context.lineTo(
            end.x,
            end.y
        );

        context.stroke();

    }


    // ------------------------------------------------------
    // Landmark points
    // ------------------------------------------------------

    const radius =
        target === "body"
            ? 3.5
            : 4.5;


    for (
        const landmark
        of landmarks
    ) {

        if (
            !isValidLandmark(
                landmark
            )
        ) {

            continue;

        }


        const point =
            landmarkToScreen(
                landmark,
                videoRect
            );


        context.beginPath();


        context.arc(
            point.x,
            point.y,
            radius,
            0,
            Math.PI * 2
        );


        context.fillStyle =
            "rgba(255, 255, 255, 0.95)";


        context.fill();


        context.lineWidth =
            2;


        context.strokeStyle =
            "rgba(0, 229, 255, 1)";


        context.stroke();

    }

}


// ==========================================================
// DRAW CURRENT LANDMARKS
// ==========================================================

function drawCurrentLandmarks() {

    const video =
        document.getElementById(
            "sample-viewer-video"
        );


    if (
        !video
        ||
        !state.landmarkData
    ) {

        clearLandmarkCanvas();

        return;

    }


    if (
        !state.showLandmarks
    ) {

        clearLandmarkCanvas();

        return;

    }


    const trim =
        getSelectedSampleTrim();


    if (
        video.currentTime
        < trim.start
        ||
        video.currentTime
        > trim.end
    ) {

        clearLandmarkCanvas();

        return;

    }


    /*
     * Os timestamps das features começam em 0
     * no início do clip.
     */

    const featureTime =
        video.currentTime
        - trim.start;


    const frameIndex =
        findNearestLandmarkFrame(
            featureTime
        );


    if (
        frameIndex < 0
    ) {

        clearLandmarkCanvas();

        return;

    }


    const landmarks =
        state.landmarkData
            .landmarks[
                frameIndex
            ];


    drawLandmarkFrame(
        landmarks
    );

}


// ==========================================================
// LANDMARK ANIMATION LOOP
// ==========================================================

function startLandmarkAnimation() {

    stopLandmarkAnimation();


    const update = () => {

        drawCurrentLandmarks();


        const video =
            document.getElementById(
                "sample-viewer-video"
            );


        if (
            video
            &&
            !video.paused
            &&
            !video.ended
        ) {

            state.landmarkAnimationFrame =
                requestAnimationFrame(
                    update
                );

        }
        else {

            state.landmarkAnimationFrame =
                null;

        }

    };


    state.landmarkAnimationFrame =
        requestAnimationFrame(
            update
        );

}


function stopLandmarkAnimation() {

    if (
        state.landmarkAnimationFrame
        !== null
    ) {

        cancelAnimationFrame(
            state.landmarkAnimationFrame
        );


        state.landmarkAnimationFrame =
            null;

    }

}


// ==========================================================
// RESIZE
// ==========================================================

window.addEventListener(
    "resize",
    () => {

        resizeLandmarkCanvas();

        drawCurrentLandmarks();

    }
);


// ==========================================================
// OPEN SAMPLE VIEWER
// ==========================================================

async function openSampleViewer(
    sampleType,
    sampleId
) {

    const modal =
        ensureSampleViewerModal();


    const loading =
        modal.querySelector(
            "#sample-viewer-loading"
        );


    const content =
        modal.querySelector(
            "#sample-viewer-content"
        );


    const title =
        modal.querySelector(
            "#sample-viewer-title"
        );


    const subtitle =
        modal.querySelector(
            "#sample-viewer-subtitle"
        );


    const range =
        modal.querySelector(
            "#sample-viewer-range"
        );


    const video =
        modal.querySelector(
            "#sample-viewer-video"
        );


    modal.classList.remove(
        "hidden"
    );


    loading.classList.remove(
        "hidden"
    );


    content.classList.add(
        "hidden"
    );


    try {

        const sample =
            await loadSample(
                sampleType,
                sampleId
            );


        state.selectedSample =
            sample;

        
        state.landmarkData =
            null;


        const landmarkToggle =
            modal.querySelector(
                "#show-sample-landmarks"
            );


        state.showLandmarks =
            true;


        landmarkToggle.checked =
            true;


        if (
            sample.features_extracted
            &&
            sample.landmarks_url
        ) {

            try {

                state.landmarkData =
                    await loadSampleLandmarks(
                        sample.landmarks_url
                    );


                console.log(
                    "Landmarks loaded:",
                    state.landmarkData
                );

            }
            catch (error) {

                console.error(
                    "Could not load landmarks:",
                    error
                );

            }

        }

        title.textContent =
            `Sample ${shortSampleId(
                sample.id
            )}`;


        subtitle.textContent =
            `${sample.gesture_name} · `
            + `${sample.sample_type}`;


        const trim =
            sample.trim
            ?? {};


        range.textContent =
            `Clip: `
            + `${formatDurationMs(
                trim.start_ms
            )}`
            + ` → `
            + `${formatDurationMs(
                trim.end_ms
            )}`
            + ` · `
            + `${formatDurationMs(
                trim.duration_ms
            )}`;


        video.src =
            sample.video_url;


        renderSampleViewerDetails(
            sample
        );


        loading.classList.add(
            "hidden"
        );


        content.classList.remove(
            "hidden"
        );

        requestAnimationFrame(
            () => {

                resizeLandmarkCanvas();

                drawCurrentLandmarks();

            }
        );

    }
    catch (error) {

        console.error(
            error
        );


        loading.innerHTML = `
            Could not load sample.
        `;

    }

}


// ==========================================================
// RENDER SAMPLE LIST
// ==========================================================

function renderSampleList(
    sampleType,
    samples
) {

    const content =
        document.getElementById(
            "samples-modal-content"
        );


    if (!content) {
        return;
    }


    content.innerHTML = "";


    if (
        samples.length === 0
    ) {

        content.innerHTML = `

            <div
                class="empty-state"
            >
                No ${sampleType} examples recorded yet.
            </div>
        `;

        return;

    }


    const list =
        document.createElement(
            "div"
        );


    list.className =
        "sample-browser-list";


    for (
        const sample
        of samples
    ) {

        list.appendChild(
            createSampleCard(
                sample
            )
        );

    }


    content.appendChild(
        list
    );

}


// ==========================================================
// OPEN SAMPLE LIST
// ==========================================================

async function openSamplesModal(
    sampleType
) {

    if (!state.gesture) {
        return;
    }


    const modal =
        ensureSamplesModal();


    const title =
        modal.querySelector(
            "#samples-modal-title"
        );


    const subtitle =
        modal.querySelector(
            "#samples-modal-subtitle"
        );


    const content =
        modal.querySelector(
            "#samples-modal-content"
        );


    const typeLabel =
        sampleType === "positive"
            ? "Positive"
            : "Negative";


    title.textContent =
        `${typeLabel} examples`;


    subtitle.textContent =
        state.gesture.name;


    content.innerHTML = `

        <div
            class="empty-state"
        >
            Loading examples...
        </div>
    `;


    modal.classList.remove(
        "hidden"
    );


    try {

        const samples =
            await loadSamples();


        renderSampleList(
            sampleType,
            samples[
                sampleType
            ]
        );

    }
    catch (error) {

        console.error(
            error
        );


        content.innerHTML = `

            <div
                class="empty-state"
            >
                Could not load examples.
            </div>
        `;

    }

}


// ==========================================================
// SAMPLE COUNTERS
// ==========================================================

function setupSampleCounters() {

    const positiveCount =
        document.getElementById(
            "positive-count"
        );


    const negativeCount =
        document.getElementById(
            "negative-count"
        );


    const configureCounter = (
        element,
        sampleType
    ) => {

        if (!element) {
            return;
        }


        element.classList.add(
            "sample-count-link"
        );


        element.setAttribute(
            "role",
            "button"
        );


        element.setAttribute(
            "tabindex",
            "0"
        );


        element.setAttribute(
            "title",
            `View ${sampleType} examples`
        );


        element.addEventListener(
            "click",
            event => {

                event.preventDefault();

                event.stopPropagation();


                openSamplesModal(
                    sampleType
                );

            }
        );


        element.addEventListener(
            "keydown",
            event => {

                if (
                    event.key !== "Enter"
                    &&
                    event.key !== " "
                ) {

                    return;

                }


                event.preventDefault();

                event.stopPropagation();


                openSamplesModal(
                    sampleType
                );

            }
        );

    };


    configureCounter(
        positiveCount,
        "positive"
    );


    configureCounter(
        negativeCount,
        "negative"
    );

}


// ==========================================================
// LOAD EXISTING GESTURES
// ==========================================================

async function loadGestures() {

    const gestureList =
        document.getElementById(
            "gesture-list"
        );


    gestureList.innerHTML = `
        <div class="empty-state">
            Loading gestures...
        </div>
    `;


    try {

        const response =
            await fetch(
                "/api/gestures"
            );


        if (!response.ok) {

            throw new Error(
                "Could not load gestures."
            );

        }


        const gestures =
            await response.json();


        gestureList.innerHTML = "";


        if (
            gestures.length === 0
        ) {

            gestureList.innerHTML = `
                <div class="empty-state">
                    No gestures created yet.
                </div>
            `;

            return;

        }


        for (
            const gesture
            of gestures
        ) {

            const card =
                document.createElement(
                    "div"
                );


            card.className =
                "gesture-card";


            const targetLabel =
                gesture.target === "hand"
                    ? "Hand"
                    : "Body";


            const positiveCount =
                gesture.samples
                    ?.positive
                ?? 0;


            const negativeCount =
                gesture.samples
                    ?.negative
                ?? 0;


            card.innerHTML = `

                <div
                    class="gesture-card-main"
                >

                    <h3>
                        ${escapeHTML(
                            gesture.name
                        )}
                    </h3>

                    <span
                        class="gesture-target"
                    >
                        ${escapeHTML(
                            targetLabel
                        )}
                    </span>

                </div>


                <div
                    class="gesture-counts"
                >

                    <span>
                        Positive:
                        <strong>
                            ${positiveCount}
                        </strong>
                    </span>

                    <span>
                        Negative:
                        <strong>
                            ${negativeCount}
                        </strong>
                    </span>

                </div>


                <button
                    class="open-gesture"
                >
                    Open →
                </button>
            `;


            card
                .querySelector(
                    ".open-gesture"
                )
                .addEventListener(
                    "click",
                    () => {

                        state.gesture =
                            gesture;

                        openGesture();

                    }
                );


            gestureList.appendChild(
                card
            );

        }

    }
    catch (error) {

        console.error(
            error
        );


        gestureList.innerHTML = `
            <div class="empty-state">
                Could not load gestures.
            </div>
        `;

    }

}


// ==========================================================
// CREATE GESTURE
// ==========================================================

document
    .getElementById(
        "create-gesture"
    )
    .addEventListener(
        "click",
        async () => {

            const name =
                document
                    .getElementById(
                        "gesture-name"
                    )
                    .value
                    .trim();


            const target =
                document.querySelector(
                    'input[name="target"]:checked'
                ).value;


            if (!name) {

                alert(
                    "Please enter a gesture name."
                );

                return;

            }


            try {

                const response =
                    await fetch(
                        "/api/gestures",
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json",
                            },

                            body: JSON.stringify({
                                name,
                                target,
                            }),
                        }
                    );


                if (!response.ok) {

                    let message =
                        "Could not create gesture.";


                    try {

                        const errorData =
                            await response.json();


                        if (
                            errorData.detail
                        ) {

                            message =
                                errorData.detail;

                        }

                    }
                    catch {
                        // Ignore JSON parsing errors
                    }


                    alert(
                        message
                    );

                    return;

                }


                state.gesture =
                    await response.json();


                document
                    .getElementById(
                        "gesture-name"
                    )
                    .value = "";


                openGesture();

            }
            catch (error) {

                console.error(
                    error
                );


                alert(
                    "Could not communicate with the server."
                );

            }

        }
    );


// ==========================================================
// UPDATE CURRENT GESTURE COUNTERS
// ==========================================================

function updateCurrentGestureCounters() {

    if (!state.gesture) {
        return;
    }


    document
        .getElementById(
            "positive-count"
        )
        .textContent =
            state.gesture
                .samples
                ?.positive
            ?? 0;


    document
        .getElementById(
            "negative-count"
        )
        .textContent =
            state.gesture
                .samples
                ?.negative
            ?? 0;

}

// ==========================================================
// OPEN GESTURE
// ==========================================================

function openGesture() {

    if (!state.gesture) {
        return;
    }


    resetTrainingUI();


    document
        .getElementById(
            "current-gesture-name"
        )
        .textContent =
            state.gesture.name;


    document
        .getElementById(
            "current-gesture-target"
        )
        .textContent =
            state.gesture.target === "hand"
                ? "Hand gesture"
                : "Body gesture";


    updateCurrentGestureCounters();


    showScreen(
        "samples"
    );

    loadStoredTraining();

}


// ==========================================================
// REFRESH CURRENT GESTURE
// ==========================================================

async function refreshCurrentGesture() {

    if (!state.gesture) {
        return;
    }


    const response =
        await fetch(
            `/api/gestures/${state.gesture.id}`
        );


    if (!response.ok) {

        console.error(
            "Could not refresh gesture."
        );

        return;

    }


    state.gesture =
        await response.json();

}

// ==========================================================
// MODEL TRAINING
// ==========================================================

function resetTrainingUI() {

    state.trainingResult =
        null;

    state.trainingDatasetStatus =
        null;

    const resetElements = [
        trainingTotalSamples,
        trainingTrainSamples,
        trainingValidationSamples,

        trainingTrainPositive,
        trainingTrainNegative,

        trainingValidationPositive,
        trainingValidationNegative,

        trainingValidationAccuracy,
        trainingValidationPrecision,
        trainingValidationRecall,
        trainingValidationF1,

        trainingCmTn,
        trainingCmFp,
        trainingCmFn,
        trainingCmTp,
    ];


    for (
        const element
        of resetElements
    ) {

        if (element) {

            element.textContent =
                "—";

        }

    }


    if (
        trainingValidationSamplesBody
    ) {

        trainingValidationSamplesBody.innerHTML =
            "";

    }

    state.trainingInProgress =
        false;


    if (trainModelButton) {

        trainModelButton.disabled =
            false;


        trainModelButton.textContent =
            "Train Decision Tree";

    }


    if (trainingStatus) {

        trainingStatus.textContent =
            "";

    }


    if (trainingResults) {

        trainingResults.classList.add(
            "hidden"
        );

    }


    if (trainingDecisionRules) {

        trainingDecisionRules.textContent =
            "";

    }


    if (trainingFeaturesBody) {

        trainingFeaturesBody.innerHTML =
            "";

    }


    if (trainingSamplesBody) {

        trainingSamplesBody.innerHTML =
            "";

    }

}


// ==========================================================
// TRAINING LABEL
// ==========================================================

function trainingLabel(
    label
) {

    return (
        Number(label) === 1
            ? "Positive"
            : "Negative"
    );

}


// ==========================================================
// TRAINING CONFIDENCE
// ==========================================================

function getTrainingConfidence(
    sample
) {

    if (
        Number(
            sample.predicted_label
        ) === 1
    ) {

        return Number(
            sample.probability_positive
        );

    }


    return Number(
        sample.probability_negative
    );

}


// ==========================================================
// RENDER TRAINING FEATURES
// ==========================================================

function renderTrainingFeatures(
    features
) {

    if (!trainingFeaturesBody) {
        return;
    }


    trainingFeaturesBody.innerHTML =
        "";


    if (
        !Array.isArray(
            features
        )
        ||
        features.length === 0
    ) {

        trainingFeaturesBody.innerHTML = `

            <tr>

                <td colspan="2">
                    No features were used by the tree.
                </td>

            </tr>
        `;

        return;

    }


    for (
        const feature
        of features
    ) {

        const row =
            document.createElement(
                "tr"
            );


        const importance =
            Number(
                feature.importance
            );


        row.innerHTML = `

            <td>
                <code>
                    ${escapeHTML(
                        feature.name
                    )}
                </code>
            </td>

            <td>
                ${
                    Number.isFinite(
                        importance
                    )
                        ? (
                            importance
                            * 100
                        ).toFixed(2)
                        + "%"
                        : "—"
                }
            </td>
        `;


        trainingFeaturesBody.appendChild(
            row
        );

    }

}


/// ==========================================================
// VALIDATION SAMPLE PREDICTIONS
// ==========================================================

function renderValidationSamples(
    samples
) {

    if (
        !trainingValidationSamplesBody
    ) {

        return;

    }


    trainingValidationSamplesBody.innerHTML =
        "";


    if (
        !Array.isArray(
            samples
        )
        || samples.length === 0
    ) {

        const row =
            document.createElement(
                "tr"
            );


        const cell =
            document.createElement(
                "td"
            );


        cell.colSpan = 5;

        cell.textContent =
            "No validation samples available.";


        row.appendChild(
            cell
        );


        trainingValidationSamplesBody.appendChild(
            row
        );


        return;

    }


    for (
        const sample
        of samples
    ) {

        const row =
            document.createElement(
                "tr"
            );


        // --------------------------------------------------
        // SAMPLE
        // --------------------------------------------------

        const sampleCell =
            document.createElement(
                "td"
            );


        sampleCell.textContent =
            shortSampleId(
                sample.sample_id
            );


        sampleCell.title =
            sample.sample_id
            ?? "";


        // --------------------------------------------------
        // ACTUAL
        // --------------------------------------------------

        const actualCell =
            document.createElement(
                "td"
            );


        actualCell.textContent =
            trainingLabel(
                sample.true_label
            );


        // --------------------------------------------------
        // PREDICTED
        // --------------------------------------------------

        const predictedCell =
            document.createElement(
                "td"
            );


        predictedCell.textContent =
            trainingLabel(
                sample.predicted_label
            );


        // --------------------------------------------------
        // CONFIDENCE
        // --------------------------------------------------

        const confidenceCell =
            document.createElement(
                "td"
            );


        const confidence =
            getTrainingConfidence(
                sample
            );


        confidenceCell.textContent =
            confidence === null
                ? "—"
                : formatTrainingMetric(
                    confidence
                );


        // --------------------------------------------------
        // RESULT
        // --------------------------------------------------

        const resultCell =
            document.createElement(
                "td"
            );


        resultCell.textContent =
            sample.correct
                ? "Correct"
                : "Incorrect";


        resultCell.classList.add(
            sample.correct
                ? "prediction-correct"
                : "prediction-incorrect"
        );


        // --------------------------------------------------
        // ROW
        // --------------------------------------------------

        row.appendChild(
            sampleCell
        );

        row.appendChild(
            actualCell
        );

        row.appendChild(
            predictedCell
        );

        row.appendChild(
            confidenceCell
        );

        row.appendChild(
            resultCell
        );


        trainingValidationSamplesBody.appendChild(
            row
        );

    }

}

// ==========================================================
// RENDER TRAINING RESULT
// ==========================================================

function renderTrainingResult(
    data
) {

    if (!data) {
        return;
    }


    state.trainingResult =
        data;


    // ======================================================
    // DATA
    // ======================================================

    const dataset =
        data.dataset
        ?? {};


    const split =
        dataset.split
        ?? {};


    const evaluation =
        data.evaluation
        ?? {};


    const confusionMatrix =
        evaluation.confusion_matrix
        ?? {};


    const model =
        data.model
        ?? {};


    const decisionTree =
        data.decision_tree
        ?? {};


    const validationSamples =
        data.validation_samples
        ?? [];


    // ======================================================
    // SHOW RESULTS
    // ======================================================

    if (
        trainingResults
    ) {

        trainingResults.classList.remove(
            "hidden"
        );

    }


    // ======================================================
    // OLD TOP SUMMARY
    // ======================================================

    /*
     * Keep the original general summary if it is still
     * present in the HTML.
     */

    if (
        trainingSampleCount
    ) {

        trainingSampleCount.textContent =
            dataset.samples
            ?? "—";

    }


    if (
        trainingPositiveCount
    ) {

        trainingPositiveCount.textContent =
            dataset.positive
            ?? "—";

    }


    if (
        trainingNegativeCount
    ) {

        trainingNegativeCount.textContent =
            dataset.negative
            ?? "—";

    }


    if (
        trainingFeatureCount
    ) {

        trainingFeatureCount.textContent =
            dataset.features
            ?? "—";

    }


    if (
        trainingTreeDepth
    ) {

        trainingTreeDepth.textContent =
            model.tree_depth
            ?? "—";

    }


    if (
        trainingLeafCount
    ) {

        trainingLeafCount.textContent =
            model.leaf_count
            ?? "—";

    }


    if (
        trainingUsedFeatureCount
    ) {

        trainingUsedFeatureCount.textContent =
            decisionTree.used_feature_count
            ?? 0;

    }


    /*
     * If the old summary still contains the old
     * training-accuracy element, show validation accuracy
     * instead.
     */

    if (
        trainingAccuracy
    ) {

        trainingAccuracy.textContent =
            formatTrainingMetric(
                evaluation.accuracy
            );

    }


    // ======================================================
    // DATASET SPLIT
    // ======================================================

    if (
        trainingTotalSamples
    ) {

        trainingTotalSamples.textContent =
            dataset.samples
            ?? "—";

    }


    if (
        trainingTrainSamples
    ) {

        trainingTrainSamples.textContent =
            split.training_samples
            ?? "—";

    }


    if (
        trainingValidationSamples
    ) {

        trainingValidationSamples.textContent =
            split.validation_samples
            ?? "—";

    }


    if (
        trainingTrainPositive
    ) {

        trainingTrainPositive.textContent =
            split.training_positive
            ?? "—";

    }


    if (
        trainingTrainNegative
    ) {

        trainingTrainNegative.textContent =
            split.training_negative
            ?? "—";

    }


    if (
        trainingValidationPositive
    ) {

        trainingValidationPositive.textContent =
            split.validation_positive
            ?? "—";

    }


    if (
        trainingValidationNegative
    ) {

        trainingValidationNegative.textContent =
            split.validation_negative
            ?? "—";

    }


    // ======================================================
    // VALIDATION METRICS
    // ======================================================

    if (
        trainingValidationAccuracy
    ) {

        trainingValidationAccuracy.textContent =
            formatTrainingMetric(
                evaluation.accuracy
            );

    }


    if (
        trainingValidationPrecision
    ) {

        trainingValidationPrecision.textContent =
            formatTrainingMetric(
                evaluation.precision
            );

    }


    if (
        trainingValidationRecall
    ) {

        trainingValidationRecall.textContent =
            formatTrainingMetric(
                evaluation.recall
            );

    }


    if (
        trainingValidationF1
    ) {

        trainingValidationF1.textContent =
            formatTrainingMetric(
                evaluation.f1
            );

    }


    // ======================================================
    // CONFUSION MATRIX
    // ======================================================

    if (
        trainingCmTn
    ) {

        trainingCmTn.textContent =
            confusionMatrix.tn
            ?? "—";

    }


    if (
        trainingCmFp
    ) {

        trainingCmFp.textContent =
            confusionMatrix.fp
            ?? "—";

    }


    if (
        trainingCmFn
    ) {

        trainingCmFn.textContent =
            confusionMatrix.fn
            ?? "—";

    }


    if (
        trainingCmTp
    ) {

        trainingCmTp.textContent =
            confusionMatrix.tp
            ?? "—";

    }


    // ======================================================
    // WARNING / EVALUATION DESCRIPTION
    // ======================================================

    if (
        trainingWarning
    ) {

        trainingWarning.textContent =
            evaluation.warning
            ?? "";

    }


    // ======================================================
    // DECISION RULES
    // ======================================================

    if (
        trainingDecisionRules
    ) {

        trainingDecisionRules.textContent =
            decisionTree.rules
            ?? "No decision rules available.";

    }


    // ======================================================
    // USED FEATURES
    // ======================================================

    renderTrainingFeatures(
        decisionTree.used_features
        ?? []
    );


    // ======================================================
    // VALIDATION PREDICTIONS
    // ======================================================

    renderValidationSamples(
        validationSamples
    );

}


// ==========================================================
// LOAD STORED TRAINING
// ==========================================================

async function loadStoredTraining() {

    if (!state.gesture) {
        return;
    }


    const gestureId =
        state.gesture.id;


    if (trainingStatus) {

        trainingStatus.textContent =
            "Loading last training...";

    }


    try {

        const response =
            await fetch(
                `/api/gestures/`
                + `${gestureId}`
                + `/training`
            );


        if (!response.ok) {

            throw new Error(
                "Could not load stored training."
            );

        }


        const data =
            await response.json();


        /*
         * The user may have opened another gesture while
         * this request was running.
         */

        if (
            !state.gesture
            ||
            state.gesture.id
                !== gestureId
        ) {

            return;

        }


        // --------------------------------------------------
        // No model has ever been trained
        // --------------------------------------------------

        if (
            data.status
            === "no_training"
        ) {

            state.trainingResult =
                null;


            state.trainingDatasetStatus =
                null;


            if (trainingResults) {

                trainingResults.classList.add(
                    "hidden"
                );

            }


            if (trainingStatus) {

                trainingStatus.textContent =
                    "No model trained yet.";

            }


            return;

        }


        // --------------------------------------------------
        // Stored model exists
        // --------------------------------------------------

        if (
            data.status
            !== "trained"
            ||
            !data.training
        ) {

            throw new Error(
                "Stored training response is invalid."
            );

        }


        state.trainingDatasetStatus =
            data.dataset_status
            ?? null;


        renderTrainingResult(
            data.training
        );


        renderStoredTrainingStatus(
            data.training,
            data.dataset_status
        );

    }
    catch (error) {

        console.error(
            "Could not load stored training:",
            error
        );


        if (trainingStatus) {

            trainingStatus.textContent =
                "Could not load last training.";

        }

    }

}


// ==========================================================
// STORED TRAINING STATUS
// ==========================================================

function renderStoredTrainingStatus(
    trainingData,
    datasetStatus
) {

    const trainedAt =
        trainingData
            ?.trained_at;


    const formattedDate =
        trainedAt
            ? formatDate(
                trainedAt
            )
            : null;


    if (trainingStatus) {

        trainingStatus.textContent =
            formattedDate
                ? `Last trained: ${formattedDate}`
                : "Stored model loaded.";

    }


    if (!trainingWarning) {
        return;
    }


    const originalWarning =
        trainingData
            ?.evaluation
            ?.warning
        ?? "";


    if (
        !datasetStatus
    ) {

        trainingWarning.textContent =
            originalWarning;

        return;

    }


    // ------------------------------------------------------
    // DATASET IS CURRENT
    // ------------------------------------------------------

    if (
        datasetStatus.status
        === "current"
    ) {

        trainingWarning.textContent =
            originalWarning;

        return;

    }


    // ------------------------------------------------------
    // DATASET CHANGED
    // ------------------------------------------------------

    if (
        datasetStatus.status
        === "outdated"
    ) {

        const storedSamples =
            datasetStatus
                .stored_samples
            ?? "—";


        const currentSamples =
            datasetStatus
                .current_samples
            ?? "—";


        const storedPositive =
            datasetStatus
                .stored_positive
            ?? "—";


        const currentPositive =
            datasetStatus
                .current_positive
            ?? "—";


        const storedNegative =
            datasetStatus
                .stored_negative
            ?? "—";


        const currentNegative =
            datasetStatus
                .current_negative
            ?? "—";


        trainingWarning.textContent =
            "The dataset has changed since this model "
            + "was trained. Retraining is recommended. "
            + `Training dataset: ${storedSamples} samples `
            + `(${storedPositive} positive, `
            + `${storedNegative} negative). `
            + `Current dataset: ${currentSamples} samples `
            + `(${currentPositive} positive, `
            + `${currentNegative} negative).`;

        return;

    }


    // ------------------------------------------------------
    // DATASET COULD NOT BE CHECKED
    // ------------------------------------------------------

    if (
        datasetStatus.status
        === "unknown"
    ) {

        trainingWarning.textContent =
            "The stored model was loaded, but the current "
            + "dataset could not be compared with the "
            + "training dataset.";

        return;

    }


    trainingWarning.textContent =
        originalWarning;

}

// ==========================================================
// TRAIN CURRENT GESTURE
// ==========================================================

async function trainCurrentGesture() {

    if (!state.gesture) {

        alert(
            "No gesture selected."
        );

        return;

    }


    if (
        state.trainingInProgress
    ) {

        return;

    }


    state.trainingInProgress =
        true;


    if (trainModelButton) {

        trainModelButton.disabled =
            true;


        trainModelButton.textContent =
            "Training...";

    }


    if (trainingStatus) {

        trainingStatus.textContent =
            "Training model...";

    }


    if (trainingResults) {

        trainingResults.classList.add(
            "hidden"
        );

    }


    try {

        const response =
            await fetch(
                `/api/gestures/`
                + `${state.gesture.id}`
                + `/train`,
                {
                    method: "POST",
                }
            );


        if (!response.ok) {

            let message =
                "Could not train model.";


            try {

                const errorData =
                    await response.json();


                if (
                    errorData.detail
                ) {

                    message =
                        errorData.detail;

                }

            }
            catch {
                // Ignore JSON parsing errors
            }


            throw new Error(
                message
            );

        }


        const data =
            await response.json();


        console.log(
            "Training result:",
            data
        );


        renderTrainingResult(
            data
        );

        state.trainingDatasetStatus = {
            status: "current",
            changed: false,
        };

        if (trainingStatus) {

            trainingStatus.textContent =
                "Training completed.";

        }

    }
    catch (error) {

        console.error(
            error
        );


        if (trainingStatus) {

            trainingStatus.textContent =
                error.message
                ?? "Training failed.";

        }


        alert(
            error.message
            ?? "Could not train model."
        );

    }
    finally {

        state.trainingInProgress =
            false;


        if (trainModelButton) {

            trainModelButton.disabled =
                false;


            trainModelButton.textContent =
                "Train Decision Tree";

        }

    }

}


// ==========================================================
// TRAIN BUTTON
// ==========================================================

if (trainModelButton) {

    trainModelButton.addEventListener(
        "click",
        trainCurrentGesture
    );

}



// ==========================================================
// TRAINING - DATASET SPLIT
// ==========================================================

const trainingTotalSamples =
    document.getElementById(
        "training-total-samples"
    );

const trainingTrainSamples =
    document.getElementById(
        "training-train-samples"
    );

const trainingValidationSamples =
    document.getElementById(
        "training-validation-samples"
    );

const trainingTrainPositive =
    document.getElementById(
        "training-train-positive"
    );

const trainingTrainNegative =
    document.getElementById(
        "training-train-negative"
    );

const trainingValidationPositive =
    document.getElementById(
        "training-validation-positive"
    );

const trainingValidationNegative =
    document.getElementById(
        "training-validation-negative"
    );


// ==========================================================
// TRAINING - VALIDATION METRICS
// ==========================================================

const trainingValidationAccuracy =
    document.getElementById(
        "training-validation-accuracy"
    );

const trainingValidationPrecision =
    document.getElementById(
        "training-validation-precision"
    );

const trainingValidationRecall =
    document.getElementById(
        "training-validation-recall"
    );

const trainingValidationF1 =
    document.getElementById(
        "training-validation-f1"
    );


// ==========================================================
// TRAINING - CONFUSION MATRIX
// ==========================================================

const trainingCmTn =
    document.getElementById(
        "training-cm-tn"
    );

const trainingCmFp =
    document.getElementById(
        "training-cm-fp"
    );

const trainingCmFn =
    document.getElementById(
        "training-cm-fn"
    );

const trainingCmTp =
    document.getElementById(
        "training-cm-tp"
    );


// ==========================================================
// TRAINING - VALIDATION SAMPLES
// ==========================================================

const trainingValidationSamplesBody =
    document.getElementById(
        "training-validation-samples-body"
    );


function formatTrainingMetric(
    value
) {

    if (
        value === null
        || value === undefined
        || Number.isNaN(
            Number(value)
        )
    ) {

        return "—";

    }


    return (
        `${(
            Number(value)
            * 100
        ).toFixed(1)}%`
    );

}


// ==========================================================
// NEGATIVE EXAMPLE SOURCE MODAL
// ==========================================================

function ensureNegativeSourceModal() {

    let modal =
        document.getElementById(
            "negative-source-modal"
        );


    if (modal) {
        return modal;
    }


    modal =
        document.createElement(
            "div"
        );


    modal.id =
        "negative-source-modal";


    modal.className =
        "samples-modal hidden";


    modal.innerHTML = `

        <div
            class="samples-modal-backdrop"
            data-close-negative-source
        ></div>


        <div
            class="samples-modal-panel negative-source-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="negative-source-title"
        >

            <div
                class="samples-modal-header"
            >

                <div>

                    <h2
                        id="negative-source-title"
                    >
                        Add Negative Example
                    </h2>

                    <p>
                        Record a new negative example
                        or reuse positive examples
                        from another compatible gesture.
                    </p>

                </div>


                <button
                    type="button"
                    id="close-negative-source"
                    class="samples-modal-close"
                    aria-label="Close"
                >
                    ×
                </button>

            </div>


            <div
                class="samples-modal-content"
            >

                <div
                    class="negative-source-options"
                >

                    <button
                        type="button"
                        id="negative-record-new"
                        class="negative-source-option"
                    >

                        <strong>
                            Record New
                        </strong>

                        <span>
                            Capture a new negative
                            example using the camera.
                        </span>

                    </button>


                    <button
                        type="button"
                        id="negative-import-library"
                        class="negative-source-option"
                    >

                        <strong>
                            Import from Library
                        </strong>

                        <span>
                            Reuse positive examples
                            from other gestures with
                            the same tracking target.
                        </span>

                    </button>

                </div>

            </div>

        </div>
    `;


    document.body.appendChild(
        modal
    );


    modal
        .querySelector(
            "#close-negative-source"
        )
        .addEventListener(
            "click",
            closeNegativeSourceModal
        );


    modal
        .querySelector(
            "[data-close-negative-source]"
        )
        .addEventListener(
            "click",
            closeNegativeSourceModal
        );


    modal
        .querySelector(
            "#negative-record-new"
        )
        .addEventListener(
            "click",
            () => {

                closeNegativeSourceModal();

                beginCapture(
                    "negative"
                );

            }
        );


    modal
        .querySelector(
            "#negative-import-library"
        )
        .addEventListener(
            "click",
            async () => {

                closeNegativeSourceModal();

                await openNegativeLibraryModal();

            }
        );


    return modal;

}


function openNegativeSourceModal() {

    if (!state.gesture) {

        alert(
            "No gesture selected."
        );

        return;

    }


    const modal =
        ensureNegativeSourceModal();


    modal.classList.remove(
        "hidden"
    );

}


function closeNegativeSourceModal() {

    const modal =
        document.getElementById(
            "negative-source-modal"
        );


    if (!modal) {
        return;
    }


    modal.classList.add(
        "hidden"
    );

}


// ==========================================================
// NEGATIVE LIBRARY MODAL
// ==========================================================

function ensureNegativeLibraryModal() {

    let modal =
        document.getElementById(
            "negative-library-modal"
        );


    if (modal) {
        return modal;
    }


    modal =
        document.createElement(
            "div"
        );


    modal.id =
        "negative-library-modal";


    modal.className =
        "samples-modal hidden";


    modal.innerHTML = `

        <div
            class="samples-modal-backdrop"
            data-close-negative-library
        ></div>


        <div
            class="samples-modal-panel negative-library-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="negative-library-title"
        >

            <div
                class="samples-modal-header"
            >

                <div>

                    <h2
                        id="negative-library-title"
                    >
                        Import Negative Examples
                    </h2>

                    <p
                        id="negative-library-subtitle"
                    >
                    </p>

                </div>


                <button
                    type="button"
                    id="close-negative-library"
                    class="samples-modal-close"
                    aria-label="Close"
                >
                    ×
                </button>

            </div>


            <div
                id="negative-library-content"
                class="samples-modal-content"
            >

                <div
                    class="empty-state"
                >
                    Loading library...
                </div>

            </div>


            <div
                class="negative-library-footer"
            >

                <span
                    id="negative-library-selection-count"
                >
                    0 samples selected
                </span>


                <div
                    class="negative-library-footer-actions"
                >

                    <button
                        type="button"
                        id="cancel-negative-library"
                    >
                        Cancel
                    </button>


                    <button
                        type="button"
                        id="import-negative-library"
                        class="primary"
                        disabled
                    >
                        Import Selected
                    </button>

                </div>

            </div>

        </div>
    `;


    document.body.appendChild(
        modal
    );


    modal
        .querySelector(
            "#close-negative-library"
        )
        .addEventListener(
            "click",
            closeNegativeLibraryModal
        );


    modal
        .querySelector(
            "[data-close-negative-library]"
        )
        .addEventListener(
            "click",
            closeNegativeLibraryModal
        );


    modal
        .querySelector(
            "#cancel-negative-library"
        )
        .addEventListener(
            "click",
            closeNegativeLibraryModal
        );


    modal
        .querySelector(
            "#import-negative-library"
        )
        .addEventListener(
            "click",
            importSelectedNegativeLibrarySamples
        );


    return modal;

}


function closeNegativeLibraryModal() {

    const modal =
        document.getElementById(
            "negative-library-modal"
        );


    if (!modal) {
        return;
    }


    modal.classList.add(
        "hidden"
    );

}


// ==========================================================
// LOAD NEGATIVE LIBRARY
// ==========================================================

async function loadNegativeLibrary() {

    if (!state.gesture) {

        throw new Error(
            "No gesture selected."
        );

    }


    const response =
        await fetch(
            `/api/gestures/`
            + `${state.gesture.id}`
            + `/negative-library`
        );


    if (!response.ok) {

        let message =
            "Could not load negative library.";


        try {

            const errorData =
                await response.json();


            if (
                errorData.detail
            ) {

                message =
                    errorData.detail;

            }

        }
        catch {
            // Ignore JSON errors
        }


        throw new Error(
            message
        );

    }


    return (
        await response.json()
    );

}


// ==========================================================
// OPEN NEGATIVE LIBRARY
// ==========================================================

async function openNegativeLibraryModal() {

    if (!state.gesture) {
        return;
    }


    const modal =
        ensureNegativeLibraryModal();


    const content =
        modal.querySelector(
            "#negative-library-content"
        );


    const subtitle =
        modal.querySelector(
            "#negative-library-subtitle"
        );


    subtitle.textContent =
        `${state.gesture.name} · `
        + `${state.gesture.target === "hand"
            ? "Hand"
            : "Body"}`;


    content.innerHTML = `

        <div
            class="empty-state"
        >
            Loading library...
        </div>
    `;


    modal.classList.remove(
        "hidden"
    );


    updateNegativeLibrarySelection();


    try {

        const library =
            await loadNegativeLibrary();


        renderNegativeLibrary(
            library
        );

    }
    catch (error) {

        console.error(
            error
        );


        content.innerHTML = `

            <div
                class="empty-state"
            >
                ${escapeHTML(
                    error.message
                    ?? "Could not load library."
                )}
            </div>
        `;

    }

}


// ==========================================================
// RENDER NEGATIVE LIBRARY
// ==========================================================

function renderNegativeLibrary(
    library
) {

    const content =
        document.getElementById(
            "negative-library-content"
        );


    if (!content) {
        return;
    }


    content.innerHTML = "";


    const gestures =
        library.gestures
        ?? [];


    if (
        gestures.length === 0
    ) {

        content.innerHTML = `

            <div
                class="empty-state"
            >
                No compatible positive examples
                are available in the library.
            </div>
        `;

        return;

    }


    const list =
        document.createElement(
            "div"
        );


    list.className =
        "negative-library-list";


    for (
        const gesture
        of gestures
    ) {

        const group =
            document.createElement(
                "section"
            );


        group.className =
            "negative-library-group";


        const availableCount =
            gesture.available_count
            ?? 0;


        group.innerHTML = `

            <div
                class="negative-library-group-header"
            >

                <div>

                    <h3>
                        ${escapeHTML(
                            gesture.name
                        )}
                    </h3>

                    <p>
                        ${escapeHTML(
                            gesture.sample_count
                            ?? 0
                        )}
                        positive examples ·
                        ${escapeHTML(
                            availableCount
                        )}
                        available
                    </p>

                </div>


                <button
                    type="button"
                    class="negative-library-select-all"
                    ${
                        availableCount === 0
                            ? "disabled"
                            : ""
                    }
                >
                    Select all
                </button>

            </div>


            <div
                class="negative-library-samples"
            >
            </div>
        `;


        const samplesContainer =
            group.querySelector(
                ".negative-library-samples"
            );


        for (
            const sample
            of gesture.samples
            ?? []
        ) {

            const row =
                createNegativeLibrarySampleRow(
                    gesture,
                    sample
                );


            samplesContainer.appendChild(
                row
            );

        }


        const selectAllButton =
            group.querySelector(
                ".negative-library-select-all"
            );


        selectAllButton
            .addEventListener(
                "click",
                () => {

                    toggleNegativeLibraryGroup(
                        group
                    );

                }
            );


        list.appendChild(
            group
        );

    }


    content.appendChild(
        list
    );


    updateNegativeLibrarySelection();

}


// ==========================================================
// NEGATIVE LIBRARY SAMPLE ROW
// ==========================================================

function createNegativeLibrarySampleRow(
    gesture,
    sample
) {

    const row =
        document.createElement(
            "label"
        );


    row.className =
        "negative-library-sample";


    const alreadyImported =
        Boolean(
            sample.already_imported
        );


    if (alreadyImported) {

        row.classList.add(
            "already-imported"
        );

    }


    row.innerHTML = `

        <input
            type="checkbox"
            class="negative-library-checkbox"
            data-source-gesture-id="${escapeHTML(
                gesture.id
            )}"
            data-source-sample-id="${escapeHTML(
                sample.id
            )}"
            ${
                alreadyImported
                    ? "disabled"
                    : ""
            }
        >


        <div
            class="negative-library-sample-info"
        >

            <div
                class="negative-library-sample-title"
            >

                <strong>
                    ${escapeHTML(
                        shortSampleId(
                            sample.id
                        )
                    )}
                </strong>

                ${
                    alreadyImported
                        ? `
                            <span
                                class="negative-library-imported-badge"
                            >
                                Already imported
                            </span>
                        `
                        : ""
                }

            </div>


            <code>
                ${escapeHTML(
                    sample.id
                )}
            </code>


            <div
                class="negative-library-sample-meta"
            >

                <span>
                    ${escapeHTML(
                        formatDurationMs(
                            sample.duration_ms
                        )
                    )}
                </span>

                <span>
                    ${escapeHTML(
                        formatDate(
                            sample.created_at
                        )
                    )}
                </span>

            </div>

        </div>
    `;


    const checkbox =
        row.querySelector(
            ".negative-library-checkbox"
        );


    checkbox.addEventListener(
        "change",
        updateNegativeLibrarySelection
    );


    return row;

}


// ==========================================================
// SELECT ALL NEGATIVE LIBRARY GROUP
// ==========================================================

function toggleNegativeLibraryGroup(
    group
) {

    const checkboxes = [
        ...group.querySelectorAll(
            ".negative-library-checkbox:not(:disabled)"
        ),
    ];


    if (
        checkboxes.length === 0
    ) {
        return;
    }


    const allSelected =
        checkboxes.every(
            checkbox =>
                checkbox.checked
        );


    for (
        const checkbox
        of checkboxes
    ) {

        checkbox.checked =
            !allSelected;

    }


    updateNegativeLibrarySelection();

}


// ==========================================================
// GET SELECTED LIBRARY SAMPLES
// ==========================================================

function getSelectedNegativeLibrarySamples() {

    const modal =
        document.getElementById(
            "negative-library-modal"
        );


    if (!modal) {
        return [];
    }


    const selected =
        modal.querySelectorAll(
            ".negative-library-checkbox:checked"
        );


    return [
        ...selected
    ].map(
        checkbox => ({

            gesture_id:
                checkbox.dataset
                    .sourceGestureId,

            sample_id:
                checkbox.dataset
                    .sourceSampleId,

        })
    );

}


// ==========================================================
// UPDATE LIBRARY SELECTION
// ==========================================================

function updateNegativeLibrarySelection() {

    const modal =
        document.getElementById(
            "negative-library-modal"
        );


    if (!modal) {
        return;
    }


    const selected =
        getSelectedNegativeLibrarySamples();


    const counter =
        modal.querySelector(
            "#negative-library-selection-count"
        );


    const importButton =
        modal.querySelector(
            "#import-negative-library"
        );


    const count =
        selected.length;


    counter.textContent =
        count === 1
            ? "1 sample selected"
            : `${count} samples selected`;


    importButton.disabled =
        count === 0;


    importButton.textContent =
        count === 0
            ? "Import Selected"
            : `Import ${count} `
                + `${count === 1
                    ? "Sample"
                    : "Samples"}`;

}


// ==========================================================
// IMPORT SELECTED LIBRARY SAMPLES
// ==========================================================

async function importSelectedNegativeLibrarySamples() {

    if (!state.gesture) {
        return;
    }


    const selected =
        getSelectedNegativeLibrarySamples();


    if (
        selected.length === 0
    ) {

        return;

    }


    const modal =
        document.getElementById(
            "negative-library-modal"
        );


    const importButton =
        modal.querySelector(
            "#import-negative-library"
        );


    const originalText =
        importButton.textContent;


    importButton.disabled =
        true;


    importButton.textContent =
        "Importing...";


    try {

        const response =
            await fetch(
                `/api/gestures/`
                + `${state.gesture.id}`
                + `/negative-library/import`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",
                    },

                    body: JSON.stringify({
                        samples:
                            selected,
                    }),
                }
            );


        if (!response.ok) {

            let message =
                "Could not import samples.";


            try {

                const errorData =
                    await response.json();


                if (
                    errorData.detail
                ) {

                    message =
                        errorData.detail;

                }

            }
            catch {
                // Ignore JSON parsing errors
            }


            throw new Error(
                message
            );

        }


        const result =
            await response.json();


        console.log(
            "Library import result:",
            result
        );


        // --------------------------------------------------
        // Dataset changed -> previous training is outdated.
        // --------------------------------------------------

        resetTrainingUI();


        // --------------------------------------------------
        // Refresh gesture counters.
        // --------------------------------------------------

        await refreshCurrentGesture();


        updateCurrentGestureCounters();


        // --------------------------------------------------
        // Refresh local sample cache.
        // --------------------------------------------------

        await loadSamples();


        // --------------------------------------------------
        // Reload stored training status so the UI can show
        // that the previous model is outdated.
        // --------------------------------------------------

        await loadStoredTraining();


        closeNegativeLibraryModal();


        const importedCount =
            result.imported_count
            ?? 0;


        const skippedCount =
            result.skipped_count
            ?? 0;


        if (
            skippedCount > 0
        ) {

            alert(
                `${importedCount} sample(s) imported. `
                + `${skippedCount} already existed `
                + `and were skipped.`
            );

        }
        else {

            alert(
                `${importedCount} negative `
                + `${importedCount === 1
                    ? "example"
                    : "examples"} imported.`
            );

        }

    }
    catch (error) {

        console.error(
            error
        );


        alert(
            error.message
            ?? "Could not import samples."
        );


        importButton.disabled =
            false;


        importButton.textContent =
            originalText;

    }

}

// ==========================================================
// SAMPLE TYPE
// ==========================================================

document
    .getElementById(
        "positive-example"
    )
    .addEventListener(
        "click",
        () => {

            beginCapture(
                "positive"
            );

        }
    );


document
    .getElementById(
        "negative-example"
    )
    .addEventListener(
        "click",
        () => {

            openNegativeSourceModal();

        }
    );


async function beginCapture(
    sampleType
) {

    state.sampleType =
        sampleType;


    document
        .getElementById(
            "capture-type"
        )
        .textContent =
            sampleType === "positive"
                ? "Positive example"
                : "Negative example";


    try {

        await startCamera();


        showScreen(
            "capture"
        );

    }
    catch (error) {

        console.error(
            error
        );


        alert(
            "Could not access the camera."
        );

    }

}


// ==========================================================
// CAMERA
// ==========================================================

async function startCamera() {

    if (state.stream) {
        return;
    }


    state.stream =
        await navigator
            .mediaDevices
            .getUserMedia({
                video: true,
                audio: false,
            });


    cameraPreview.srcObject =
        state.stream;

}


function stopCamera() {

    if (!state.stream) {
        return;
    }


    state.stream
        .getTracks()
        .forEach(
            track => track.stop()
        );


    state.stream = null;


    cameraPreview.srcObject =
        null;

}


// ==========================================================
// RECORDING OPTIONS
// ==========================================================

function getRecorderOptions() {

    const candidates = [

        "video/webm;codecs=vp9",

        "video/webm;codecs=vp8",

        "video/webm",

    ];


    for (
        const mimeType
        of candidates
    ) {

        if (
            MediaRecorder
                .isTypeSupported(
                    mimeType
                )
        ) {

            return {
                mimeType,
            };

        }

    }


    return {};

}


// ==========================================================
// START RECORDING
// ==========================================================

document
    .getElementById(
        "start-recording"
    )
    .addEventListener(
        "click",
        () => {

            if (!state.stream) {

                alert(
                    "Camera is not available."
                );

                return;

            }


            state.chunks = [];


            state.recorder =
                new MediaRecorder(
                    state.stream,
                    getRecorderOptions()
                );


            state.recorder
                .addEventListener(
                    "dataavailable",
                    event => {

                        if (
                            event.data.size > 0
                        ) {

                            state.chunks.push(
                                event.data
                            );

                        }

                    }
                );


            state.recorder
                .addEventListener(
                    "stop",
                    prepareReview
                );


            state.recorder.start();


            document
                .getElementById(
                    "start-recording"
                )
                .classList.add(
                    "hidden"
                );


            document
                .getElementById(
                    "stop-recording"
                )
                .classList.remove(
                    "hidden"
                );

        }
    );


// ==========================================================
// STOP RECORDING
// ==========================================================

document
    .getElementById(
        "stop-recording"
    )
    .addEventListener(
        "click",
        () => {

            if (
                state.recorder
                &&
                state.recorder.state
                    !== "inactive"
            ) {

                state.recorder.stop();

            }

        }
    );


// ==========================================================
// PREPARE REVIEW
// ==========================================================

function prepareReview() {

    const mimeType =
        state.recorder.mimeType
        || "video/webm";


    state.videoBlob =
        new Blob(
            state.chunks,
            {
                type: mimeType,
            }
        );


    if (
        state.videoURL
    ) {

        URL.revokeObjectURL(
            state.videoURL
        );

    }


    state.videoURL =
        URL.createObjectURL(
            state.videoBlob
        );


    reviewVideo.src =
        state.videoURL;


    reviewVideo.onloadedmetadata =
        () => {

            const duration =
                reviewVideo.duration;


            trimStart.min = 0;
            trimStart.max = duration;
            trimStart.value = 0;


            trimEnd.min = 0;
            trimEnd.max = duration;
            trimEnd.value = duration;


            updateTrimUI();

        };


    document
        .getElementById(
            "start-recording"
        )
        .classList.remove(
            "hidden"
        );


    document
        .getElementById(
            "stop-recording"
        )
        .classList.add(
            "hidden"
        );


    showScreen(
        "trim"
    );

}


// ==========================================================
// TRIM UI
// ==========================================================

function updateTrimUI() {

    let start =
        Number(
            trimStart.value
        );


    let end =
        Number(
            trimEnd.value
        );


    if (
        start >= end
    ) {

        start =
            Math.max(
                0,
                end - 0.01
            );


        trimStart.value =
            start;

    }


    trimStartLabel.textContent =
        `${start.toFixed(2)} s`;


    trimEndLabel.textContent =
        `${end.toFixed(2)} s`;


    selectionInfo.textContent =
        `Selected: `
        + `${start.toFixed(2)} s → `
        + `${end.toFixed(2)} s `
        + `(${(end - start).toFixed(2)} s)`;

}


// ==========================================================
// TRIM SLIDERS
// ==========================================================

trimStart.addEventListener(
    "input",
    () => {

        const start =
            Number(
                trimStart.value
            );


        const end =
            Number(
                trimEnd.value
            );


        if (
            start >= end
        ) {

            trimStart.value =
                Math.max(
                    0,
                    end - 0.01
                );

        }


        reviewVideo.currentTime =
            Number(
                trimStart.value
            );


        updateTrimUI();

    }
);


trimEnd.addEventListener(
    "input",
    () => {

        const start =
            Number(
                trimStart.value
            );


        const end =
            Number(
                trimEnd.value
            );


        if (
            end <= start
        ) {

            trimEnd.value =
                Math.min(
                    Number(
                        trimEnd.max
                    ),
                    start + 0.01
                );

        }


        reviewVideo.currentTime =
            Number(
                trimEnd.value
            );


        updateTrimUI();

    }
);


// ==========================================================
// SET START
// ==========================================================

document
    .getElementById(
        "set-start"
    )
    .addEventListener(
        "click",
        () => {

            const current =
                reviewVideo.currentTime;


            if (
                current
                < Number(
                    trimEnd.value
                )
            ) {

                trimStart.value =
                    current;


                updateTrimUI();

            }

        }
    );


// ==========================================================
// SET END
// ==========================================================

document
    .getElementById(
        "set-end"
    )
    .addEventListener(
        "click",
        () => {

            const current =
                reviewVideo.currentTime;


            if (
                current
                > Number(
                    trimStart.value
                )
            ) {

                trimEnd.value =
                    current;


                updateTrimUI();

            }

        }
    );


// ==========================================================
// PLAY SELECTED REGION
// ==========================================================

document
    .getElementById(
        "play-selection"
    )
    .addEventListener(
        "click",
        async () => {

            reviewVideo.pause();


            reviewVideo.currentTime =
                Number(
                    trimStart.value
                );


            try {

                await reviewVideo.play();

            }
            catch (error) {

                console.error(
                    error
                );

            }

        }
    );


reviewVideo.addEventListener(
    "timeupdate",
    () => {

        if (
            reviewVideo.currentTime
            >= Number(
                trimEnd.value
            )
        ) {

            reviewVideo.pause();

        }

    }
);


// ==========================================================
// RETAKE
// ==========================================================

document
    .getElementById(
        "retake"
    )
    .addEventListener(
        "click",
        () => {

            reviewVideo.pause();


            showScreen(
                "capture"
            );

        }
    );


// ==========================================================
// CANCEL CAPTURE
// ==========================================================

document
    .getElementById(
        "cancel-capture"
    )
    .addEventListener(
        "click",
        () => {

            if (
                state.recorder
                &&
                state.recorder.state
                    !== "inactive"
            ) {

                state.recorder.stop();

            }


            stopCamera();


            openGesture();

        }
    );


// ==========================================================
// CONFIRM CLIP
// ==========================================================

document
    .getElementById(
        "confirm-clip"
    )
    .addEventListener(
        "click",
        async () => {

            if (
                !state.gesture
                ||
                !state.videoBlob
            ) {

                alert(
                    "No clip available."
                );

                return;

            }


            const trimStartMs =
                Math.round(
                    Number(
                        trimStart.value
                    )
                    * 1000
                );


            const trimEndMs =
                Math.round(
                    Number(
                        trimEnd.value
                    )
                    * 1000
                );


            if (
                trimEndMs
                <= trimStartMs
            ) {

                alert(
                    "Invalid trim range."
                );

                return;

            }


            const formData =
                new FormData();


            formData.append(
                "sample_type",
                state.sampleType
            );


            formData.append(
                "trim_start_ms",
                trimStartMs
            );


            formData.append(
                "trim_end_ms",
                trimEndMs
            );


            formData.append(
                "video",
                state.videoBlob,
                "capture.webm"
            );


            try {

                const response =
                    await fetch(
                        `/api/gestures/`
                        + `${state.gesture.id}`
                        + `/samples`,
                        {
                            method: "POST",
                            body: formData,
                        }
                    );


                if (!response.ok) {

                    let message =
                        "Could not save clip.";


                    try {

                        const errorData =
                            await response.json();


                        if (
                            errorData.detail
                        ) {

                            message =
                                errorData.detail;

                        }

                    }
                    catch {
                        // Ignore JSON errors
                    }


                    alert(
                        message
                    );

                    return;

                }


                const sample =
                    await response.json();


                console.log(
                    "Sample saved:",
                    sample
                );


                stopCamera();


                await refreshCurrentGesture();


                openGesture();

            }
            catch (error) {

                console.error(
                    error
                );


                alert(
                    "Could not communicate with the server."
                );

            }

        }
    );


// ==========================================================
// BACK TO GESTURE LIST
// ==========================================================

document
    .getElementById(
        "new-gesture"
    )
    .addEventListener(
        "click",
        async () => {

            stopCamera();


            reviewVideo.pause();


            closeSampleViewer();

            closeSamplesModal();

            resetTrainingUI();

            state.gesture = null;

            state.sampleType = null;

            state.selectedSample = null;


            state.samples = {
                positive: [],
                negative: [],
            };


            await loadGestures();


            showScreen(
                "gesture"
            );

        }
    );


// ==========================================================
// INITIALIZATION
// ==========================================================

async function initialize() {

    setupSampleCounters();


    ensureSamplesModal();

    ensureSampleViewerModal();


    await loadGestures();


    showScreen(
        "gesture"
    );

}


initialize();