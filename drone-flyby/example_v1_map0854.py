"""
FlySight V1 detector for Nordic AI Cup 2026 - Drone Flyby.

V1 goals:
1. Run the trained Nordic YOLO detector.
2. Convert detections from the received 960x540 view into the
   frame-global normalized coordinates required by the evaluator.
3. Keep the camera at Level 0 for a clean detector baseline.

Later versions will add:
- persistent world memory
- Scout / Tracker / Inspector / Sentinel agents
- active L0/L1/L2 camera control
- information-gain camera arbitration
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from dtos import (
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)

from utils import (
    clip_bbox_to_frame,
    decode_view,
    view_bbox_to_global,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FlySight YOLO model
# ---------------------------------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "runs"
    / "detect"
    / "runs"
    / "flysight"
    / "yolo11n_camera_v1"
    / "weights"
    / "best.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"FlySight model not found at: {MODEL_PATH}"
    )

logger.info("Loading FlySight model from %s", MODEL_PATH)

MODEL = YOLO(str(MODEL_PATH))


# Start with a permissive confidence threshold.
# We will tune this using local_evaluator.py.
YOLO_CONFIDENCE = 0.10

# YOLO NMS IoU threshold.
YOLO_IOU = 0.70

# Avoid pathological response sizes.
MAX_DETECTIONS = 100


# ---------------------------------------------------------------------------
# Main competition entry point
# ---------------------------------------------------------------------------

def predict(
    request: DroneFlybyPredictRequestDto,
) -> DroneFlybyPredictResponseDto:
    """
    Process one competition frame.

    The incoming image is always the transmitted camera view.
    Detections must be returned in normalized coordinates relative
    to the complete original source frame.
    """

    if request.camera_command_feedback is not None:
        feedback = request.camera_command_feedback

        logger.warning(
            "Camera command from frame %s was ignored: %s",
            feedback.frame,
            feedback.reason,
        )

    image = decode_view(request.view)

    try:
        annotations = detect(image, request)

    except Exception:
        # An empty response is preferable to losing the complete frame
        # because of an exception.
        logger.exception(
            "FlySight detector failed on frame %s",
            request.frame,
        )

        annotations = []

    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=choose_next_view(request),
    )


# ---------------------------------------------------------------------------
# FlySight V1 detector
# ---------------------------------------------------------------------------

def detect(
    image: np.ndarray,
    request: DroneFlybyPredictRequestDto,
) -> List[DroneFlybyPredictionDto]:
    """
    Detect Nordic objects using the trained YOLO model.

    YOLO produces bounding boxes in pixels relative to the received
    960x540 view.

    The competition expects normalized bounding boxes relative to
    the complete original frame.

    Conversion:

        YOLO view pixels
             ↓
        view normalized [0,1]
             ↓
        source_region_xyxy
             ↓
        original source pixels
             ↓
        frame-global normalized [0,1]
    """

    height, width = image.shape[:2]

    results = MODEL.predict(
        source=image,
        imgsz=960,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        device=0,
        verbose=False,
        max_det=MAX_DETECTIONS,
    )

    annotations: List[DroneFlybyPredictionDto] = []

    if not results:
        return annotations

    result = results[0]

    if result.boxes is None:
        return annotations

    for detection in result.boxes:

        # ---------------------------------------------------------------
        # Read YOLO detection
        # ---------------------------------------------------------------

        x1, y1, x2, y2 = (
            detection.xyxy[0]
            .detach()
            .cpu()
            .tolist()
        )

        confidence = float(
            detection.conf[0]
            .detach()
            .cpu()
        )

        class_id = int(
            detection.cls[0]
            .detach()
            .cpu()
        )

        object_id = str(result.names[class_id])


        # ---------------------------------------------------------------
        # YOLO pixels -> normalized coordinates inside current view
        # ---------------------------------------------------------------

        view_bbox = (
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
        )


        # ---------------------------------------------------------------
        # Current camera view -> complete 3840x2160 source frame
        # ---------------------------------------------------------------

        global_bbox = view_bbox_to_global(
            view_bbox,
            request.view.source_region_xyxy,
            request.original_width,
            request.original_height,
        )


        # ---------------------------------------------------------------
        # Protect evaluator against invalid / degenerate boxes
        # ---------------------------------------------------------------

        global_bbox = clip_bbox_to_frame(global_bbox)

        if global_bbox is None:
            continue


        # ---------------------------------------------------------------
        # Build competition DTO
        # ---------------------------------------------------------------

        annotation = DroneFlybyPredictionDto(
            object_id=object_id,
            bbox=[
                float(global_bbox[0]),
                float(global_bbox[1]),
                float(global_bbox[2]),
                float(global_bbox[3]),
            ],
            confidence=float(
                max(0.0, min(1.0, confidence))
            ),
        )

        annotations.append(annotation)


    logger.info(
        "Frame %s | L%s | detections=%s",
        request.frame,
        request.view.resolution_level,
        len(annotations),
    )

    return annotations


# ---------------------------------------------------------------------------
# FlySight V1 camera policy
# ---------------------------------------------------------------------------

def choose_next_view(
    request: DroneFlybyPredictRequestDto,
) -> Optional[RequestedViewDto]:
    """
    V1 deliberately keeps the camera at Level 0.

    Why?

    We first need a clean measurement of detector performance over the
    complete source frame.

    The baseline sweep immediately zooms into only part of the scene.
    Ground truth, however, continues to cover the complete frame.

    Active perception will be introduced after this baseline works:
        V2 - world memory
        V3 - Scout / Tracker
        V4 - Inspector
        V5 - active zoom
        V6 - Swarm Arbiter
    """

    return None
