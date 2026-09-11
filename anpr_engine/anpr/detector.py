"""
detector.py
===========
Number-plate detection ONLY.

Given a full video frame and a vehicle bounding box (produced upstream
by the separate Vehicle Detection Engine), this module crops the
vehicle region and runs a dedicated plate-detection YOLO model inside
that crop. It never looks at the full frame for detection (cheaper,
and avoids picking up plates belonging to other vehicles) and it never
attempts to detect or classify the vehicle itself.

Model: Ultralytics YOLO (v8/v11-style .pt weights), trained solely on
a "license-plate" class. Swap `plate_detector_weights` in config.py to
point at your trained weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import ANPRConfig
from .utils.image_utils import expand_bbox, safe_crop, translate_bbox, bbox_area, clip_bbox

logger = logging.getLogger("anpr.detector")


@dataclass
class PlateDetection:
    bbox: tuple  # full-frame coordinates (x1, y1, x2, y2)
    confidence: float


class PlateDetector:
    """Wraps a YOLO model specialized for license-plate detection."""

    def __init__(self, config: ANPRConfig):
        self.config = config
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        try:
            self._model = YOLO(self.config.plate_detector_weights)
            logger.info("Loaded plate detection model from %s", self.config.plate_detector_weights)
        except Exception as exc:
            logger.error("Failed to load plate detector weights: %s", exc)
            raise

    def detect(self, frame: np.ndarray, vehicle_bbox: tuple) -> List[PlateDetection]:
        """
        Detect number plate(s) inside the given vehicle bounding box.

        Returns a list of PlateDetection sorted by confidence, descending,
        capped at config.max_plates_per_vehicle. Empty list if nothing
        found or inputs are invalid.
        """
        if frame is None or frame.size == 0:
            logger.warning("detect() received an empty frame")
            return []

        h, w = frame.shape[:2]
        vehicle_bbox = clip_bbox(vehicle_bbox, w, h)
        if bbox_area(vehicle_bbox) <= 0:
            logger.warning("detect() received a degenerate vehicle_bbox=%s", vehicle_bbox)
            return []

        search_bbox = expand_bbox(vehicle_bbox, self.config.vehicle_crop_margin, w, h)
        vehicle_crop = safe_crop(frame, search_bbox)
        if vehicle_crop is None:
            return []

        origin = (search_bbox[0], search_bbox[1])

        try:
            results = self._model.predict(
                source=vehicle_crop,
                conf=self.config.plate_detection_conf_threshold,
                iou=self.config.plate_detection_iou_threshold,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Plate detection inference failed: %s", exc)
            return []

        detections: List[PlateDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                local_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                full_frame_bbox = translate_bbox(local_bbox, origin)
                full_frame_bbox = clip_bbox(full_frame_bbox, w, h)
                detections.append(PlateDetection(bbox=full_frame_bbox, confidence=conf))

        detections.sort(key=lambda d: d.confidence, reverse=True)

        if len(detections) > 1:
            logger.debug(
                "Multiple plate candidates (%d) found for vehicle_bbox=%s; keeping top %d",
                len(detections), vehicle_bbox, self.config.max_plates_per_vehicle,
            )

        return detections[: self.config.max_plates_per_vehicle]
