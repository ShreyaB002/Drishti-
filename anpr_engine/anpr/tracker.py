"""
tracker.py
==========
Optional per-vehicle Kalman filter for SMOOTHING/PREDICTING the plate
bounding-box location across consecutive video frames.

IMPORTANT: this is strictly a spatial tracker. It estimates where the
plate rectangle is likely to be (useful for stabilizing crops and
skipping a full re-detection every single frame), and it must NEVER be
used to infer or "fill in" characters of the plate text. Character
recovery is handled exclusively by fusion.py's confidence-weighted
voting over actual OCR observations.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger("anpr.tracker")

BBox = Tuple[int, int, int, int]


class _BBoxKalmanFilter:
    """
    Tracks the plate bbox center (cx, cy) and size (w, h) with a
    constant-velocity Kalman filter (state: cx, cy, w, h, vcx, vcy).
    """

    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ], dtype=np.float32)

        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self._initialized = False

    @staticmethod
    def _bbox_to_measurement(bbox: BBox) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = float(x2 - x1), float(y2 - y1)
        return np.array([[cx], [cy], [w], [h]], dtype=np.float32)

    @staticmethod
    def _state_to_bbox(state: np.ndarray) -> BBox:
        cx, cy, w, h = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        return (
            int(cx - w / 2), int(cy - h / 2),
            int(cx + w / 2), int(cy + h / 2),
        )

    def update(self, bbox: BBox) -> BBox:
        measurement = self._bbox_to_measurement(bbox)
        if not self._initialized:
            self.kf.statePost = np.array(
                [[measurement[0, 0]], [measurement[1, 0]],
                 [measurement[2, 0]], [measurement[3, 0]], [0], [0]],
                dtype=np.float32,
            )
            self._initialized = True
            return bbox

        self.kf.predict()
        corrected = self.kf.correct(measurement)
        return self._state_to_bbox(corrected)

    def predict_only(self) -> Optional[BBox]:
        if not self._initialized:
            return None
        state = self.kf.predict()
        return self._state_to_bbox(state)


class PlateTracker:
    """Per-vehicle_id registry of bbox Kalman filters."""

    def __init__(self):
        self._filters: Dict[str, _BBoxKalmanFilter] = {}

    def update(self, vehicle_id: str, bbox: BBox) -> BBox:
        """Feed a new detected bbox in; returns the smoothed bbox."""
        if vehicle_id not in self._filters:
            self._filters[vehicle_id] = _BBoxKalmanFilter()
        return self._filters[vehicle_id].update(bbox)

    def predict(self, vehicle_id: str) -> Optional[BBox]:
        """Predict the plate location for a frame where detection was skipped/missed."""
        tracker = self._filters.get(vehicle_id)
        if tracker is None:
            return None
        return tracker.predict_only()

    def reset(self, vehicle_id: str) -> None:
        self._filters.pop(vehicle_id, None)
