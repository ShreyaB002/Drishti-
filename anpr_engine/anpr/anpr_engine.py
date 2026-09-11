"""
anpr_engine.py
==============
Top-level orchestrator: wires detector -> preprocessor -> OCR ->
validator -> (optional) temporal fusion into a single reusable
`ANPREngine` class, consumed by the rest of the IBVAP system.

Scope reminder: this engine receives an already-detected vehicle
region (bbox) from the external Vehicle Detection Engine. It performs
no vehicle detection/classification of its own.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import ANPRConfig
from .detector import PlateDetector, PlateDetection
from .preprocessor import PlatePreprocessor
from .ocr import PlateOCR, OCRResult
from .validator import PlateValidator, ValidationResult
from .fusion import TemporalFusion, FrameObservation
from .tracker import PlateTracker
from .utils.image_utils import safe_crop, bbox_area

logger = logging.getLogger("anpr.engine")

BBox = Tuple[int, int, int, int]

STATUS_VALIDATED = "VALIDATED"
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"
STATUS_UNRELIABLE = "UNRELIABLE"
STATUS_NO_PLATE_DETECTED = "NO_PLATE_DETECTED"
STATUS_NO_TEXT_READ = "NO_TEXT_READ"
STATUS_ERROR = "ERROR"


def _empty_result(
    reason: str,
    vehicle_id: Optional[str],
    camera_id: Optional[str],
    timestamp: Optional[str],
    plate_bbox: Optional[BBox] = None,
) -> Dict[str, Any]:
    return {
        "plate_number": None,
        "confidence": 0.0,
        "plate_bbox": plate_bbox,
        "vehicle_id": vehicle_id,
        "camera_id": camera_id,
        "timestamp": timestamp,
        "frames_used": 0,
        "valid_format": False,
        "status": reason,
    }


class ANPREngine:
    """
    Reusable ANPR engine.

    Typical usage (single image / one frame at a time):

        engine = ANPREngine()
        result = engine.process(
            frame=frame,
            vehicle_bbox=(x1, y1, x2, y2),
            vehicle_id="V_042",
            camera_id="CAM_07",
            timestamp="2026-09-09T10:15:00Z",
        )

    For video, call `process()` once per frame with the SAME
    `vehicle_id` for the same physical vehicle (as provided by the
    Vehicle Detection Engine's tracker) and pass `use_temporal_fusion=True`.
    Each call still returns a best-effort single-frame result, but also
    updates an internal rolling buffer; call `get_fused_result(vehicle_id)`
    (or rely on the auto-fused fields in the returned dict) to obtain the
    temporally-fused final reading.
    """

    def __init__(self, config: Optional[ANPRConfig] = None):
        self.config = config or ANPRConfig()
        logging.basicConfig(level=self.config.log_level)

        logger.info("Initializing ANPR engine components...")
        self.detector = PlateDetector(self.config)
        self.preprocessor = PlatePreprocessor(self.config)
        self.ocr = PlateOCR(self.config)
        self.validator = PlateValidator(self.config)
        self.fusion = TemporalFusion(self.config)
        self.tracker = PlateTracker()
        logger.info("ANPR engine ready.")

    # -- public API ------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        vehicle_bbox: BBox,
        vehicle_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        use_temporal_fusion: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full single-frame pipeline. If `use_temporal_fusion` is
        True and `vehicle_id` is provided, the observation is also fed
        into the rolling fusion buffer and the returned result reflects
        the fused (multi-frame) reading whenever enough frames are
        available.
        """
        try:
            return self._process_impl(
                frame, vehicle_bbox, vehicle_id, camera_id, timestamp, use_temporal_fusion
            )
        except Exception:
            logger.exception("Unhandled error while processing frame")
            return _empty_result(STATUS_ERROR, vehicle_id, camera_id, timestamp)

    def get_fused_result(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the current best fused reading for a tracked vehicle_id."""
        fusion_result = self.fusion.fuse(vehicle_id)
        if fusion_result is None:
            return None
        validation = self.validator.validate(fusion_result.final_text)
        return {
            "plate_number": validation.corrected_text,
            "confidence": fusion_result.confidence,
            "frames_used": fusion_result.frames_used,
            "valid_format": validation.is_valid,
            "status": self._status_for(fusion_result.confidence, validation.is_valid),
        }

    def reset_vehicle_track(self, vehicle_id: str) -> None:
        """Call this once a tracked vehicle leaves the scene."""
        self.fusion.reset(vehicle_id)
        self.tracker.reset(vehicle_id)

    # -- internals ---------------------------------------------------------

    def _process_impl(
        self,
        frame: np.ndarray,
        vehicle_bbox: BBox,
        vehicle_id: Optional[str],
        camera_id: Optional[str],
        timestamp: Optional[str],
        use_temporal_fusion: bool,
    ) -> Dict[str, Any]:
        if frame is None or frame.size == 0 or bbox_area(vehicle_bbox) <= 0:
            logger.warning("Invalid frame or vehicle_bbox supplied to process()")
            return _empty_result(STATUS_ERROR, vehicle_id, camera_id, timestamp)

        # 1. Plate detection (within the given vehicle region only).
        detections = self.detector.detect(frame, vehicle_bbox)
        if not detections:
            return _empty_result(STATUS_NO_PLATE_DETECTED, vehicle_id, camera_id, timestamp)

        best_detection: PlateDetection = detections[0]

        if vehicle_id:
            smoothed_bbox = self.tracker.update(vehicle_id, best_detection.bbox)
        else:
            smoothed_bbox = best_detection.bbox

        # 2. Capture & crop.
        plate_crop = safe_crop(frame, best_detection.bbox, padding_px=self.config.plate_crop_padding_px)
        if plate_crop is None:
            return _empty_result(
                STATUS_NO_PLATE_DETECTED, vehicle_id, camera_id, timestamp, plate_bbox=best_detection.bbox
            )

        h, w = plate_crop.shape[:2]
        if w < self.config.min_plate_width_px or h < self.config.min_plate_height_px:
            logger.debug("Plate crop too small (%dx%d) for vehicle_id=%s", w, h, vehicle_id)
            return _empty_result(
                STATUS_NO_PLATE_DETECTED, vehicle_id, camera_id, timestamp, plate_bbox=best_detection.bbox
            )

        # 3. Preprocessing (adaptive).
        processed_crop, quality = self.preprocessor.process(plate_crop)

        # 4. OCR.
        ocr_result: Optional[OCRResult] = self.ocr.read(processed_crop)
        if ocr_result is None:
            return _empty_result(
                STATUS_NO_TEXT_READ, vehicle_id, camera_id, timestamp, plate_bbox=best_detection.bbox
            )

        # 5. Validation + character correction.
        validation: ValidationResult = self.validator.validate(ocr_result.clean_text)

        # 6. Confidence scoring (single frame).
        confidence = self._compute_confidence(
            detection_conf=best_detection.confidence,
            ocr_conf=ocr_result.confidence,
            is_valid_format=validation.is_valid,
            image_quality=quality.as_quality_scalar(),
            temporal_consistency=None,
        )

        result_text = validation.corrected_text
        frames_used = 1

        # 7. Optional temporal fusion across frames for the same vehicle.
        if use_temporal_fusion and vehicle_id:
            self.fusion.add_observation(
                vehicle_id,
                FrameObservation(
                    text=validation.corrected_text,
                    ocr_confidence=ocr_result.confidence,
                    detection_confidence=best_detection.confidence,
                    timestamp=time.time(),
                ),
            )
            fused = self.fusion.fuse(vehicle_id)
            if fused is not None and fused.frames_used >= self.config.min_frames_for_fusion:
                fused_validation = self.validator.validate(fused.final_text)
                result_text = fused_validation.corrected_text
                validation = fused_validation
                frames_used = fused.frames_used
                confidence = self._compute_confidence(
                    detection_conf=best_detection.confidence,
                    ocr_conf=ocr_result.confidence,
                    is_valid_format=validation.is_valid,
                    image_quality=quality.as_quality_scalar(),
                    temporal_consistency=fused.temporal_consistency,
                )

        status = self._status_for(confidence, validation.is_valid)

        return {
            "plate_number": result_text,
            "confidence": round(confidence, 3),
            "plate_bbox": smoothed_bbox,
            "vehicle_id": vehicle_id,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "frames_used": frames_used,
            "valid_format": validation.is_valid,
            "status": status,
        }

    def _compute_confidence(
        self,
        detection_conf: float,
        ocr_conf: float,
        is_valid_format: bool,
        image_quality: float,
        temporal_consistency: Optional[float],
    ) -> float:
        cfg = self.config
        format_score = 1.0 if is_valid_format else 0.3

        if temporal_consistency is not None:
            temporal_score = temporal_consistency
        else:
            # No multi-frame data available -- neutral mid score so it
            # neither rewards nor penalizes single-frame reads.
            temporal_score = 0.5

        score = (
            cfg.weight_detection_conf * detection_conf
            + cfg.weight_ocr_conf * ocr_conf
            + cfg.weight_format_valid * format_score
            + cfg.weight_image_quality * image_quality
            + cfg.weight_temporal_consistency * temporal_score
        )
        return max(0.0, min(1.0, score))

    def _status_for(self, confidence: float, is_valid_format: bool) -> str:
        cfg = self.config
        if confidence >= cfg.confidence_validated_threshold and is_valid_format:
            return STATUS_VALIDATED
        if confidence >= cfg.confidence_low_confidence_threshold:
            return STATUS_LOW_CONFIDENCE
        return STATUS_UNRELIABLE
