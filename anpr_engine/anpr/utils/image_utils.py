"""
utils/image_utils.py
=====================
Small, dependency-light image helpers shared by detector/preprocessor.
Kept separate from business logic so they're easy to unit test.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import cv2

logger = logging.getLogger("anpr.image_utils")

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


def clip_bbox(bbox: BBox, frame_width: int, frame_height: int) -> BBox:
    """Clamp a bounding box so it lies fully inside the frame."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), frame_width - 1))
    y1 = max(0, min(int(y1), frame_height - 1))
    x2 = max(0, min(int(x2), frame_width))
    y2 = max(0, min(int(y2), frame_height))
    if x2 <= x1:
        x2 = min(frame_width, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_height, y1 + 1)
    return x1, y1, x2, y2


def expand_bbox(bbox: BBox, margin_fraction: float, frame_width: int, frame_height: int) -> BBox:
    """Expand a bbox outward by a fraction of its own width/height, then clip."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    dx, dy = int(w * margin_fraction), int(h * margin_fraction)
    return clip_bbox((x1 - dx, y1 - dy, x2 + dx, y2 + dy), frame_width, frame_height)


def safe_crop(frame: np.ndarray, bbox: BBox, padding_px: int = 0) -> np.ndarray | None:
    """
    Crop `bbox` from `frame`, clipping to image bounds and applying
    optional padding. Returns None if the resulting crop is empty.
    """
    if frame is None or frame.size == 0:
        logger.warning("safe_crop called with an empty frame")
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 -= padding_px
    y1 -= padding_px
    x2 += padding_px
    y2 += padding_px

    # Reject boxes that don't actually overlap the frame at all, before
    # clipping -- clip_bbox guarantees a minimum 1px box which would
    # otherwise turn a fully out-of-bounds bbox into a spurious crop.
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h or x2 <= x1 or y2 <= y1:
        logger.debug("safe_crop bbox=%s does not overlap frame (%dx%d)", bbox, w, h)
        return None

    x1, y1, x2, y2 = clip_bbox((x1, y1, x2, y2), w, h)

    if x2 - x1 <= 0 or y2 - y1 <= 0:
        logger.debug("safe_crop produced a degenerate region for bbox=%s", bbox)
        return None

    return frame[y1:y2, x1:x2].copy()


def bbox_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def translate_bbox(inner_bbox: BBox, origin: Tuple[int, int]) -> BBox:
    """Convert a bbox that is local to a sub-crop into full-frame coordinates."""
    ox, oy = origin
    x1, y1, x2, y2 = inner_bbox
    return x1 + ox, y1 + oy, x2 + ox, y2 + oy


def laplacian_blur_score(gray_image: np.ndarray) -> float:
    """Higher = sharper. Common cheap blur metric (variance of Laplacian)."""
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def contrast_std(gray_image: np.ndarray) -> float:
    """Standard deviation of pixel intensities; low value == low contrast."""
    return float(gray_image.std())


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
