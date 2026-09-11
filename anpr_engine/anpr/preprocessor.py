"""
preprocessor.py
================
Adaptive preprocessing of a cropped plate image, applied BEFORE OCR.

Design goal: don't blindly run every filter on every crop. Instead,
inspect basic quality signals (size, blur, contrast) and only apply
the corrective operations that are actually needed. Over-processing a
clean, sharp plate image (e.g. unnecessary sharpening + threshold)
frequently *hurts* OCR accuracy, so each step is conditional.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .config import ANPRConfig
from .utils.image_utils import laplacian_blur_score, contrast_std, to_gray

logger = logging.getLogger("anpr.preprocessor")


@dataclass
class QualityReport:
    width: int
    height: int
    blur_score: float
    contrast: float
    is_blurry: bool
    is_low_contrast: bool
    is_too_small: bool

    def as_quality_scalar(self) -> float:
        """
        Collapse the quality signals into a single 0..1 score used later
        as one of the inputs to overall confidence scoring.
        """
        score = 1.0
        if self.is_too_small:
            score -= 0.4
        if self.is_blurry:
            score -= 0.3
        if self.is_low_contrast:
            score -= 0.2
        return max(0.0, min(1.0, score))


class PlatePreprocessor:
    def __init__(self, config: ANPRConfig):
        self.config = config

    def assess_quality(self, plate_image: np.ndarray) -> QualityReport:
        gray = to_gray(plate_image)
        h, w = gray.shape[:2]
        blur = laplacian_blur_score(gray)
        contrast = contrast_std(gray)

        return QualityReport(
            width=w,
            height=h,
            blur_score=blur,
            contrast=contrast,
            is_blurry=blur < self.config.blur_variance_threshold,
            is_low_contrast=contrast < self.config.low_contrast_std_threshold,
            is_too_small=(
                w < self.config.min_plate_width_px or h < self.config.min_plate_height_px
            ),
        )

    def process(self, plate_image: np.ndarray) -> tuple[np.ndarray, QualityReport]:
        """
        Returns (preprocessed_image_bgr, quality_report).

        The output is still a 3-channel BGR image (most OCR engines,
        including PaddleOCR/EasyOCR, expect BGR/RGB input rather than
        a binarized single-channel image -- aggressive thresholding is
        applied only as a last resort for very poor quality crops).
        """
        quality = self.assess_quality(plate_image)
        image = plate_image.copy()

        # 1. Resize small plates up to a workable height, preserving aspect ratio.
        if quality.height < self.config.ocr_target_height_px:
            scale = self.config.ocr_target_height_px / max(1, quality.height)
            new_w = max(1, int(quality.width * scale))
            image = cv2.resize(
                image, (new_w, self.config.ocr_target_height_px),
                interpolation=cv2.INTER_CUBIC,
            )

        # 2. Denoise only if the image looks noisy/blurry (skip on clean crops
        #    to avoid softening genuinely sharp edges further).
        if quality.is_blurry:
            image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
            # Mild unsharp-mask style sharpening after denoise.
            gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
            image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

        # 3. Contrast enhancement only if contrast is genuinely low.
        if quality.is_low_contrast:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            lab = cv2.merge((l_channel, a_channel, b_channel))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Quality is (re)computed once up front; we intentionally do NOT
        # recompute post-processing quality here, since the report is
        # meant to reflect the *original* capture conditions for
        # confidence scoring, not the enhanced output.
        return image, quality
