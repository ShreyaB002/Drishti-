"""
ocr.py
======
OCR reading + text normalization for a preprocessed plate crop.

Supports PaddleOCR (default, generally stronger on number plates) or
EasyOCR as an alternative backend, selected via config.ocr_engine.

Normalization here is intentionally conservative:
  - uppercase, strip whitespace/punctuation
  - collapse multiple text fragments returned by the OCR engine into
    a single line (plates are read left-to-right)
It does NOT attempt character-level O/0, I/1, etc. correction --
that requires knowledge of the expected format and is handled in
validator.py, which has access to the plate-format char masks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import ANPRConfig

logger = logging.getLogger("anpr.ocr")

_ALLOWED_CHARS_RE = re.compile(r"[^A-Z0-9]")


@dataclass
class OCRResult:
    raw_text: str
    clean_text: str
    confidence: float


class BaseOCRBackend:
    def read(self, image: np.ndarray) -> Optional[OCRResult]:
        raise NotImplementedError


class PaddleOCRBackend(BaseOCRBackend):
    def __init__(self, config: ANPRConfig):
        try:
            from paddleocr import PaddleOCR  # noqa: F401 – version check
            import paddleocr as _poc
            _ver = tuple(int(x) for x in getattr(_poc, "__version__", "0.0.0").split(".")[:2])
        except ImportError as exc:
            raise RuntimeError(
                "paddleocr is not installed. Run: pip install paddleocr paddlepaddle"
            ) from exc

        # PaddleOCR 3.x changed the constructor — no positional kwargs accepted.
        if _ver >= (3, 0):
            self._engine = PaddleOCR()  # new API: auto-detects language & device
            self._new_api = True
        else:
            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang=config.ocr_lang,
                use_gpu=config.ocr_use_gpu,
                show_log=False,
            )
            self._new_api = False

    def read(self, image: np.ndarray) -> Optional[OCRResult]:
        try:
            if self._new_api:
                # v3 returns a list of Result objects with .rec_texts / .rec_scores
                results = self._engine.predict(image)
                if not results:
                    return None
                texts, confs = [], []
                for res in results:
                    rec_texts = getattr(res, "rec_texts", None) or []
                    rec_scores = getattr(res, "rec_scores", None) or []
                    texts.extend(rec_texts)
                    confs.extend(rec_scores)
                if not texts:
                    return None
                raw_text = " ".join(str(t) for t in texts)
                avg_conf = sum(float(c) for c in confs) / len(confs)
                return OCRResult(raw_text=raw_text, clean_text=raw_text, confidence=avg_conf)
            else:
                result = self._engine.ocr(image, cls=True)
        except Exception as exc:
            logger.error("PaddleOCR inference failed: %s", exc)
            return None

        if not result or not result[0]:
            return None

        lines = result[0]
        lines.sort(key=lambda ln: min(pt[0] for pt in ln[0]))

        texts = [ln[1][0] for ln in lines]
        confs = [ln[1][1] for ln in lines]
        raw_text = " ".join(texts)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return OCRResult(raw_text=raw_text, clean_text=raw_text, confidence=float(avg_conf))


class EasyOCRBackend(BaseOCRBackend):
    def __init__(self, config: ANPRConfig):
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "easyocr is not installed. Run: pip install easyocr"
            ) from exc

        self._engine = easyocr.Reader([config.ocr_lang], gpu=config.ocr_use_gpu)

    def read(self, image: np.ndarray) -> Optional[OCRResult]:
        try:
            result = self._engine.readtext(image)
        except Exception as exc:
            logger.error("EasyOCR inference failed: %s", exc)
            return None

        if not result:
            return None

        result.sort(key=lambda r: min(pt[0] for pt in r[0]))
        texts = [r[1] for r in result]
        confs = [r[2] for r in result]
        raw_text = " ".join(texts)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return OCRResult(raw_text=raw_text, clean_text=raw_text, confidence=float(avg_conf))


class PlateOCR:
    """Facade selecting the configured OCR backend and normalizing output."""

    def __init__(self, config: ANPRConfig):
        self.config = config
        self._backend = self._build_backend()

    def _build_backend(self) -> BaseOCRBackend:
        engine = self.config.ocr_engine.lower()
        if engine == "paddleocr":
            try:
                return PaddleOCRBackend(self.config)
            except Exception as exc:
                logger.warning("PaddleOCR unavailable (%s), falling back to EasyOCR", exc)
                try:
                    return EasyOCRBackend(self.config)
                except Exception as exc2:
                    logger.error("EasyOCR unavailable (%s)", exc2)
                    raise exc
        if engine == "easyocr":
            return EasyOCRBackend(self.config)
        raise ValueError(f"Unsupported ocr_engine in config: {self.config.ocr_engine!r}")

    @staticmethod
    def normalize_text(raw_text: str) -> str:
        """Uppercase, strip spaces and any non-alphanumeric characters."""
        upper = raw_text.upper()
        return _ALLOWED_CHARS_RE.sub("", upper)

    def read(self, plate_image: np.ndarray) -> Optional[OCRResult]:
        result = self._backend.read(plate_image)
        if result is None:
            return None

        clean = self.normalize_text(result.raw_text)
        if not clean:
            return None

        if result.confidence < self.config.ocr_min_confidence:
            logger.debug(
                "Discarding OCR read %r: confidence %.2f below threshold %.2f",
                clean, result.confidence, self.config.ocr_min_confidence,
            )
            return None

        return OCRResult(raw_text=result.raw_text, clean_text=clean, confidence=result.confidence)
