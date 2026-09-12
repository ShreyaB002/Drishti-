"""
config.py
=========
Central, editable configuration for the ANPR engine.

Keeping every tunable value here (instead of scattered across modules)
makes the engine easy to retune per-deployment (different camera
quality, different states, stricter/looser thresholds) without
touching pipeline code.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PlateFormat:
    """
    A single configurable Indian number-plate format definition.

    `pattern` is a regex applied to the *cleaned* (uppercase,
    no-space) OCR string.

    `char_mask` is a position-by-position description of what kind of
    character is expected at each index: 'A' = alphabet, 'N' = digit.
    It is used by the validator/OCR-correction step to decide whether
    an ambiguous character (e.g. 'O' vs '0') should be coerced, based
    on *where* it sits in the plate -- never guessed blindly.

    Standard Indian format: SS DD LL(1-3) NNNN
        SS   -> 2-letter state code            (A)
        DD   -> 2-digit RTO code                (N)
        LL   -> 1-3 letter series                (A)
        NNNN -> 4-digit unique number            (N)
    """
    name: str
    pattern: str
    char_mask: str  # e.g. "AANNAANNNN" for the common case


# Registry of supported plate formats. Add more entries here (e.g. BH
# series, defence vehicles, new-format plates) without touching any
# other module.
DEFAULT_PLATE_FORMATS: List[PlateFormat] = [
    # Standard: MH12AB1234 / DL01CA1234 (2 letter series)
    PlateFormat(
        name="IN_STANDARD_2LETTER",
        pattern=r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$",
        char_mask="AANNAANNNN",
    ),
    # Standard with single letter series: KA05M1234
    PlateFormat(
        name="IN_STANDARD_1LETTER",
        pattern=r"^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$",
        char_mask="AANNANNNN",
    ),
    # Standard with 3 letter series: MH12ABC1234
    PlateFormat(
        name="IN_STANDARD_3LETTER",
        pattern=r"^[A-Z]{2}[0-9]{2}[A-Z]{3}[0-9]{4}$",
        char_mask="AANNAAANNNN",
    ),
    # BH-series (Bharat series): 22BH1234AB
    PlateFormat(
        name="IN_BH_SERIES",
        pattern=r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$",
        char_mask="NNAANNNNAA",
    ),
]

# Recognized Indian state/UT codes. Used as an optional, soft signal
# (not a hard rejection) during confidence scoring -- new/rare codes
# should not cause an otherwise well-formed plate to be discarded.
INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JK", "JH",
    "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "OR", "PB",
    "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "UA", "WB", "AN", "CH",
    "DD", "DL", "DN", "LD", "PY", "LA",
}

# Characters that OCR engines commonly confuse. Mapping is used only
# to *propose* a correction; it is applied conditionally based on the
# expected char_mask at that position (see validator.correct_plate).
OCR_CONFUSION_MAP: Dict[str, str] = {
    "O": "0", "0": "O",
    "I": "1", "1": "I",
    "B": "8", "8": "B",
    "S": "5", "5": "S",
    "Z": "2", "2": "Z",
    "G": "6", "6": "G",
    "Q": "0",
    "D": "0",
}


@dataclass
class ANPRConfig:
    # ---- Plate detection ----
    plate_detector_weights: str = "anpr/models/plate_detector.pt"
    plate_detection_conf_threshold: float = 0.35
    plate_detection_iou_threshold: float = 0.45
    # Margin (fraction of vehicle bbox size) added around the vehicle
    # crop before running plate detection, to avoid clipping plates
    # that sit right at the edge of a loose vehicle box.
    vehicle_crop_margin: float = 0.05
    max_plates_per_vehicle: int = 1  # keep the best-scoring plate

    # ---- Plate capture / crop ----
    min_plate_width_px: int = 40
    min_plate_height_px: int = 12
    plate_crop_padding_px: int = 3

    # ---- Preprocessing ----
    ocr_target_height_px: int = 64  # upscale small plates to this height
    blur_variance_threshold: float = 60.0  # below this -> treat as blurry
    low_contrast_std_threshold: float = 35.0  # below this -> boost contrast

    # ---- OCR ----
    ocr_engine: str = "paddleocr"  # "paddleocr" | "easyocr"
    ocr_lang: str = "en"
    ocr_min_confidence: float = 0.3  # below this, discard the read entirely
    ocr_use_gpu: bool = True

    # ---- Validation ----
    plate_formats: List[PlateFormat] = field(
        default_factory=lambda: list(DEFAULT_PLATE_FORMATS)
    )
    allow_unvalidated_output: bool = True  # emit best-effort even if format fails

    # ---- Confidence fusion weights (must sum to ~1.0) ----
    weight_detection_conf: float = 0.20
    weight_ocr_conf: float = 0.35
    weight_format_valid: float = 0.20
    weight_image_quality: float = 0.10
    weight_temporal_consistency: float = 0.15

    # ---- Temporal fusion (video) ----
    min_frames_for_fusion: int = 2
    max_frame_buffer_per_vehicle: int = 15
    track_expiry_seconds: float = 3.0
    fusion_min_char_agreement: float = 0.5  # fraction of votes needed to accept a char

    # ---- Output status thresholds ----
    confidence_validated_threshold: float = 0.75
    confidence_low_confidence_threshold: float = 0.45
    # below low_confidence_threshold -> status = "UNRELIABLE"

    # ---- Logging ----
    log_level: str = "INFO"
