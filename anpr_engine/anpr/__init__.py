"""
ANPR (Automatic Number Plate Recognition) Engine
==================================================

Independent module of the Drishti / IBVAP system.

Scope
-----
This module is responsible ONLY for:
    - Detecting a number plate inside a vehicle bounding box supplied
      by the (external) Vehicle Detection Engine
    - Cropping / normalizing the plate image
    - Running OCR on the plate
    - Validating the OCR text against configurable Indian plate formats
    - Producing a confidence-scored, structured result
    - (Optionally) fusing multiple frame observations for video streams

This module explicitly does NOT perform vehicle detection or
vehicle classification. That is owned by the Vehicle Detection Engine.

Public API
----------
    from anpr import ANPREngine, ANPRConfig

    engine = ANPREngine()
    result = engine.process(frame, vehicle_bbox, vehicle_id, camera_id, timestamp)
"""

from .anpr_engine import ANPREngine
from .config import ANPRConfig

__all__ = ["ANPREngine", "ANPRConfig"]
__version__ = "1.0.0"
