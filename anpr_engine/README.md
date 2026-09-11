# ANPR Engine (Drishti / IBVAP)

An independent **Automatic Number Plate Recognition** module.

**Architecture boundary:** this engine performs *only* plate detection,
reading, and validation. Vehicle detection/classification is owned by
the separate **Vehicle Detection Engine**, which feeds this module a
frame + vehicle bounding box.

```
Vehicle Detection Engine                     ANPR Engine
──────────────────────────                   ──────────────────────────────────
frame, vehicle_bbox, vehicle_id,     ──────▶  Plate Detection (inside vehicle_bbox)
camera_id, timestamp                          → Crop → Preprocess → OCR
                                               → Validate + Confidence
                                               → (video) Temporal Fusion
                                               → Structured result ──▶ Dashboard
```

---

## 1. How each module works

| Module | Responsibility |
|---|---|
| `config.py` | All tunables in one place: thresholds, plate formats, confidence weights. Add new state formats here without touching logic. |
| `detector.py` | `PlateDetector` — runs a YOLO model **inside** the given `vehicle_bbox` only, never on the full frame, never classifying the vehicle. Returns plate bbox(es) + confidence, translated back to full-frame coordinates. |
| `utils/image_utils.py` | Bbox clipping/expansion, safe cropping (handles out-of-bounds boxes, degenerate/empty crops), blur/contrast metrics. |
| `preprocessor.py` | `PlatePreprocessor` — inspects the crop's size/blur/contrast and *conditionally* applies resize, denoise+sharpen, or CLAHE contrast boost. Never runs every filter blindly. |
| `ocr.py` | `PlateOCR` — wraps PaddleOCR or EasyOCR, normalizes raw text (uppercase, strip non-alphanumerics), discards low-confidence reads. No character guessing happens here. |
| `validator.py` | `PlateValidator` — regex-based validation against configurable `PlateFormat` entries (2/3-letter series, BH-series, extensible). Performs *position-aware* O/0, I/1, B/8, S/5, Z/2-style correction only where the format's character mask says a swap makes sense — unresolved characters become `?`, never a guess. |
| `fusion.py` | `TemporalFusion` — for video, buffers per-`vehicle_id` OCR observations and produces a confidence-weighted, position-wise majority vote across frames. Low-agreement positions stay `?`. |
| `tracker.py` | `PlateTracker` — a Kalman filter that smooths/predicts the **plate bounding box location** across frames. It never touches plate text/characters. |
| `anpr_engine.py` | `ANPREngine` — orchestrates the full pipeline and computes the final blended confidence score and status. This is the class the rest of IBVAP calls. |

---

## 2. Models / libraries used

- **Plate detection:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (v8/11-style `.pt` weights), trained on a single `license-plate` class. You must supply/train these weights — see `anpr/models/README.txt`.
- **OCR:** [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) by default (`config.ocr_engine = "paddleocr"`); [EasyOCR](https://github.com/JaidedAI/EasyOCR) is supported as a drop-in alternative.
- **Preprocessing / cropping:** OpenCV (`opencv-python`) + NumPy.
- **Tracking:** OpenCV's `cv2.KalmanFilter` (bbox smoothing only).

---

## 3. Installing dependencies

```bash
pip install -r anpr/requirements.txt
```

This installs `ultralytics`, `paddleocr`/`paddlepaddle`, `opencv-python-headless`, `numpy`, and `pytest`. If you prefer EasyOCR, install it separately (`pip install easyocr`) and set `ocr_engine="easyocr"` in `ANPRConfig`.

Place your trained plate-detection weights at `anpr/models/plate_detector.pt` (or point `config.plate_detector_weights` elsewhere).

---

## 4. Running on a single image

```bash
python examples/run_on_image.py path/to/image.jpg 120 80 640 480
```

(the four numbers are the vehicle bbox `x1 y1 x2 y2`, standing in for what the Vehicle Detection Engine would normally supply.)

Programmatically:

```python
from anpr import ANPREngine
import cv2

engine = ANPREngine()
frame = cv2.imread("car.jpg")

result = engine.process(
    frame=frame,
    vehicle_bbox=(120, 80, 640, 480),
    vehicle_id="V_042",
    camera_id="CAM_07",
    timestamp="2026-09-10T10:15:00+05:30",
)
print(result)
```

---

## 5. Running on a video

```bash
python examples/run_on_video.py path/to/video.mp4
```

Programmatically, call `engine.process(..., use_temporal_fusion=True)` once per frame using the **same `vehicle_id`** for the same physical vehicle (as tracked by the Vehicle Detection Engine), then fetch the consolidated reading:

```python
for frame, vehicle_bbox in video_frames_with_bboxes:
    engine.process(frame, vehicle_bbox, vehicle_id="V_042",
                    camera_id="CAM_07", timestamp=ts, use_temporal_fusion=True)

final = engine.get_fused_result("V_042")   # call once the vehicle exits the scene
```

Call `engine.reset_vehicle_track("V_042")` once a vehicle leaves the scene to free its buffer/tracker state.

---

## 6. How the Vehicle Detection Engine passes data in

Per frame, for each tracked vehicle, the Vehicle Detection Engine calls:

```python
anpr_engine.process(
    frame=current_frame,          # full video frame (numpy BGR array)
    vehicle_bbox=(x1, y1, x2, y2),# vehicle's bounding box in that frame
    vehicle_id="V_042",           # optional, but required for temporal fusion
    camera_id="CAM_07",
    timestamp="2026-09-10T10:15:00+05:30",
    use_temporal_fusion=True,     # True for video streams
)
```

The ANPR engine never re-detects or classifies the vehicle itself — it trusts `vehicle_bbox` as given and only searches for a plate within (a small margin around) that region.

---

## 7. Sending results to the IBVAP dashboard

`ANPREngine.process()` / `get_fused_result()` return a plain JSON-serializable dict:

```json
{
    "plate_number": "MH12AB1234",
    "confidence": 0.93,
    "plate_bbox": [412, 511, 560, 545],
    "vehicle_id": "V_042",
    "camera_id": "CAM_07",
    "timestamp": "2026-09-10T10:15:00+05:30",
    "frames_used": 5,
    "valid_format": true,
    "status": "VALIDATED"
}
```

`status` is one of: `VALIDATED`, `LOW_CONFIDENCE`, `UNRELIABLE`, `NO_PLATE_DETECTED`, `NO_TEXT_READ`, `ERROR`. The IBVAP dashboard/service layer can consume this dict directly — e.g. publish it to a message queue, write it to the event/database layer, or push it over a websocket to the UI. This module does not implement transport itself, keeping it decoupled from IBVAP's messaging choices; wrap the returned dict in whatever the dashboard's ingestion contract expects (REST call, Kafka/Redis message, DB insert, etc.).

---

## Running tests

```bash
python -m pytest anpr/tests/test_anpr.py -v
```

These cover `validator`, `fusion`, `image_utils`, and `preprocessor` with no model weights required (26 tests, all passing). Detector/OCR integration tests require real weights and an installed OCR backend and are intentionally left for you to add once you have your trained `plate_detector.pt`.

---

## Design notes / guardrails baked in

- **No character hallucination**: both `validator.py` (single frame) and `fusion.py` (multi-frame) leave `?` at any character position they can't resolve with confidence, rather than guessing (`MH12A?79` instead of a fabricated `MH12AB79`).
- **Kalman filter scope**: `tracker.py` only smooths the plate's spatial bounding box across frames — it is never used to infer plate characters.
- **Configurable formats**: adding a new state/format is a one-line addition to `DEFAULT_PLATE_FORMATS` in `config.py`; no other file needs to change.
- **Adaptive preprocessing**: filters in `preprocessor.py` are applied conditionally based on measured blur/contrast/size, not unconditionally.
