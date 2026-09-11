# DRISHTI (दृष्टि) — Integrated Border Video Analytics Platform (IBVAP)
## Complete Technical Architecture, Engineering Specifications, and System Manual

---

## Executive Summary

**Drishti (दृष्टि)** is an advanced, edge-deployable, multi-camera surveillance and tactical intelligence system developed for high-security defense perimeters, Border Outposts (BOP), military installations, and critical national infrastructure.

The platform unifies cutting-edge computer vision models, custom spatial multi-object tracking (MOT), optical character recognition (OCR), facial biometrics, and deterministic spatial geometry into a unified command and control system. Drishti processes up to 6 concurrent video feeds in real time, generates structured telemetry and actionable security events, and delivers sub-second incident alerts through a centralized tactical dashboard.

---

## 1. System Architecture & High-Level Data Flow

The Drishti platform is built upon a decoupled, thread-isolated architecture designed for high throughput, zero-latency frame dropping, and robust fault isolation.

```
                                  CAMERA INGESTION LAYER
           [Cam 1: RTSP/MP4]    [Cam 2: RTSP/MP4]    [Cam 3: RTSP/MP4]
           [Cam 4: RTSP/MP4]    [Cam 5: RTSP/MP4]    [Cam 6: USB/Webcam]
                                      │
                                      ▼
                             FRAME BUFFER & WORKERS
                 (Thread-isolated ingestion with FPS regulation)
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
           AI INFERENCE ENGINES               SPATIAL MOT TRACKER
       • YOLOv11n Object Detection         • Dynamic Greedy IoU Matching
       • YOLOv11n-Pose (17 Keypoints)      • Centroid Distance Fallback
       • License Plate Detector (YOLO)     • Non-Maximum Suppression (NMS)
       • EasyOCR / PaddleOCR               • Deterministic ID Recycling
       • Haar + NCC Face Verifier          • Zero-Ghost Lost Track Pruning
                     │                                 │
                     └────────────────┬────────────────┘
                                      │
                                      ▼
                        EVENT & ALERT BROKER (IBVAP)
               • Spatial Polygon Intrusion (Ray Casting / Cross-Product)
               • Event Cooldown & Temporal Deduplication
               • Rolling 500-Event Memory Ring Buffer
               • Live CSV Audit Exporter
                                      │
                                      ▼
                       TACTICAL COMMAND DASHBOARD (FastAPI)
         • Multipart MJPEG Low-Latency Video Streaming
         • Normalized SVG Real-Time Bounding Box & Fence Overlays (16 FPS)
         • Dynamic Threat Breach Banner (Alert Tiering)
         • Telemetry REST API (/api/status, /api/events, /api/export-events)
```

### Key Architectural Principles
1. **Thread Isolation**: Each camera feed runs in an autonomous Python thread. A slow inference cycle or network hitch on one feed cannot block or degrade the frame rate of other feeds.
2. **Lock-Free Concurrency**: Shared frame buffers and detection caches are guarded by lightweight threading locks (`threading.Lock`), ensuring instant read/write cycles without pipeline contention.
3. **Normalized Vector Coordinates**: All bounding boxes and fence coordinates are computed as relative ratios (0.0 to 1.0). This ensures dynamic responsiveness across responsive displays, mobile devices, and enlarged modal inspection windows without pixel stretching or calibration drift.
4. **Pruned Spatial Memory**: Detections are tied strictly to active frame evidence. Inactive/unmatched tracks are aggressively pruned, preventing frozen "ghost boxes" on video displays.

---

## 2. Core Camera Modules & Analytical Engines

### Camera 1: Virtual Fence Intrusion Detection (`virtual_fence_engine.py`)
* **Objective**: Guard physical boundaries and restricted corridors against unauthorized pedestrian crossing.
* **Technology**: YOLOv11 Neural Detector + Custom `VirtualFenceEngine`.
* **Geometry Modes**:
  * **Line Boundary**: Computes the geometric intersection between an object's trajectory vector (P_{t-1} -> P_t) and the virtual fence segment (A -> B) using 2D cross-product orientation:
    ```
    ccw(A, B, C) = (C_y - A_y) * (B_x - A_x) > (B_y - A_y) * (C_x - A_x)
    ```
    An intersection occurs if and only if `ccw(A, B, C) != ccw(A, B, D)` and `ccw(C, D, A) != ccw(C, D, B)`.
  * **Polygon Boundary**: Computes point-in-polygon containment using `cv2.pointPolygonTest`. Grounding is determined at the subject's base (feet):
    ```
    curr_pos = ((x1 + x2) / 2, y2)
    ```
* **False-Positive Mitigation**: The default polygon is aligned to the foreground property boundary (y >= 125), completely excluding background public sidewalks and bike racks. Objects under 35 pixels in height are filtered as optical noise.

---

### Camera 2: Vehicle Detection & Classification (`vehicle_detection.py`)
* **Objective**: Monitor vehicular checkpoints, identify transport types, and calculate approach traffic.
* **Technology**: YOLOv11 multi-class object detection.
* **Classes Detected**: `car`, `truck`, `bus`, `motorcycle`.
* **Processing**: Bounding boxes are filtered by confidence threshold (>= 0.40). Detected vehicles receive unique spatial track IDs and feed downstream ANPR pipelines.

---

### Camera 3: Human Detection & Pose Estimation (`human_detection.py`)
* **Objective**: Detect human presence, track multiple individuals simultaneously, and analyze physical movement states.
* **Technology**: YOLOv11 Detection + YOLOv11-Pose (17 COCO Keypoints).
* **Keypoint Topology**: 17 anatomical joints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
* **Movement Classification**: Analyzes the average Euclidean displacement of visible joints across consecutive frames:
  * Delta d < 4.0 px: **STILL**
  * 4.0 px <= Delta d <= 15.0 px: **SLOW MOTION / WALKING**
  * Delta d > 15.0 px: **FAST MOTION / RUNNING**
* **NMS Filtering**: Integrated Non-Maximum Suppression (IoU threshold 0.35) eliminates stacked/duplicate bounding boxes on the same individual.

---

### Camera 4: Automatic Number Plate Recognition (ANPR) Engine (`anpr_engine/`)
* **Objective**: High-accuracy vehicle license plate localization, optical recognition, and legal syntax validation.
* **Architecture**: 2-Stage Cascaded Neural Pipeline:
  1. **Stage 1 (Vehicle RoI Extraction)**: The vehicle detection engine isolates vehicle bounding boxes.
  2. **Stage 2 (Plate Localization)**: A specialized YOLO detector (`anpr/models/plate_detector.pt`) runs strictly inside the vehicle RoI. Coordinates are mapped back to full-frame space.
  3. **Adaptive Preprocessing (`preprocessor.py`)**: Evaluates crop blurriness (Laplacian variance) and contrast. Conditionally applies:
     * Bicubic super-resolution resizing.
     * Contrast Limited Adaptive Histogram Equalization (CLAHE).
     * Bilateral denoising and edge sharpening.
  4. **Optical Character Recognition (`ocr.py`)**: Runs EasyOCR / PaddleOCR text recognition on the preprocessed crop.
  5. **Syntax Validation & Ambiguity Correction (`validator.py`)**: Validates text against standardized Indian Motor Vehicle registration formats (Standard State Series: `^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$`, Bharat Series: `^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$`). Conducts position-aware optical ambiguity repair (e.g. O <-> 0, I <-> 1, B <-> 8, S <-> 5, Z <-> 2).
  6. **Temporal Fusion (`fusion.py`)**: For continuous video streams, maintains a rolling observation buffer per vehicle ID and executes a position-wise, confidence-weighted majority vote across frames.

---

### Camera 5: Suspicious Activity Detection (Cam 5)
* **Objective**: Nocturnal intrusion detection, stairwell breach monitoring, and loitering identification.
* **Technology**: YOLOv11 person detection optimized for low-light IR imagery.
* **Visual Alerting**: When unauthorized presence is detected in restricted stairwells, the feed generates high-priority telemetry and triggers dynamic red pulsing border highlights on the dashboard.

---

### Camera 6: Facial Biometrics & Verification (Cam 6)
* **Objective**: Checkpoint personnel verification and unauthorized visitor identification.
* **Technology**: Real-time Haar Cascade frontal face localization + Normalized Cross-Correlation (NCC) template embedding comparison against authorized personnel vectors.
* **Decision Threshold**:
  * Similarity >= 0.55: **Facial Recognition Successful** (Green highlight, verified status).
  * Similarity < 0.55: **Face Not Recognised** (Red highlight, security alert).
* **Input Flexibility**: Operates seamlessly with live connected USB webcams (Camera Index 0) or high-resolution photographic reference archives.

---

## 3. Spatial Multi-Object Tracking (MOT) & Stability Engineering

A surveillance platform must maintain stable tracking IDs without identity jumps, count explosions, or frozen ghost boxes. Drishti includes a custom-built `SurveillanceTracker` (`virtual_fence_engine.py`):

### Algorithm Workflow
1. **Input Candidates**: Receives raw candidate bounding boxes from neural detectors.
2. **Non-Maximum Suppression (NMS)**: Pre-filters candidate boxes with an IoU threshold of 0.35. If two boxes overlap substantially on the same individual (e.g., body vs. torso), the lower-confidence box is discarded.
3. **Dual-Metric Cost Assignment**:
   * **Intersection-over-Union (IoU)**: If IoU >= 0.20, the score is the direct IoU value.
   * **Centroid Euclidean Distance**: If IoU fails due to fast motion, fallback scoring evaluates spatial proximity:
     ```
     Score = 1.0 / (1.0 + dist(centroid_A, centroid_B)) for dist <= 80 px
     ```
4. **Greedy Matching**: Assigns unmatched detections to existing active tracks using the highest association score.
5. **Deterministic Low-Integer ID Recycling**: Unmatched new detections receive the smallest available positive integer (#1, #2, #3...) not currently assigned. IDs never explode into arbitrary thousands.
6. **Zero-Ghost Pruning**: Tracks that are not detected in the current frame are marked lost. If not re-acquired within `max_lost_frames = 4`, the track is permanently deleted. **Only actively matched detections are returned to the rendering layer**, guaranteeing that no dead boxes freeze on screen.

---

## 4. Event Broker, Alerting & Audit Logging

### Supported Security Event Types
| Event Type | Trigger Condition | Severity Level | Cooldown |
|---|---|---|---|
| `INTRUSION_DETECTED` | Target crossed virtual line or entered polygon restricted zone | CRITICAL | 1.5s |
| `SUSPICIOUS` | Unauthorized presence in nocturnal zones or unverified face | HIGH | 2.0s |
| `PLATE_DETECTED` | Vehicle license plate successfully read and validated | INFO | 3.0s |
| `HUMAN_DETECTED` | Human presence or authorized face verification | NORMAL | 2.0s |
| `VEHICLE_DETECTED`| Motor vehicle approaching perimeter | NORMAL | 2.0s |
| `ERROR` | Camera connection drop or stream timeout | WARNING | Instant |

### Telemetry & Audit Features
* **In-Memory Ring Buffer**: Stores up to 500 real-time structured event payloads with timestamps, camera origin, confidence score, and contextual details.
* **Audit CSV Export**: Built-in endpoint `/api/export-events` exports RFC-4180-compliant CSV logs formatted for administrative review:
  `ID, Timestamp, Camera, EventType, Description, Confidence, ObjectType, PlateNumber`

---

## 5. Tactical Command Dashboard & UI Specifications

The Drishti Web Dashboard is engineered for mission-critical command centers:

### Design & Typography
* **Typography**: Unified **Times New Roman** serif styling across all headers, status badges, HUD labels, and data tables for formal government reporting aesthetics.
* **National Defense Branding**: Features the National Emblem / Indian Flag badge, product logo, and official metadata.
* **Responsive Video Grid**: 6 synchronized live camera tiles with responsive aspect ratios.
* **Enlarged Inspection Modal**: Clicking any feed enlarges it into high-definition modal view with synchronized SVG overlays and diagnostic telemetry.
* **Dynamic Threat Banner**: Displays a persistent high-visibility banner (`Cam X · Breach Detected`) **strictly when active intrusions or suspicious incidents occur**, automatically retracting when threats clear.

---

## 6. REST API Reference

| Endpoint | Method | Response Type | Description |
|---|---|---|---|
| `/` | `GET` | `text/html` | Serves the single-page tactical command dashboard. |
| `/video_feed/{feed_key}` | `GET` | `multipart/x-mixed-replace` | Low-latency MJPEG video stream for a specific camera. |
| `/api/status` | `GET` | `application/json` | Real-time system telemetry: active feeds, bounding boxes, polygon coords, active breaches. |
| `/api/events` | `GET` | `application/json` | Returns rolling list of recent system security events. |
| `/api/export-events` | `GET` | `text/csv` | Streams downloadable CSV audit log for incident reporting. |
| `/api/config/fence` | `POST` | `application/json` | Dynamically updates the virtual fence boundary coordinates. |

---

## 7. Installation, Setup & Deployment

### Hardware Requirements
* **Processor**: Intel Core i5/i7 (8th Gen+) or AMD Ryzen 5/7 (or ARM-based edge devices like NVIDIA Jetson Orin).
* **RAM**: 8 GB minimum (16 GB recommended for 6 concurrent streams).
* **GPU (Optional)**: NVIDIA GPU with CUDA 11.8/12.X for hardware-accelerated YOLO inference.
* **Camera**: Standard RTSP IP CCTV cameras, USB Webcams, or pre-recorded MP4/AVI tactical streams.

### Software Dependencies
* Python 3.10 or higher
* FastAPI & Uvicorn
* OpenCV (`opencv-python` or `opencv-python-headless`)
* Ultralytics YOLO (`ultralytics`)
* EasyOCR (`easyocr`) or PaddleOCR (`paddleocr`)
* NumPy, Pydantic

### Quickstart Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ShreyaB002/Drishti-.git
   cd Drishti-
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn opencv-python ultralytics numpy easyocr pydantic
   ```

4. **Verify Model Weights**:
   Ensure `yolo11n.pt` is in the project root (downloaded automatically on first execution if absent).

5. **Launch the Command Dashboard**:
   ```bash
   python fastapi_dashboard.py
   ```
   Open your browser and navigate to:
   **`http://localhost:8000`**

---

## 8. Directory & File Structure

```
Drishti/
├── assets/                          # UI Branding, Favicons & Official Badges
│   ├── favion.png
│   ├── product_logo.png
│   └── team_logo.png
├── anpr_engine/                     # Independent ANPR Subsystem
│   ├── README.md                    # ANPR Architecture Manual
│   ├── anpr/
│   │   ├── anpr_engine.py           # Core ANPR Orchestrator
│   │   ├── config.py                # Regex rules, weights, state formats
│   │   ├── detector.py              # YOLO Plate Localizer
│   │   ├── fusion.py                # Multi-frame Temporal Fusion
│   │   ├── ocr.py                   # EasyOCR / PaddleOCR wrapper
│   │   ├── preprocessor.py          # Adaptive CLAHE/Bilateral Preprocessor
│   │   ├── tracker.py               # Kalman Filter BBox Smoother
│   │   └── validator.py             # Position-aware Syntax Validator
│   └── examples/
├── test/                            # Test Feeds & Reference Data
│   ├── facial_recognition.jpg       # Authorized Personnel Face Reference
│   ├── virtual_fence.mp4
│   ├── human_deetection.mp4
│   ├── suspicious_activity.mp4
│   └── vehicle_detection.mp4
├── templates/                       # Jinja2 / HTML UI templates
│   └── index.html
├── virtual_fence_engine.py          # Standalone Virtual Fence Engine & MOT Tracker
├── fastapi_dashboard.py             # Production 6-Camera Tactical Server & Web UI
├── fastapi_app.py                   # Lightweight Single-Feed Virtual Fence Service
├── human_detection.py               # Standalone Pose Estimation & Motion Tracker
├── vehicle_detection.py             # Standalone Vehicle Classifier
├── .gitignore                       # Repository exclusion rules
├── README.md                        # Project landing document
└── DOCUMENTATION.md                 # Complete System Architecture & Manual
```

---

## 9. Verification & Quality Assurance

* **Tracking Reliability**: Confirmed 0 duplicate stacked IDs in multi-person scenarios through pre-tracking NMS (`iou_thresh=0.35`).
* **Intrusion Precision**: Grounding polygon evaluates feet location (x_mid, y2), preventing overhead perspective errors.
* **Fault Recovery**: Auto-reconnects on stream drops and automatically re-initializes trackers on loop boundaries without leaking memory.
