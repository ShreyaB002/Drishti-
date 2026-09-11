# DRISHTI (दृष्टि) — Integrated Border Video Analytics Platform (IBVAP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-orange.svg)](https://docs.ultralytics.com/)
[![License: Defense / Smart India Hackathon](https://img.shields.io/badge/License-Defense%20%2F%20SIH-red.svg)](#)

**Drishti (दृष्टि)** is a unified multi-camera tactical CCTV surveillance and perimeter defense platform developed for border outposts (BOP), military installations, and critical national infrastructure.

---

## Key Capabilities

1. **Cam 1 · Virtual Fence & Restricted Area Intrusion**: Real-time polygon/tripwire boundary monitoring with zero-ghost dynamic tracking.
2. **Cam 2 · Vehicle Detection & Traffic Classification**: Multi-class vehicular recognition (cars, trucks, buses, motorcycles).
3. **Cam 3 · Human Detection & Movement Analysis**: Multi-person tracking with Non-Maximum Suppression (NMS) and 17-keypoint pose estimation.
4. **Cam 4 · Automatic Number Plate Recognition (ANPR)**: Cascaded vehicle RoI extraction, license plate localization, adaptive preprocessing, OCR, and Indian vehicle registration syntax validation with multi-frame temporal fusion.
5. **Cam 5 · Suspicious Activity Monitoring**: Low-light infrared restricted zone monitoring with dynamic red pulse threat alerting.
6. **Cam 6 · Facial Recognition & Verification**: Real-time webcam face localization and similarity verification against authorized personnel databases.
7. **Tactical Command Dashboard**: Real-time 6-feed video grid, sub-second threat breach banner, interactive camera inspection modal, and downloadable RFC-4180 audit CSV logs.

---

## Comprehensive Technical Documentation

For the complete technical whitepaper, system architecture, geometric algorithms, multi-object tracking (MOT) specifications, API documentation, and deployment guidelines, please refer to:

👉 **[DOCUMENTATION.md](DOCUMENTATION.md)**

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ShreyaB002/Drishti-.git
cd Drishti-

# 2. Setup virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install fastapi uvicorn opencv-python ultralytics numpy easyocr pydantic

# 4. Run tactical dashboard
python fastapi_dashboard.py
```

Access the command center at: **`http://localhost:8000`**
