# Drishti -- Camera Source Configuration
# =====================================================================
# Set live camera URLs here. Set None to automatically use test video.
#
# Supported formats:
#   RTSP:   rtsp://user:pass@192.168.1.100:554/stream
#   HTTP:   http://192.168.1.108:8080/video
#   USB webcam:  0  (integer index, 0 = first camera)
#   Local file:  r"C:\path\to\video.mp4"
# =====================================================================

from pathlib import Path

ROOT     = Path(__file__).resolve().parent
TEST_DIR = ROOT / "test"

# --------------------------------------------------------------------
# LIVE CAMERA SOURCES
# Set to None to use the test video fallback for that camera.
# --------------------------------------------------------------------

CAM1_SOURCE = None   # Cam 1 -- Virtual Fence
# CAM1_SOURCE = "rtsp://admin:password@192.168.1.101:554/stream1"
# CAM1_SOURCE = "http://192.168.1.101:8080/video"

CAM2_SOURCE = None   # Cam 2 -- Vehicle Detection
# CAM2_SOURCE = "rtsp://admin:password@192.168.1.102:554/stream1"

CAM3_SOURCE = None   # Cam 3 -- Human Detection
# CAM3_SOURCE = "rtsp://admin:password@192.168.1.103:554/stream1"

CAM4_SOURCE = None   # Cam 4 -- ANPR (License Plate Recognition)
# CAM4_SOURCE = "rtsp://admin:password@192.168.1.104:554/stream1"

CAM5_SOURCE = None   # Cam 5 -- Suspicious Activity Detection
# CAM5_SOURCE = "rtsp://admin:password@192.168.1.105:554/stream1"

CAM6_SOURCE = 0      # Cam 6 -- Facial Recognition (USB webcam index 0)
# CAM6_SOURCE = "rtsp://admin:password@192.168.1.106:554/stream1"


# --------------------------------------------------------------------
# TEST / FALLBACK VIDEOS
# Used automatically when the live source above is None or fails.
# --------------------------------------------------------------------

CAM1_FALLBACK = TEST_DIR / "virtual_fence.mp4"
CAM2_FALLBACK = TEST_DIR / "vehicle_detection.mp4"
CAM3_FALLBACK = TEST_DIR / "human_deetection.mp4"
_anpr_filename = "Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4"
CAM4_FALLBACK = TEST_DIR / _anpr_filename
CAM5_FALLBACK = TEST_DIR / "suspicious_activity.mp4"
CAM6_FALLBACK = TEST_DIR / "facial_recognition.jpg"  # Static image fallback if webcam unavailable


def _resolve(live, fallback):
    """Return live source if set, otherwise return the fallback path."""
    if live is not None:
        return live
    return fallback
    

# --------------------------------------------------------------------
# SOURCES -- imported by fastapi_dashboard.py
# This is the final dict the dashboard uses. Do not rename the keys.
# --------------------------------------------------------------------

SOURCES = {
    "virtual_fence":       _resolve(CAM1_SOURCE, CAM1_FALLBACK),
    "vehicle_detection":   _resolve(CAM2_SOURCE, CAM2_FALLBACK),
    "human_detection":     _resolve(CAM3_SOURCE, CAM3_FALLBACK),
    "anpr":                _resolve(CAM4_SOURCE, CAM4_FALLBACK),
    "suspicious_activity": _resolve(CAM5_SOURCE, CAM5_FALLBACK),
    "facial_recognition":  _resolve(CAM6_SOURCE, CAM6_FALLBACK),
}
