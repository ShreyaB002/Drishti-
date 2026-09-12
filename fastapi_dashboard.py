import base64
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE.startswith("cuda")
print(f"[Drishti Engine] Active Compute Device: {DEVICE} (FP16 Half-Precision: {USE_HALF})")

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "test"
ASSETS_DIR = ROOT / "assets"

FAVICON_DATA_URI = ""
_fav_file = ASSETS_DIR / "favion.png"
if _fav_file.exists():
    try:
        _b64 = base64.b64encode(_fav_file.read_bytes()).decode("ascii")
        FAVICON_DATA_URI = f"data:image/png;base64,{_b64}"
    except Exception:
        pass

ANPR_ROOT = ROOT / "anpr_engine"
sys.path.insert(0, str(ANPR_ROOT))

from virtual_fence_engine import VirtualFenceEngine, SurveillanceTracker

try:
    from anpr import ANPREngine
except Exception:
    ANPREngine = None

app = FastAPI(title="Drishti Monitoring MVP")

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

from config import SOURCES

frames = {name: None for name in SOURCES}
locks = {name: threading.Lock() for name in SOURCES}
statuses = {name: {"running": False, "error": None} for name in SOURCES}
events = deque(maxlen=200)
status_lock = threading.Lock()
last_event = {}
stop_event = threading.Event()

latest_detections = {name: [] for name in SOURCES}
detections_lock = threading.Lock()

CAM_NAMES = {
    "virtual_fence": "Cam 1",
    "vehicle_detection": "Cam 2",
    "human_detection": "Cam 3",
    "anpr": "Cam 4",
    "suspicious_activity": "Cam 5",
    "facial_recognition": "Cam 6",
}

# ── Event metadata tables ────────────────────────────────────────────────────
EVENT_DISPLAY = {
    "HUMAN_DETECTED":    "Person Detected",
    "VEHICLE_DETECTED":  "Vehicle Detected",
    "PLATE_DETECTED":    "ANPR Detected",
    "INTRUSION_DETECTED":"Fence Breach",
    "SUSPICIOUS":        "Suspicious Activity",
    "ERROR":             "Camera/System Alert",
}
EVENT_SEVERITY = {
    "HUMAN_DETECTED":    "Low",
    "VEHICLE_DETECTED":  "Low",
    "PLATE_DETECTED":    "Medium",
    "INTRUSION_DETECTED":"High",
    "SUSPICIOUS":        "High",
    "ERROR":             "High",
}
EVENT_OBJECT = {
    "HUMAN_DETECTED":    "Person",
    "VEHICLE_DETECTED":  "Vehicle",
    "PLATE_DETECTED":    "Vehicle",
    "INTRUSION_DETECTED":"Person",
    "SUSPICIOUS":        "Person",
    "ERROR":             None,
}


def get_snapshot(feed):
    """Capture a 192×108 JPEG thumbnail of the current frame as a base64 string."""
    with locks[feed]:
        data = frames[feed]
    if data is None:
        return None
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        scale = min(192 / w, 108 / h)
        thumb = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, enc = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 65])
        return base64.b64encode(enc.tobytes()).decode()
    except Exception:
        return None


def event(feed, event_type, message, details=None, cooldown=1.5,
          object_type=None, confidence=None, plate_number=None):
    cam_label = CAM_NAMES.get(feed, feed)
    key = (cam_label, event_type)        # key no longer includes message so edits
    now = time.time()                    # don't leak duplicate-suppression state
    if now - last_event.get(key, 0) < cooldown:
        return
    last_event[key] = now

    obj = object_type or EVENT_OBJECT.get(event_type)
    severity = EVENT_SEVERITY.get(event_type, "Low")
    display = EVENT_DISPLAY.get(event_type, event_type)
    snapshot = get_snapshot(feed)

    item = {
        "time": time.strftime("%H:%M:%S"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feed": cam_label,
        "event_type": display,
        "raw_type": event_type,
        "object": obj,
        "plate_number": plate_number,
        "confidence": round(confidence, 2) if confidence is not None else None,
        "severity": severity,
        "status": "Active",
        "snapshot": snapshot,
        "message": message,
        "details": details or {},
    }
    with status_lock:
        events.appendleft(item)


def box(frame, coords, color=(0, 255, 0), thickness=1):
    x1, y1, x2, y2 = map(int, coords)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def vehicles(frame, model, tracker, feed):
    output = frame.copy()
    h, w = frame.shape[:2]
    dets = []
    try:
        results = model(frame, conf=0.50, device=DEVICE, half=USE_HALF, verbose=False)[0]
    except Exception:
        results = None

    raw = []
    if results is not None and results.boxes is not None:
        for b in results.boxes:
            name = results.names[int(b.cls[0])]
            if name.lower() not in {"car", "truck", "bus", "motorcycle"}:
                continue
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            # Filter out road lane markings, stripes, and tiny artifacts
            if bw < 18 or bh < 14 or (bh / bw) < 0.30:
                continue
            raw.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "name": name.capitalize()
            })

    tracked = tracker.update(raw) if tracker is not None else []
    max_conf = 0.0

    for d in tracked:
        x1, y1, x2, y2 = d["bbox"]
        tid = d["track_id"]
        vname = d.get("name", "Vehicle")
        conf = d.get("conf", 0.75)
        max_conf = max(max_conf, conf)
        dets.append({
            "x1": round(x1 / w, 4),
            "y1": round(y1 / h, 4),
            "w": round((x2 - x1) / w, 4),
            "h": round((y2 - y1) / h, 4),
            "label": f"{vname} #{tid}",
            "type": "vehicle",
            "color": "#f59e0b"
        })

    if tracked:
        event(feed, "VEHICLE_DETECTED", f"{len(tracked)} vehicle(s) detected",
              {"count": len(tracked)}, confidence=max_conf, object_type="Vehicle")

    with detections_lock:
        latest_detections[feed] = dets
    return output


def humans(frame, model, tracker, feed):
    output = frame.copy()
    h, w = frame.shape[:2]
    dets = []
    try:
        results = model(frame, conf=0.22, classes=[0], device=DEVICE, half=USE_HALF, verbose=False)[0]
    except Exception:
        results = None

    raw = []
    if results is not None and results.boxes is not None:
        for b in results.boxes:
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            # Min 18px height for 240p video (was 25px — too aggressive)
            if (y2 - y1) >= 18:
                raw.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf
                })

    tracked = tracker.update(raw) if tracker is not None else []
    max_conf = 0.0

    for d in tracked:
        x1, y1, x2, y2 = d["bbox"]
        tid = d["track_id"]
        conf = d.get("conf", 0.8)
        max_conf = max(max_conf, conf)
        dets.append({
            "x1": round(x1 / w, 4),
            "y1": round(y1 / h, 4),
            "w": round((x2 - x1) / w, 4),
            "h": round((y2 - y1) / h, 4),
            "label": f"Person #{tid}",
            "type": "human",
            "color": "#10b981"
        })

    if tracked:
        event(feed, "HUMAN_DETECTED", f"{len(tracked)} person(s) detected",
              {"count": len(tracked)}, confidence=max_conf, object_type="Person", cooldown=2.0)

    with detections_lock:
        latest_detections[feed] = dets
    return output


def fence(frame, engine, feed):
    output, alerts, detections = engine.process_frame(frame)
    h, w = frame.shape[:2]
    dets = []
    for alert in alerts:
        event(feed, "INTRUSION_DETECTED",
              f"Person crossed restricted zone (track #{alert['track_id']})",
              details=alert, confidence=float(alert.get("confidence", 0.8)),
              object_type="Person", cooldown=1.5)
    
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        is_intruder = d["is_intruder"]
        color = "#ef4444" if is_intruder else "#10b981"
        label = f"INTRUDER #{d['track_id']}" if is_intruder else f"Person #{d['track_id']}"
        dets.append({
            "x1": round(x1 / w, 4),
            "y1": round(y1 / h, 4),
            "w": round((x2 - x1) / w, 4),
            "h": round((y2 - y1) / h, 4),
            "label": label,
            "type": "intruder" if is_intruder else "human",
            "color": color
        })
    with detections_lock:
        latest_detections[feed] = dets
    return output


def anpr(frame, vehicle_model, plate_engine, feed, frame_idx=0, plate_cache=None):
    output = frame.copy()
    h, w = frame.shape[:2]
    dets = []
    if plate_cache is None:
        plate_cache = {}

    try:
        results = vehicle_model.track(frame, conf=0.4, persist=True, device=DEVICE, half=USE_HALF, verbose=False)
    except Exception:
        results = vehicle_model(frame, conf=0.4, device=DEVICE, half=USE_HALF, verbose=False)

    for result in results:
        if result.boxes is None:
            continue
        for detection in result.boxes:
            name = result.names[int(detection.cls[0])]
            if name.lower() not in {"car", "truck", "bus", "motorcycle"}:
                continue
            vx1, vy1, vx2, vy2 = map(int, detection.xyxy[0])
            vehicle_box = (vx1, vy1, vx2, vy2)
            tid = None
            if detection.id is not None:
                try:
                    tid = int(detection.id[0])
                except Exception:
                    pass
            vehicle_id = f"{feed}_track_{tid}" if tid is not None else f"{feed}_{vx1}_{vy1}"

            dets.append({
                "x1": round(vx1 / w, 4),
                "y1": round(vy1 / h, 4),
                "w": round((vx2 - vx1) / w, 4),
                "h": round((vy2 - vy1) / h, 4),
                "label": f"{name} {float(detection.conf[0]):.2f}",
                "type": "vehicle",
                "color": "#f59e0b"
            })

            # High-performance plate detection: run OCR at intervals until high confidence achieved
            cached = plate_cache.get(vehicle_id)
            need_ocr = (cached is None or cached.get("confidence", 0) < 0.65) and (frame_idx % 4 == 0)

            if need_ocr and plate_engine is not None:
                try:
                    anpr_res = plate_engine.process(
                        frame,
                        vehicle_box,
                        vehicle_id=vehicle_id,
                        camera_id=feed,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        use_temporal_fusion=True,
                    )
                    plate = anpr_res.get("plate_number")
                    if plate:
                        plate_cache[vehicle_id] = anpr_res
                        event(feed, "PLATE_DETECTED", f"Plate detected: {plate}",
                              anpr_res, plate_number=plate,
                              confidence=anpr_res.get("confidence"),
                              object_type="Vehicle")
                except Exception:
                    pass

            active_plate = plate_cache.get(vehicle_id)
            if active_plate and active_plate.get("plate_number"):
                plate = active_plate["plate_number"]
                pb = active_plate.get("plate_bbox") or vehicle_box
                px1, py1, px2, py2 = pb
                dets.append({
                    "x1": round(px1 / w, 4),
                    "y1": round(py1 / h, 4),
                    "w": round((px2 - px1) / w, 4),
                    "h": round((py2 - py1) / h, 4),
                    "label": f"PLATE: {plate}",
                    "type": "plate",
                    "color": "#06b6d4"
                })

    with detections_lock:
        latest_detections[feed] = dets
    return output


AUTHORIZED_FACE_VECTORS = []

def init_face_recognition():
    global AUTHORIZED_FACE_VECTORS
    AUTHORIZED_FACE_VECTORS = []
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    ref_paths = [TEST_DIR / "facial_recognition.jpg"]
    for p in TEST_DIR.glob("*.*"):
        if p not in ref_paths and p.suffix.lower() in ('.jpg', '.jpeg', '.png') and ("face" in p.name.lower() or "person" in p.name.lower()):
            ref_paths.append(p)

    for ref_path in ref_paths:
        if not ref_path.exists():
            continue
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            continue
        gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_crop = cv2.resize(gray[y:y+h, x:x+w], (64, 64)).astype(np.float32)
                vec = (face_crop - np.mean(face_crop)) / (np.std(face_crop) + 1e-6)
                # Flatten and normalize to unit vector for cosine similarity
                flat = vec.flatten()
                flat = flat / (np.linalg.norm(flat) + 1e-8)
                AUTHORIZED_FACE_VECTORS.append(flat)
        else:
            # No face detected — use full image as reference (resize to 64x64)
            face_crop = cv2.resize(gray, (64, 64)).astype(np.float32)
            vec = (face_crop - np.mean(face_crop)) / (np.std(face_crop) + 1e-6)
            flat = vec.flatten()
            flat = flat / (np.linalg.norm(flat) + 1e-8)
            AUTHORIZED_FACE_VECTORS.append(flat)


def suspicious(frame, model, tracker, feed):
    output = frame.copy()
    h, w = frame.shape[:2]
    dets = []

    try:
        results = model(frame, conf=0.22, classes=[0], device=DEVICE, half=USE_HALF, verbose=False)[0]
    except Exception:
        results = None

    raw = []
    if results is not None and results.boxes is not None:
        for b in results.boxes:
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            # Min 20px height for 240p video (was 40px — too aggressive, missed most detections)
            if (y2 - y1) >= 20:
                raw.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "type": "person"
                })

    tracked = tracker.update(raw) if tracker is not None else []
    max_conf = 0.0

    for d in tracked:
        x1, y1, x2, y2 = d["bbox"]
        tid = d["track_id"]
        conf = d.get("conf", 0.75)
        max_conf = max(max_conf, conf)
        # In Cam 5, any nocturnal presence in this restricted stairwell is flagged as suspicious activity
        dets.append({
            "x1": round(x1 / w, 4),
            "y1": round(y1 / h, 4),
            "w": round((x2 - x1) / w, 4),
            "h": round((y2 - y1) / h, 4),
            "label": f"SUSPICIOUS: Person #{tid} ({int(conf * 100)}%)",
            "type": "suspicious",
            "color": "#ef4444"
        })

    if tracked:
        event(feed, "SUSPICIOUS", f"Suspicious Activity: Unauthorized entry in restricted stairwell (Person #{tracked[0]['track_id']})",
              {"track_id": tracked[0]['track_id']}, confidence=max_conf, object_type="Person", cooldown=2.0)

    with detections_lock:
        latest_detections[feed] = dets
    return output


def facial_rec(frame, detector, feed):
    output = frame.copy()
    h, w = frame.shape[:2]
    dets = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(50, 50))

    for (x, y, fw, fh) in faces:
        face_crop = cv2.resize(gray[y:y+fh, x:x+fw], (64, 64)).astype(np.float32)
        norm_crop = (face_crop - np.mean(face_crop)) / (np.std(face_crop) + 1e-6)
        # Flatten and re-normalize to unit vector for proper cosine similarity
        flat_crop = norm_crop.flatten()
        flat_crop = flat_crop / (np.linalg.norm(flat_crop) + 1e-8)
        sim = 0.0
        if AUTHORIZED_FACE_VECTORS:
            sims = [float(np.dot(flat_crop, v)) for v in AUTHORIZED_FACE_VECTORS]
            sim = max(sims)

        if sim >= 0.55:
            label = "Facial Recognition Successful"
            color = "#10b981"
            det_type = "verified_face"
            event(feed, "HUMAN_DETECTED", "Facial Recognition Successful: Authorized Face Verified",
                  {"status": "Verified"}, confidence=round(sim, 2), object_type="Person", cooldown=3.0)
        else:
            label = "Face Not Recognised"
            color = "#ef4444"
            det_type = "unrecognized_face"
            event(feed, "SUSPICIOUS", "Security Alert: Face Not Recognised",
                  {"status": "Unrecognized"}, confidence=round(max(0.0, sim), 2), object_type="Person", cooldown=3.0)

        dets.append({
            "x1": round(x / w, 4),
            "y1": round(y / h, 4),
            "w": round(fw / w, 4),
            "h": round(fh / h, 4),
            "label": label,
            "type": det_type,
            "color": color
        })

    with detections_lock:
        latest_detections[feed] = dets
    return output


def reset_tracker(processor):
    try:
        from ultralytics.trackers.basetrack import BaseTrack
        BaseTrack._count = 0
    except Exception:
        pass
    if hasattr(processor, "tracked_objects"):
        processor.tracked_objects.clear()
    model = getattr(processor, "model", processor)
    if hasattr(model, "predictor") and model.predictor is not None:
        if hasattr(model.predictor, "trackers"):
            for trk in model.predictor.trackers:
                try:
                    trk.reset()
                except Exception:
                    pass


def worker(feed, source):
    # Handle live webcam feed for Cam 6 (Facial Recognition)
    if feed == "facial_recognition":
        try:
            init_face_recognition()
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            capture = cv2.VideoCapture(0)
            if not capture.isOpened():
                # Fallback to static reference image if webcam 0 is unavailable
                ref_fallback = TEST_DIR / "facial_recognition.jpg"
                if ref_fallback.exists():
                    img = cv2.imread(str(ref_fallback))
                    statuses[feed]["running"] = True
                    while not stop_event.is_set():
                        frame = img.copy()
                        output = facial_rec(frame, detector, feed)
                        ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            with locks[feed]:
                                frames[feed] = encoded.tobytes()
                        time.sleep(0.1)
                    return
                else:
                    raise RuntimeError("Webcam 0 could not be opened and no reference image found.")

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            statuses[feed]["running"] = True
            while not stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    time.sleep(0.04)
                    continue
                output = facial_rec(frame, detector, feed)
                ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    with locks[feed]:
                        frames[feed] = encoded.tobytes()
                time.sleep(0.03)
        except Exception as exc:
            statuses[feed]["error"] = str(exc)
            event(feed, "ERROR", str(exc), cooldown=0)
        finally:
            if 'capture' in locals() and hasattr(capture, 'isOpened') and capture.isOpened():
                capture.release()
        return

    # Handle static image feeds
    if isinstance(source, (str, Path)) and str(source).lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            init_face_recognition()
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            img = cv2.imread(str(source))
            if img is None:
                raise RuntimeError(f"Could not open image: {source}")
            statuses[feed]["running"] = True
            while not stop_event.is_set():
                frame = img.copy()
                if feed == "facial_recognition":
                    output = facial_rec(frame, detector, feed)
                else:
                    output = frame
                ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    with locks[feed]:
                        frames[feed] = encoded.tobytes()
                time.sleep(0.1)
        except Exception as exc:
            statuses[feed]["error"] = str(exc)
            event(feed, "ERROR", str(exc), cooldown=0)
        return

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        error = f"Could not open video: {source}"
        statuses[feed]["error"] = error
        event(feed, "ERROR", error, cooldown=0)
        return

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        fps = 25.0
    frame_duration = 1.0 / fps
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        tracker = None
        if feed == "virtual_fence":
            polygon_pts = [(90, 125), (250, 125), (250, 235), (90, 235)]
            processor = VirtualFenceEngine(str(ROOT / "yolo11n.pt"), "polygon", polygon_pts, [0], device=DEVICE)
        elif feed == "vehicle_detection":
            processor = YOLO(str(ROOT / "yolo11n.pt"))
            if USE_HALF:
                processor.to(DEVICE)
            tracker = SurveillanceTracker(max_lost_frames=4, iou_thresh=0.25, dist_thresh=100)
        elif feed == "human_detection":
            processor = YOLO(str(ROOT / "yolo11n.pt"))
            if USE_HALF:
                processor.to(DEVICE)
            tracker = SurveillanceTracker(max_lost_frames=6, iou_thresh=0.2, dist_thresh=120)
        elif feed == "suspicious_activity":
            processor = YOLO(str(ROOT / "yolo11n.pt"))
            if USE_HALF:
                processor.to(DEVICE)
            tracker = SurveillanceTracker(max_lost_frames=6, iou_thresh=0.15, dist_thresh=120)
        else:
            processor = YOLO(str(ROOT / "yolo11n.pt"))
            if USE_HALF:
                processor.to(DEVICE)
            model_path = ANPR_ROOT / "anpr" / "models" / "plate_detector.pt"
            if ANPREngine is None or not model_path.exists():
                raise RuntimeError(f"ANPR unavailable; missing model: {model_path}")
            from anpr.config import ANPRConfig
            plate_engine = ANPREngine(ANPRConfig(
                plate_detector_weights=str(model_path),
                ocr_engine="easyocr",
                ocr_use_gpu=torch.cuda.is_available(),
            ))

        statuses[feed]["running"] = True
        frame_idx = 0
        plate_cache = {}
        # Inference Cadence: Run neural network inference every 2nd frame (15 FPS AI)
        # This keeps video streams running at smooth 30 FPS while cutting compute load and heat by 50%!
        INFER_CADENCE = 2

        while not stop_event.is_set():
            loop_start = time.time()
            frame_idx += 1

            ok, frame = capture.read()
            if not ok or frame is None:
                # End of video — reset and loop back to start
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                plate_cache.clear()
                if hasattr(processor, "reset"):
                    processor.reset()
                if tracker is not None:
                    tracker.reset()
                with detections_lock:
                    latest_detections[feed] = []
                continue

            should_infer = (frame_idx % INFER_CADENCE == 0)

            if should_infer:
                if feed == "virtual_fence":
                    output = fence(frame, processor, feed)
                elif feed == "vehicle_detection":
                    output = vehicles(frame, processor, tracker, feed)
                elif feed == "human_detection":
                    output = humans(frame, processor, tracker, feed)
                elif feed == "suspicious_activity":
                    output = suspicious(frame, processor, tracker, feed)
                else:
                    output = anpr(frame, processor, plate_engine, feed, frame_idx=frame_idx, plate_cache=plate_cache)
            else:
                output = frame

            ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with locks[feed]:
                    frames[feed] = encoded.tobytes()

            # Dynamic pacing to video FPS — sleep just enough to maintain real-time speed
            elapsed = time.time() - loop_start
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as exc:
        statuses[feed]["error"] = str(exc)
        event(feed, "ERROR", str(exc), cooldown=0)
    finally:
        capture.release()



def stream(feed):
    while not stop_event.is_set():
        with locks[feed]:
            current = frames[feed]
        if current is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + current + b"\r\n"


@app.on_event("startup")
def start():
    for feed, source in SOURCES.items():
        threading.Thread(target=worker, args=(feed, source), daemon=True).start()


@app.on_event("shutdown")
def stop():
    stop_event.set()


@app.get("/", response_class=HTMLResponse)
def home():
    rendered = HTML.replace("{{FAVICON_URL}}", FAVICON_DATA_URI or "/assets/favion.png")
    return HTMLResponse(rendered)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    fav_path = ASSETS_DIR / "favion.png"
    if fav_path.exists():
        return FileResponse(fav_path, media_type="image/png")
    return HTMLResponse("", status_code=404)


@app.get("/feed/{feed}")
def feed(feed: str):
    if feed not in SOURCES:
        return HTMLResponse("Unknown feed", status_code=404)
    return StreamingResponse(stream(feed), media_type="multipart/x-mixed-replace; boundary=frame")


fence_metadata = {
    "type": "polygon",
    "label": "Restricted Zone",
    "points": [[28.125, 52.08], [78.125, 52.08], [78.125, 97.92], [28.125, 97.92]]
}


@app.get("/api/status")
def status():
    with status_lock:
        ev_list = list(events)
    with detections_lock:
        det_map = dict(latest_detections)
    active_breaches = []
    for feed_key, dets in det_map.items():
        if feed_key == "virtual_fence" and any(d.get("type") == "intruder" for d in dets):
            active_breaches.append({
                "feed": feed_key,
                "cam": CAM_NAMES.get(feed_key, feed_key),
                "title": "Breach Detected",
            })
        elif feed_key == "suspicious_activity" and any(d.get("type") == "suspicious" for d in dets):
            active_breaches.append({
                "feed": feed_key,
                "cam": CAM_NAMES.get(feed_key, feed_key),
                "title": "Breach Detected",
            })
        elif feed_key == "facial_recognition" and any(d.get("type") == "unrecognized_face" for d in dets):
            active_breaches.append({
                "feed": feed_key,
                "cam": CAM_NAMES.get(feed_key, feed_key),
                "title": "Breach Detected",
            })
    return {
        "feeds": statuses,
        "events": ev_list,
        "detections": det_map,
        "fence": fence_metadata,
        "active_breaches": active_breaches,
    }


@app.get("/api/events")
def get_events():
    with status_lock:
        return list(events)


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drishti · Intelligent Border Video Analytics Platform</title>
  <link rel="icon" type="image/png" href="{{FAVICON_URL}}">
  <link rel="shortcut icon" type="image/png" href="{{FAVICON_URL}}">
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --accent: #2563eb;
      --live-green: #22c55e;
      --font-stack: "Times New Roman", Times, Georgia, serif;
    }
    * {
      box-sizing: border-box;
      font-family: "Times New Roman", Times, Georgia, serif !important;
    }
    body {
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text-main);
      font-family: var(--font-stack) !important;
      -webkit-font-smoothing: antialiased;
    }
    .top-nav-container {
      position: sticky;
      top: 0;
      z-index: 20;
    }
    /* ── GOI National Official Identity Bar ── */
    .gov-strip {
      background: #0b1329;
      color: #e2e8f0;
      font-size: 11.5px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .gov-strip-inner {
      display: flex;
      align-items: center;
      padding: 5px 24px;
      max-width: 1400px;
      margin: 0 auto;
    }
    .gov-left {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .gov-flag-icon {
      font-size: 17px;
      line-height: 1;
      display: inline-block;
    }
    .gov-badge-in {
      font-weight: 700;
      color: #fdba74;
    }
    .gov-divider {
      color: #475569;
    }
    .gov-bullet {
      color: #64748b;
      font-size: 7px;
    }
    .gov-mha {
      color: #94a3b8;
    }
    .gov-tricolor {
      height: 4px;
      width: 100%;
      background: linear-gradient(90deg, #FF9933 0%, #FF9933 33.33%, #FFFFFF 33.33%, #FFFFFF 66.66%, #138808 66.66%, #138808 100%);
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 24px;
      background: #ffffff;
      border-bottom: 1px solid var(--border);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-logo {
      height: 40px;
      width: auto;
      max-width: 150px;
      object-fit: contain;
      display: block;
    }
    .brand-text {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .brand-title {
      font-size: 17px;
      font-weight: 700;
      margin: 0;
      color: var(--text-main);
      line-height: 1.2;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .brand-sub {
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 500;
      letter-spacing: 0.02em;
    }
    .badge-live {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 11px;
      font-weight: 600;
      color: #15803d;
      background: #f0fdf4;
      padding: 2px 7px;
      border-radius: 12px;
      border: 1px solid #bbf7d0;
    }
    .dot {
      width: 6px;
      height: 6px;
      background: var(--live-green);
      border-radius: 50%;
      display: inline-block;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #0f172a;
      color: #ffffff;
      border: none;
      border-radius: 6px;
      padding: 7px 13px;
      font-size: 12.5px;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    .btn:hover { background: #1e293b; }
    .btn-export {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #ffffff;
      color: #0f172a;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .btn-export:hover {
      background: #f1f5f9;
      border-color: #cbd5e1;
      color: #0f172a;
    }
    .event-count-badge {
      background: var(--accent);
      color: #ffffff;
      font-size: 11px;
      font-weight: 600;
      padding: 1px 6px;
      border-radius: 10px;
    }

    /* ── Tactical Red Alert Threat Banner ── */
    .alert-banner {
      background: linear-gradient(90deg, #991b1b 0%, #b91c1c 50%, #991b1b 100%);
      color: #ffffff;
      border-bottom: 2px solid #ef4444;
      animation: alertPulse 2.5s infinite ease-in-out;
      box-shadow: 0 4px 14px rgba(185, 28, 28, 0.3);
    }
    @keyframes alertPulse {
      0% { background-color: #991b1b; }
      50% { background-color: #dc2626; }
      100% { background-color: #991b1b; }
    }
    .alert-banner-inner {
      max-width: 1400px;
      margin: 0 auto;
      padding: 6px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .alert-left-group {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .alert-pulse-icon {
      font-size: 16px;
      animation: iconThrob 1s infinite alternate;
    }
    @keyframes iconThrob {
      from { transform: scale(1); }
      to { transform: scale(1.2); }
    }
    .alert-message {
      font-size: 13px;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: 0.02em;
    }
    .alert-controls {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .alert-action-btn {
      background: #ffffff;
      color: #991b1b;
      border: none;
      padding: 4px 12px;
      font-size: 12px;
      font-weight: 700;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.15s;
      white-space: nowrap;
    }
    .alert-action-btn:hover {
      background: #fef2f2;
    }

    /* Drawer Header Actions */
    .drawer-title-wrap {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .drawer-subtitle {
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 500;
    }
    .drawer-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    main {
      max-width: 1400px;
      margin: 24px auto;
      padding: 0 24px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
      transition: transform 0.15s ease, box-shadow 0.25s ease, border-color 0.25s ease;
      cursor: pointer;
    }
    .card:hover {
      border-color: #cbd5e1;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }
    /* Subtle pulsing red glow on camera card when an active breach is happening */
    .card.breach-glow {
      border-color: #ef4444 !important;
      box-shadow: 0 0 0 2px #ef4444, 0 0 22px rgba(239, 68, 68, 0.45) !important;
      animation: cardBreachPulse 1.8s infinite ease-in-out;
    }
    @keyframes cardBreachPulse {
      0% { box-shadow: 0 0 0 2px #ef4444, 0 0 8px rgba(239, 68, 68, 0.35); }
      50% { box-shadow: 0 0 0 2.5px #dc2626, 0 0 22px rgba(239, 68, 68, 0.6); }
      100% { box-shadow: 0 0 0 2px #ef4444, 0 0 8px rgba(239, 68, 68, 0.35); }
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 14px;
      background: #ffffff;
      border-bottom: 1px solid var(--border);
    }
    .card-header h2 {
      font-size: 14px;
      font-weight: 600;
      margin: 0;
      color: var(--text-main);
    }
    .feed-wrapper {
      position: relative;
      background: #000;
      aspect-ratio: 16/9;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .feed-wrapper img {
      width: 100%;
      height: 100%;
      object-fit: fill;
      display: block;
    }

    /* Crisp SVG Restricted Zone Overlay (Solid Red Line) */
    .fence-svg {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1;
    }

    #modalFenceLayer {
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 1;
    }

    /* Crisp HTML Vector Overlay Layers */
    .bbox-layer {
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 2;
    }
    .bbox-tag {
      position: absolute;
      border: 1.5px solid #2563eb;
      border-radius: 3px;
      box-sizing: border-box;
      transition: all 0.05s ease-out;
    }
    .bbox-label {
      position: absolute;
      top: -22px;
      left: -1px;
      background: #2563eb;
      color: #ffffff;
      font-size: 11px;
      font-weight: 600;
      padding: 1px 5px;
      border-radius: 3px;
      white-space: nowrap;
      font-family: var(--font-stack);
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
      letter-spacing: 0.01em;
    }

    /* Large plate number readout — top-left corner of the feed */
    .plate-banner {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(6, 182, 212, 0.92);
      color: #ffffff;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0.12em;
      padding: 6px 16px;
      border-radius: 6px;
      font-family: 'Courier New', Courier, monospace;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
      pointer-events: none;
      z-index: 5;
      white-space: nowrap;
    }

    /* Bigger plate banner inside enlarged modal */
    #bbox-modal .plate-banner {
      font-size: 36px;
      padding: 10px 24px;
      top: 16px;
      left: 16px;
      border-radius: 8px;
    }

    .feed-overlay-hint {
      position: absolute;
      bottom: 8px;
      right: 8px;
      background: rgba(15, 23, 42, 0.7);
      color: #ffffff;
      font-size: 11px;
      padding: 4px 8px;
      border-radius: 4px;
      opacity: 0;
      transition: opacity 0.15s ease;
      pointer-events: none;
    }
    .card:hover .feed-overlay-hint { opacity: 1; }

    .drawer-overlay {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.3);
      backdrop-filter: blur(2px);
      z-index: 100;
      display: none;
    }
    .drawer-overlay.active { display: block; }

    .drawer {
      position: fixed;
      top: 0;
      right: 0;
      width: 500px;
      max-width: 95vw;
      height: 100vh;
      background: #ffffff;
      box-shadow: -4px 0 24px rgba(0, 0, 0, 0.1);
      z-index: 101;
      display: flex;
      flex-direction: column;
      transform: translateX(100%);
      transition: transform 0.25s ease-in-out;
    }
    .drawer.active { transform: translateX(0); }

    .drawer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
    }
    .drawer-header h3 {
      font-size: 16px;
      font-weight: 600;
      margin: 0;
    }
    .close-btn {
      background: none;
      border: none;
      font-size: 20px;
      cursor: pointer;
      color: var(--text-muted);
      padding: 4px 8px;
      border-radius: 4px;
    }
    .close-btn:hover { background: var(--bg); color: var(--text-main); }

    .drawer-body {
      flex: 1;
      overflow-y: auto;
      padding: 10px 14px;
    }

    /* ── Structured Event Card ─────────────────────────────────────── */
    .event-card {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 10px;
      overflow: hidden;
      font-size: 12.5px;
    }
    .event-card-header {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-bottom: 1px solid var(--border);
      background: #f8fafc;
    }
    .event-type-pill {
      font-size: 11px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 12px;
      white-space: nowrap;
    }
    .pill-person   { background: #dcfce7; color: #15803d; }
    .pill-vehicle  { background: #fef3c7; color: #92400e; }
    .pill-anpr     { background: #cffafe; color: #0e7490; }
    .pill-fence    { background: #fee2e2; color: #991b1b; }
    .pill-suspicious { background: #fce7f3; color: #9d174d; }
    .pill-system   { background: #f1f5f9; color: #475569; }
    .severity-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
    }
    .sev-high   { background: #ef4444; }
    .sev-medium { background: #f59e0b; }
    .sev-low    { background: #22c55e; }
    .event-header-time {
      margin-left: auto;
      font-size: 11px;
      color: var(--text-muted);
      white-space: nowrap;
    }

    .event-card-body {
      display: grid;
      grid-template-columns: 96px 1fr;
      gap: 0;
    }
    .event-snapshot {
      width: 96px;
      min-height: 54px;
      background: #0f172a;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .event-snapshot img {
      width: 100%;
      height: auto;
      display: block;
    }
    .event-snapshot-placeholder {
      color: #475569;
      font-size: 10px;
      text-align: center;
      padding: 4px;
    }
    .event-meta {
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .event-meta-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 2px 10px;
    }
    .meta-row {
      display: flex;
      flex-direction: column;
    }
    .meta-label {
      font-size: 10px;
      color: var(--text-muted);
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .meta-value {
      font-size: 12px;
      color: var(--text-main);
      font-weight: 500;
    }
    .plate-value {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0.08em;
      font-family: 'Courier New', Courier, monospace;
      color: #0e7490;
    }
    .conf-bar-wrap {
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .conf-bar {
      flex: 1;
      height: 4px;
      background: #e2e8f0;
      border-radius: 2px;
      overflow: hidden;
    }
    .conf-bar-fill {
      height: 100%;
      border-radius: 2px;
      background: #2563eb;
    }

    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(6px);
      z-index: 200;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 12px;
    }
    .modal-overlay.active { display: flex; }
    .modal-content {
      background: #ffffff;
      border-radius: 8px;
      overflow: hidden;
      width: 98vw;
      height: 96vh;
      max-width: 1800px;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 20px;
      border-bottom: 1px solid var(--border);
    }
    .modal-header h3 { font-size: 16px; font-weight: 600; margin: 0; }
    .modal-body {
      background: #000;
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;
      padding: 0;
    }
    .modal-feed-wrapper {
      position: relative;
      width: 100%;
      height: 100%;
      max-width: 100%;
      max-height: 100%;
      aspect-ratio: 16/9;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .modal-feed-wrapper img {
      width: 100%;
      height: 100%;
      object-fit: fill;
      display: block;
    }

    @media (max-width: 1100px) {
      .grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 768px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="top-nav-container">
    <!-- GOI National Official Identity Bar -->
    <div class="gov-strip">
      <div class="gov-strip-inner">
        <div class="gov-left">
          <span class="gov-flag-icon">🇮🇳</span>
          <span class="gov-badge-in">भारत सरकार</span>
          <span class="gov-divider">|</span>
          <span>Government of India</span>
          <span class="gov-bullet">•</span>
          <span class="gov-mha">Ministry of Home Affairs (MHA)</span>
        </div>
      </div>
      <div class="gov-tricolor"></div>
    </div>

    <!-- Main Navigation Bar -->
    <header>
      <div class="brand">
        <img src="/assets/product_logo.png" alt="Drishti Logo" class="brand-logo">
        <div class="brand-text">
          <h1 class="brand-title">Drishti Monitoring <span class="badge-live"><span class="dot"></span> Live</span></h1>
          <span class="brand-sub">Intelligent Border Video Analytics Platform</span>
        </div>
      </div>

      <div class="header-actions">
        <button class="btn-export" onclick="exportIncidentLog()" title="Export Incident Log (CSV)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Export Log
        </button>
        <button class="btn" id="openEventsBtn">Events</button>
      </div>
    </header>

    <!-- Tactical Red Alert Threat Banner (Active Breaches Only) -->
    <div class="alert-banner" id="alertBanner" style="display: none;">
      <div class="alert-banner-inner">
        <div class="alert-left-group">
          <div class="alert-pulse-icon">⚠️</div>
          <div class="alert-message" id="alertBannerMsg">Cam 1 · Breach Detected</div>
        </div>
        <div class="alert-controls">
          <button class="alert-action-btn" onclick="inspectAlertFeed()">Inspect Feed</button>
        </div>
      </div>
    </div>
  </div>

  <main>
    <div class="grid">
      <div class="card" id="card-virtual_fence" onclick="openEnlarged('virtual_fence', 'Cam 1')">
        <div class="card-header">
          <h2>Cam 1</h2>
        </div>
        <div class="feed-wrapper">
          <img src="/feed/virtual_fence" alt="Cam 1 Feed">
          <svg class="fence-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
            <polygon points="28.125,52.08 78.125,52.08 78.125,97.92 28.125,97.92" fill="rgba(239, 68, 68, 0.18)" stroke="#ef4444" stroke-width="1.8" />
            <text x="30" y="58" fill="#ef4444" font-size="3.5" font-weight="600" font-family="Times New Roman, serif">Restricted Zone</text>
          </svg>
          <div class="bbox-layer" id="bbox-virtual_fence"></div>
        </div>
      </div>

      <div class="card" id="card-vehicle_detection" onclick="openEnlarged('vehicle_detection', 'Cam 2')">
        <div class="card-header">
          <h2>Cam 2</h2>
        </div>
        <div class="feed-wrapper">
          <img src="/feed/vehicle_detection" alt="Cam 2 Feed">
          <div class="bbox-layer" id="bbox-vehicle_detection"></div>
        </div>
      </div>

      <div class="card" id="card-human_detection" onclick="openEnlarged('human_detection', 'Cam 3')">
        <div class="card-header">
          <h2>Cam 3</h2>
        </div>
        <div class="feed-wrapper">
          <img src="/feed/human_detection" alt="Cam 3 Feed">
          <div class="bbox-layer" id="bbox-human_detection"></div>
        </div>
      </div>

      <div class="card" id="card-anpr" onclick="openEnlarged('anpr', 'Cam 4')">
        <div class="card-header">
          <h2>Cam 4</h2>
        </div>
        <div class="feed-wrapper">
          <img src="/feed/anpr" alt="Cam 4 Feed">
          <div class="bbox-layer" id="bbox-anpr"></div>
        </div>
      </div>

      <div class="card" id="card-suspicious_activity" onclick="openEnlarged('suspicious_activity', 'Cam 5')">
        <div class="card-header">
          <h2>Cam 5</h2>
        </div>
        <div class="feed-wrapper">
          <img src="/feed/suspicious_activity" alt="Cam 5 Feed">
          <div class="bbox-layer" id="bbox-suspicious_activity"></div>
        </div>
      </div>

      <div class="card" id="card-facial_recognition" onclick="openEnlarged('facial_recognition', 'Cam 6')">
        <div class="card-header">
          <h2>Cam 6</h2>
        </div>
        <div class="feed-wrapper">
          <img src="/feed/facial_recognition" alt="Cam 6 Feed">
          <div class="bbox-layer" id="bbox-facial_recognition"></div>
        </div>
      </div>
    </div>
  </main>

  <div class="drawer-overlay" id="drawerOverlay" onclick="closeEvents()"></div>
  <div class="drawer" id="eventsDrawer">
    <div class="drawer-header">
      <div class="drawer-title-wrap">
        <h3>Incident Event Log</h3>
        <span class="drawer-subtitle">Border Security Evidence Stream</span>
      </div>
      <div class="drawer-actions">
        <button class="btn-export" onclick="exportIncidentLog()" title="Export log as CSV">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Export CSV
        </button>
        <button class="close-btn" onclick="closeEvents()">&times;</button>
      </div>
    </div>
    <div class="drawer-body" id="eventsList">
      <p style="color: var(--text-muted); font-size: 13px;">No events recorded yet.</p>
    </div>
  </div>

  <div class="modal-overlay" id="modalOverlay" onclick="closeEnlarged(event)">
    <div class="modal-content" onclick="event.stopPropagation()">
      <div class="modal-header">
        <h3 id="modalTitle">Cam Feed</h3>
        <button class="close-btn" onclick="closeEnlarged()">&times;</button>
      </div>
      <div class="modal-body" id="modalBody">
        <div class="modal-feed-wrapper" id="modalFeedWrapper">
          <img id="enlargedImg" src="" alt="Enlarged Feed">
          <div id="modalFenceLayer"></div>
          <div class="bbox-layer" id="bbox-modal"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const eventsDrawer = document.getElementById('eventsDrawer');
    const drawerOverlay = document.getElementById('drawerOverlay');
    const modalOverlay = document.getElementById('modalOverlay');
    const modalTitle = document.getElementById('modalTitle');
    const enlargedImg = document.getElementById('enlargedImg');
    const eventsList = document.getElementById('eventsList');
    const modalFenceLayer = document.getElementById('modalFenceLayer');
    let activeEnlargedKey = null;
    let allEventsData = [];
    const dismissedAlerts = new Set();
    let currentAlertEvent = null;


    // ── Tactical Red Alert Banner ──
    let currentAlertFeed = 'virtual_fence';

    function inspectAlertFeed() {
      if (currentAlertFeed) {
        const camMap = {
          'virtual_fence': 'Cam 1',
          'vehicle_detection': 'Cam 2',
          'human_detection': 'Cam 3',
          'anpr': 'Cam 4',
          'suspicious_activity': 'Cam 5',
          'facial_recognition': 'Cam 6'
        };
        const camLabel = camMap[currentAlertFeed] || currentAlertFeed;
        openEnlarged(currentAlertFeed, camLabel);
      }
    }

    // ── Export Incident Log (CSV) ──
    function exportIncidentLog() {
      if (!allEventsData || allEventsData.length === 0) {
        alert("No incident events recorded yet to export.");
        return;
      }
      const headers = ["Log ID", "Date", "Time IST", "Camera", "Event Type", "Object", "Plate Number", "Confidence", "Severity", "Status", "Details"];
      const rows = allEventsData.map((e, idx) => {
        const date = e.timestamp ? e.timestamp.split(' ')[0] : '';
        const time = e.time || (e.timestamp ? e.timestamp.split(' ')[1] : '');
        const conf = e.confidence != null ? `${Math.round(e.confidence * 100)}%` : '';
        const plate = e.plate_number ? `"${e.plate_number}"` : '';
        const msg = `"${(e.message || '').replace(/"/g, '""')}"`;
        return [idx + 1, date, time, e.feed || '', e.event_type || '', e.object || '', plate, conf, e.severity || '', e.status || 'Active', msg];
      });

      const csvContent = "\\uFEFF" + [
        headers.join(','),
        ...rows.map(r => r.join(','))
      ].join('\\r\\n');

      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const now = new Date();
      const dateTag = now.toISOString().replace(/[:.]/g, '-').slice(0, 19);
      a.href = url;
      a.download = `Drishti_Incident_Log_${dateTag}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }

    document.getElementById('openEventsBtn').addEventListener('click', () => {
      eventsDrawer.classList.add('active');
      drawerOverlay.classList.add('active');
    });

    function closeEvents() {
      eventsDrawer.classList.remove('active');
      drawerOverlay.classList.remove('active');
    }

    let lastReceivedDetections = {};

    function openEnlarged(feedKey, title) {
      activeEnlargedKey = feedKey;
      modalTitle.textContent = title;

      // Ensure 16:9 for ANPR, 4:3 for virtual_fence, human, suspicious
      const wrapper = document.getElementById('modalFeedWrapper');
      if (wrapper) {
        wrapper.style.aspectRatio = (feedKey === 'anpr') ? '16/9' : '4/3';
      }

      // Temporarily pause card image to stay strictly within browser HTTP/1.1 connection limit
      const cardImg = document.querySelector('#card-' + feedKey + ' img');
      if (cardImg) {
        cardImg.dataset.activeSrc = cardImg.src;
        cardImg.src = '';
      }

      enlargedImg.src = '/feed/' + feedKey;

      if (feedKey === 'virtual_fence') {
        modalFenceLayer.innerHTML = `
          <svg class="fence-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
            <polygon points="28.125,52.08 78.125,52.08 78.125,97.92 28.125,97.92" fill="rgba(239, 68, 68, 0.18)" stroke="#ef4444" stroke-width="1.8" />
            <text x="30" y="58" fill="#ef4444" font-size="3.5" font-weight="600" font-family="Times New Roman, serif">Restricted Zone</text>
          </svg>
        `;
      } else {
        modalFenceLayer.innerHTML = '';
      }
      modalOverlay.classList.add('active');

      // Immediately render current active detections without waiting for next poll
      if (lastReceivedDetections && lastReceivedDetections[feedKey]) {
        renderFeedDetections('modal', lastReceivedDetections[feedKey]);
      }
    }

    function closeEnlarged(e) {
      if (!e || e.target === modalOverlay || e.target.classList.contains('close-btn')) {
        modalOverlay.classList.remove('active');
        enlargedImg.src = '';
        if (activeEnlargedKey) {
          const cardImg = document.querySelector('#card-' + activeEnlargedKey + ' img');
          if (cardImg && cardImg.dataset.activeSrc) {
            cardImg.src = cardImg.dataset.activeSrc;
          }
        }
        activeEnlargedKey = null;
        modalFenceLayer.innerHTML = '';
        document.getElementById('bbox-modal').innerHTML = '';
      }
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeEvents();
        closeEnlarged();
      }
    });

    function escapeHtml(str) {
      return String(str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function renderFeedDetections(feedKey, detections) {
      const container = document.getElementById('bbox-' + feedKey);
      if (!container) return;
      if (!detections || detections.length === 0) {
        container.innerHTML = '';
        return;
      }

      // Extract the best (first) plate reading for the banner
      const plateDet = detections.find(d => d.type === 'plate');

      let html = detections.map(d => {
        const isPlate = d.type === 'plate';
        const borderPx = isPlate ? '2.5px' : '1.5px';
        const rawLabel = d.label || '';
        const label = isPlate ? escapeHtml(rawLabel.replace(/^PLATE:\s*/i, '')) : escapeHtml(rawLabel);
        return `
        <div class="bbox-tag" style="left:${d.x1 * 100}%; top:${d.y1 * 100}%; width:${d.w * 100}%; height:${d.h * 100}%; border-color:${d.color}; border-width:${borderPx};">
          <span class="bbox-label" style="background:${d.color}; ${isPlate ? 'font-size:13px; padding:2px 7px; letter-spacing:0.05em;' : ''}">${label}</span>
        </div>`;
      }).join('');

      // Big plate banner in top-left corner
      if (plateDet) {
        const plateNum = escapeHtml((plateDet.label || '').replace(/^PLATE:\s*/i, ''));
        html += `<div class="plate-banner">${plateNum}</div>`;
      }

      container.innerHTML = html;
    }

    let lastRenderedEventKey = '';
    async function refreshStatus() {
      try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        const events = data.events || [];
        allEventsData = events;

        // Tactical Red Alert Banner & Camera Card Glow (ONLY when active breach is happening right now!)
        const activeBreaches = data.active_breaches || [];
        const isBreached = activeBreaches.length > 0;
        const banner = document.getElementById('alertBanner');
        const bannerMsg = document.getElementById('alertBannerMsg');

        // Apply / remove subtle red glow on each camera card
        ['virtual_fence', 'vehicle_detection', 'human_detection', 'anpr', 'suspicious_activity', 'facial_recognition'].forEach(fk => {
          const cardEl = document.getElementById('card-' + fk);
          if (cardEl) {
            const hasBreach = activeBreaches.some(b => b.feed === fk);
            if (hasBreach) {
              cardEl.classList.add('breach-glow');
            } else {
              cardEl.classList.remove('breach-glow');
            }
          }
        });

        // Banner only appears when there is an active breach happening right now
        if (isBreached) {
          const alertTexts = activeBreaches.map(b => `${b.cam} · ${b.title || 'Breach Detected'}`).join('  |  ');
          if (bannerMsg) bannerMsg.textContent = alertTexts;
          currentAlertFeed = activeBreaches[0].feed;
          if (banner) banner.style.display = 'block';
        } else {
          if (banner) banner.style.display = 'none';
        }

        // Only rebuild DOM if the event list actually changed
        const currentEventKey = events.length > 0 ? (events[0].timestamp + '_' + events.length) : 'empty';
        if (currentEventKey !== lastRenderedEventKey) {
          lastRenderedEventKey = currentEventKey;
          if (events.length === 0) {
            eventsList.innerHTML = '<p style="color: var(--text-muted); font-size: 13px;">No events recorded yet.</p>';
          } else {
            eventsList.innerHTML = '';
            const PILL_CLASS = {
              'Person Detected':   'pill-person',
              'Vehicle Detected':  'pill-vehicle',
              'ANPR Detected':     'pill-anpr',
              'Fence Breach':      'pill-fence',
              'Suspicious Activity': 'pill-suspicious',
              'Camera/System Alert': 'pill-system',
            };
            const SEV_CLASS = { High: 'sev-high', Medium: 'sev-medium', Low: 'sev-low' };
            events.forEach(e => {
              const pillCls = PILL_CLASS[e.event_type] || 'pill-system';
              const sevCls  = SEV_CLASS[e.severity] || 'sev-low';
              const confPct = e.confidence != null ? Math.round(e.confidence * 100) : null;
              const snapHtml = e.snapshot
                ? `<img src="data:image/jpeg;base64,${e.snapshot}" alt="snap">`
                : `<div class="event-snapshot-placeholder">No<br>Preview</div>`;

              const card = document.createElement('div');
              card.className = 'event-card';
              card.innerHTML = `
                <div class="event-card-header">
                  <div class="severity-dot ${sevCls}"></div>
                  <span class="event-type-pill ${pillCls}">${escapeHtml(e.event_type)}</span>
                  <span class="event-header-time">${escapeHtml(e.feed)} · ${escapeHtml(e.time)}</span>
                </div>
                <div class="event-card-body">
                  <div class="event-snapshot">${snapHtml}</div>
                  <div class="event-meta">
                    <div class="event-meta-grid">
                      <div class="meta-row">
                        <span class="meta-label">Camera</span>
                        <span class="meta-value">${escapeHtml(e.feed)}</span>
                      </div>
                      <div class="meta-row">
                        <span class="meta-label">Time</span>
                        <span class="meta-value">${escapeHtml(e.time)}</span>
                      </div>
                      <div class="meta-row">
                        <span class="meta-label">Object</span>
                        <span class="meta-value">${escapeHtml(e.object || '—')}</span>
                      </div>
                      <div class="meta-row">
                        <span class="meta-label">Severity</span>
                        <span class="meta-value">${escapeHtml(e.severity)}</span>
                      </div>
                    </div>
                    ${e.plate_number ? `
                    <div class="meta-row" style="margin-top:4px;">
                      <span class="meta-label">Plate Number</span>
                      <span class="plate-value">${escapeHtml(e.plate_number)}</span>
                    </div>` : ''}
                    ${confPct != null ? `
                    <div class="meta-row" style="margin-top:4px;">
                      <span class="meta-label">Confidence</span>
                      <div class="conf-bar-wrap">
                        <div class="conf-bar"><div class="conf-bar-fill" style="width:${confPct}%"></div></div>
                        <span class="meta-value">${confPct}%</span>
                      </div>
                    </div>` : ''}
                    <div class="meta-row" style="margin-top:4px;">
                      <span class="meta-label">Status</span>
                      <span class="meta-value">${escapeHtml(e.status || 'Active')}</span>
                    </div>
                  </div>
                </div>`;
              eventsList.appendChild(card);
            });
          }
        }

        const detections = data.detections || {};
        lastReceivedDetections = detections;
        for (const [feedKey, detList] of Object.entries(detections)) {
          renderFeedDetections(feedKey, detList);
          if (activeEnlargedKey === feedKey) {
            renderFeedDetections('modal', detList);
          }
        }
      } catch (err) {
        console.error('Failed to fetch status', err);
      }
    }

    refreshStatus();
    setInterval(refreshStatus, 60);
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_dashboard:app", host="0.0.0.0", port=8000, reload=False)

