import base64
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "test"
ANPR_ROOT = ROOT / "anpr_engine"
sys.path.insert(0, str(ANPR_ROOT))

from virtual_fence_engine import VirtualFenceEngine

try:
    from anpr import ANPREngine
except Exception:
    ANPREngine = None

app = FastAPI(title="Drishti Monitoring MVP")

SOURCES = {
    "virtual_fence": TEST_DIR / "virtual_fence.mp4",
    "vehicle_detection": TEST_DIR / "vehicle_detection.mp4",
    "human_detection": TEST_DIR / "human_deetection.mp4",
    "anpr": TEST_DIR / "Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4",
}

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


def vehicles(frame, model, feed):
    output = frame.copy()
    h, w = frame.shape[:2]
    dets = []
    count = 0
    max_conf = 0.0
    for result in model(frame, conf=0.4, verbose=False):
        if result.boxes is None:
            continue
        for detection in result.boxes:
            name = result.names[int(detection.cls[0])]
            if name.lower() not in {"car", "truck", "bus", "motorcycle"}:
                continue
            count += 1
            conf = float(detection.conf[0])
            max_conf = max(max_conf, conf)
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            cv2.rectangle(output, (x1, y1), (x2, y2), (235, 140, 50), 1)
            dets.append({
                "x1": round(x1 / w, 4),
                "y1": round(y1 / h, 4),
                "w": round((x2 - x1) / w, 4),
                "h": round((y2 - y1) / h, 4),
                "label": f"{name} {conf:.2f}",
                "type": "vehicle",
                "color": "#f59e0b"
            })
    if count:
        event(feed, "VEHICLE_DETECTED", f"{count} vehicle(s) detected",
              {"count": count}, confidence=max_conf, object_type="Vehicle")
    with detections_lock:
        latest_detections[feed] = dets
    return output


def humans(frame, model, feed):
    output = frame.copy()
    h, w = frame.shape[:2]
    dets = []
    count = 0
    max_conf = 0.0
    for result in model(frame, conf=0.45, verbose=False):
        if result.boxes is None:
            continue
        for detection in result.boxes:
            count += 1
            conf = float(detection.conf[0])
            max_conf = max(max_conf, conf)
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            cv2.rectangle(output, (x1, y1), (x2, y2), (50, 205, 50), 1)
            dets.append({
                "x1": round(x1 / w, 4),
                "y1": round(y1 / h, 4),
                "w": round((x2 - x1) / w, 4),
                "h": round((y2 - y1) / h, 4),
                "label": f"person {conf:.2f}",
                "type": "human",
                "color": "#10b981"
            })
    if count:
        event(feed, "HUMAN_DETECTED", f"{count} person(s) detected",
              {"count": count}, confidence=max_conf, object_type="Person")
    with detections_lock:
        latest_detections[feed] = dets
    return output


def fence(frame, engine, feed):
    output, alerts, detections = engine.process_frame(frame)
    h, w = frame.shape[:2]
    dets = []
    for alert in alerts:
        event(feed, "INTRUSION_DETECTED",
              f"Person crossed restricted zone (track {alert['track_id']})",
              alert, 0.8, object_type="Person")
    
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


def anpr(frame, vehicle_model, plate_engine, feed):
    output = frame.copy()
    h, w = frame.shape[:2]
    dets = []
    # Use track() so each vehicle gets a stable integer track_id across frames
    # which enables temporal fusion to accumulate OCR observations correctly.
    try:
        results = vehicle_model.track(frame, conf=0.4, persist=True, verbose=False)
    except Exception:
        results = vehicle_model(frame, conf=0.4, verbose=False)

    for result in results:
        if result.boxes is None:
            continue
        for detection in result.boxes:
            name = result.names[int(detection.cls[0])]
            if name.lower() not in {"car", "truck", "bus", "motorcycle"}:
                continue
            vx1, vy1, vx2, vy2 = map(int, detection.xyxy[0])
            vehicle_box = (vx1, vy1, vx2, vy2)
            # Use YOLO track_id for stable vehicle identity; fallback to position
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
                    pb = anpr_res.get("plate_bbox") or vehicle_box
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
                    event(feed, "PLATE_DETECTED", f"Plate detected: {plate}",
                          anpr_res, plate_number=plate,
                          confidence=anpr_res.get("confidence"),
                          object_type="Vehicle")
            except Exception:
                pass
    with detections_lock:
        latest_detections[feed] = dets
    return output


def worker(feed, source):
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
        if feed == "virtual_fence":
            polygon_pts = [(20, 40), (160, 40), (160, 230), (20, 230)]
            processor = VirtualFenceEngine(str(ROOT / "yolo11n.pt"), "polygon", polygon_pts, [0])
        elif feed == "vehicle_detection":
            processor = YOLO(str(ROOT / "yolo11n.pt"))
        elif feed == "human_detection":
            processor = YOLO(str(ROOT / "yolo11n-pose.pt"))
        else:
            processor = YOLO(str(ROOT / "yolo11n.pt"))
            model_path = ANPR_ROOT / "anpr" / "models" / "plate_detector.pt"
            if ANPREngine is None or not model_path.exists():
                raise RuntimeError(f"ANPR unavailable; missing model: {model_path}")
            from anpr.config import ANPRConfig
            plate_engine = ANPREngine(ANPRConfig(
                plate_detector_weights=str(model_path),
                ocr_engine="easyocr",
            ))

        statuses[feed]["running"] = True
        start_wall_time = time.time()

        while not stop_event.is_set():
            now = time.time()
            elapsed = now - start_wall_time
            target_frame = int(elapsed * fps)

            if total_frames > 0 and target_frame >= total_frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                start_wall_time = time.time()
                target_frame = 0
                if hasattr(processor, "tracked_objects"):
                    processor.tracked_objects.clear()

            current_pos = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if target_frame > current_pos + 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                start_wall_time = time.time()
                if hasattr(processor, "tracked_objects"):
                    processor.tracked_objects.clear()
                continue

            if feed == "virtual_fence":
                output = fence(frame, processor, feed)
            elif feed == "vehicle_detection":
                output = vehicles(frame, processor, feed)
            elif feed == "human_detection":
                output = humans(frame, processor, feed)
            else:
                output = anpr(frame, processor, plate_engine, feed)

            ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with locks[feed]:
                    frames[feed] = encoded.tobytes()

            proc_duration = time.time() - now
            if proc_duration < frame_duration:
                time.sleep(frame_duration - proc_duration)
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
    return HTMLResponse(HTML)


@app.get("/feed/{feed}")
def feed(feed: str):
    if feed not in SOURCES:
        return HTMLResponse("Unknown feed", status_code=404)
    return StreamingResponse(stream(feed), media_type="multipart/x-mixed-replace; boundary=frame")


fence_metadata = {
    "type": "polygon",
    "label": "Restricted Zone",
    "points": [[6.25, 16.67], [50.0, 16.67], [50.0, 95.83], [6.25, 95.83]]
}


@app.get("/api/status")
def status():
    with status_lock:
        ev_list = list(events)
    with detections_lock:
        det_map = dict(latest_detections)
    return {"feeds": statuses, "events": ev_list, "detections": det_map, "fence": fence_metadata}


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
  <title>Drishti Monitoring</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --accent: #2563eb;
      --live-green: #22c55e;
      --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text-main);
      font-family: var(--font-stack);
      -webkit-font-smoothing: antialiased;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 24px;
      background: #ffffff;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .brand h1 {
      font-size: 18px;
      font-weight: 600;
      margin: 0;
      color: var(--text-main);
    }
    .badge-live {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 12px;
      font-weight: 500;
      color: #15803d;
      background: #f0fdf4;
      padding: 2px 8px;
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
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #0f172a;
      color: #ffffff;
      border: none;
      border-radius: 6px;
      padding: 8px 14px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    .btn:hover { background: #1e293b; }
    .event-count-badge {
      background: var(--accent);
      color: #ffffff;
      font-size: 11px;
      font-weight: 600;
      padding: 1px 6px;
      border-radius: 10px;
    }
    main {
      max-width: 1400px;
      margin: 24px auto;
      padding: 0 24px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
      transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
      cursor: pointer;
    }
    .card:hover {
      border-color: #cbd5e1;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
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
      object-fit: cover;
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
    .modal-body img {
      width: 100%;
      height: 100%;
      max-width: 100%;
      max-height: 100%;
      object-fit: cover;
    }

    @media (max-width: 768px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>Drishti Monitoring</h1>
      <span class="badge-live"><span class="dot"></span> Live</span>
    </div>
    <div class="header-actions">
      <button class="btn" id="openEventsBtn">
        Events <span class="event-count-badge" id="eventBadge">0</span>
      </button>
    </div>
  </header>

  <main>
    <div class="grid">
      <div class="card" onclick="openEnlarged('virtual_fence', 'Cam 1')">
        <div class="card-header">
          <h2>Cam 1</h2>
        </div>
        <div class="feed-wrapper">
          <img src="/feed/virtual_fence" alt="Cam 1 Feed">
          <svg class="fence-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
            <polygon points="6.25,16.67 50,16.67 50,95.83 6.25,95.83" fill="rgba(239, 68, 68, 0.18)" stroke="#ef4444" stroke-width="1.8" />
            <text x="7.5" y="21.5" fill="#ef4444" font-size="3.5" font-weight="600" font-family="system-ui">Restricted Zone</text>
          </svg>
          <div class="bbox-layer" id="bbox-virtual_fence"></div>
        </div>
      </div>

      <div class="card" onclick="openEnlarged('vehicle_detection', 'Cam 2')">
        <div class="card-header">
          <h2>Cam 2</h2>
        </div>
        <div class="feed-wrapper">
          <img src="/feed/vehicle_detection" alt="Cam 2 Feed">
          <div class="bbox-layer" id="bbox-vehicle_detection"></div>
        </div>
      </div>

      <div class="card" onclick="openEnlarged('human_detection', 'Cam 3')">
        <div class="card-header">
          <h2>Cam 3</h2>
        </div>
        <div class="feed-wrapper">
          <img src="/feed/human_detection" alt="Cam 3 Feed">
          <div class="bbox-layer" id="bbox-human_detection"></div>
        </div>
      </div>

      <div class="card" onclick="openEnlarged('anpr', 'Cam 4')">
        <div class="card-header">
          <h2>Cam 4</h2>
        </div>
        <div class="feed-wrapper">
          <img src="/feed/anpr" alt="Cam 4 Feed">
          <div class="bbox-layer" id="bbox-anpr"></div>
        </div>
      </div>
    </div>
  </main>

  <div class="drawer-overlay" id="drawerOverlay" onclick="closeEvents()"></div>
  <div class="drawer" id="eventsDrawer">
    <div class="drawer-header">
      <h3>Event Log</h3>
      <button class="close-btn" onclick="closeEvents()">&times;</button>
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
        <img id="enlargedImg" src="" alt="Enlarged Feed">
        <div id="modalFenceLayer"></div>
        <div class="bbox-layer" id="bbox-modal"></div>
      </div>
    </div>
  </div>

  <script>
    const eventsDrawer = document.getElementById('eventsDrawer');
    const drawerOverlay = document.getElementById('drawerOverlay');
    const modalOverlay = document.getElementById('modalOverlay');
    const modalTitle = document.getElementById('modalTitle');
    const enlargedImg = document.getElementById('enlargedImg');
    const eventBadge = document.getElementById('eventBadge');
    const eventsList = document.getElementById('eventsList');
    const modalFenceLayer = document.getElementById('modalFenceLayer');
    let activeEnlargedKey = null;

    document.getElementById('openEventsBtn').addEventListener('click', () => {
      eventsDrawer.classList.add('active');
      drawerOverlay.classList.add('active');
    });

    function closeEvents() {
      eventsDrawer.classList.remove('active');
      drawerOverlay.classList.remove('active');
    }

    function openEnlarged(feedKey, title) {
      activeEnlargedKey = feedKey;
      modalTitle.textContent = title;
      enlargedImg.src = '/feed/' + feedKey;
      if (feedKey === 'virtual_fence') {
        modalFenceLayer.innerHTML = `
          <svg class="fence-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
            <polygon points="6.25,16.67 50,16.67 50,95.83 6.25,95.83" fill="rgba(239, 68, 68, 0.18)" stroke="#ef4444" stroke-width="1.8" />
            <text x="7.5" y="21.5" fill="#ef4444" font-size="3.5" font-weight="600" font-family="system-ui">Restricted Zone</text>
          </svg>
        `;
      } else {
        modalFenceLayer.innerHTML = '';
      }
      modalOverlay.classList.add('active');
    }

    function closeEnlarged(e) {
      if (!e || e.target === modalOverlay || e.target.classList.contains('close-btn')) {
        modalOverlay.classList.remove('active');
        enlargedImg.src = '';
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
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
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
        const label = isPlate ? escapeHtml(d.label.replace(/^PLATE:\s*/i, '')) : escapeHtml(d.label);
        return `
        <div class="bbox-tag" style="left:${d.x1 * 100}%; top:${d.y1 * 100}%; width:${d.w * 100}%; height:${d.h * 100}%; border-color:${d.color}; border-width:${borderPx};">
          <span class="bbox-label" style="background:${d.color}; ${isPlate ? 'font-size:13px; padding:2px 7px; letter-spacing:0.05em;' : ''}">${label}</span>
        </div>`;
      }).join('');

      // Big plate banner in top-left corner
      if (plateDet) {
        const plateNum = escapeHtml(plateDet.label.replace(/^PLATE:\s*/i, ''));
        html += `<div class="plate-banner">${plateNum}</div>`;
      }

      container.innerHTML = html;
    }

    async function refreshStatus() {
      try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        const events = data.events || [];
        eventBadge.textContent = events.length;
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

        const detections = data.detections || {};
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
    setInterval(refreshStatus, 100);
  </script>
</body>
</html>
"""
