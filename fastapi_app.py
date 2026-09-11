import threading
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from virtual_fence_engine import VirtualFenceEngine

ROOT = Path(__file__).resolve().parent
MODEL_PATH = str(ROOT / "yolo11n.pt")
CAMERA_SOURCE = 0

app = FastAPI(title="Drishti Virtual Fence")
engine = VirtualFenceEngine(
    model_path=MODEL_PATH,
    fence_type="line",
    target_classes=[0],
)
camera = cv2.VideoCapture(CAMERA_SOURCE)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_lock = threading.Lock()
latest_jpeg = None
latest_status = {
    "fence_selected": False,
    "alerts": [],
    "timestamp": None,
}
fence_lock = threading.Lock()
fence_normalized = None
stop_event = threading.Event()


class FenceRequest(BaseModel):
    points: list[list[float]] = Field(..., min_length=2, max_length=2)


def normalized_to_pixels(points, frame):
    height, width = frame.shape[:2]
    return [
        (round(max(0.0, min(1.0, point[0])) * width),
         round(max(0.0, min(1.0, point[1])) * height))
        for point in points
    ]


def camera_worker():
    global latest_jpeg, latest_status

    while not stop_event.is_set():
        success, frame = camera.read()
        if not success:
            time.sleep(0.2)
            continue

        with fence_lock:
            selected_fence = fence_normalized

        if selected_fence is None:
            annotated = frame
            alerts = []
        else:
            engine.set_fence(normalized_to_pixels(selected_fence, frame))
            annotated, alerts = engine.process_frame(frame)

        success, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), 82],
        )
        if success:
            with frame_lock:
                latest_jpeg = buffer.tobytes()
                latest_status = {
                    "fence_selected": selected_fence is not None,
                    "alerts": alerts,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }


def generate_mjpeg():
    while not stop_event.is_set():
        with frame_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


@app.on_event("startup")
def start_camera_worker():
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions or CAMERA_SOURCE.")
    threading.Thread(target=camera_worker, daemon=True).start()


@app.on_event("shutdown")
def stop_camera_worker():
    stop_event.set()
    camera.release()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
def status():
    with frame_lock:
        return latest_status


@app.post("/api/fence")
def set_fence(request: FenceRequest):
    points = request.points
    if any(len(point) != 2 for point in points):
        raise HTTPException(status_code=400, detail="Each fence point must contain x and y")

    with fence_lock:
        global fence_normalized
        fence_normalized = points

    with frame_lock:
        latest_status["fence_selected"] = True
        latest_status["alerts"] = []

    return {"fence_selected": True, "points": points}


@app.delete("/api/fence")
def clear_fence():
    with fence_lock:
        global fence_normalized
        fence_normalized = None
        engine.tracked_objects.clear()

    with frame_lock:
        latest_status["fence_selected"] = False
        latest_status["alerts"] = []

    return {"fence_selected": False}


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drishti Virtual Fence</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #0b1220; color: #edf6ff; }
    main { width: min(100% - 32px, 1100px); margin: 24px auto; }
    h1 { margin: 0 0 8px; }
    p { color: #a8b8c8; }
    .stage { position: relative; width: 100%; background: #000; }
    .stage img { display: block; width: 100%; height: auto; }
    .stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; }
    .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-top: 14px; }
    button { border: 0; border-radius: 6px; padding: 10px 14px; cursor: pointer; background: #2563eb; color: white; }
    button.secondary { background: #334155; }
    #status { color: #facc15; }
  </style>
</head>
<body>
  <main>
    <h1>Virtual Fence</h1>
    <p>Click two points on the video to create the fence. Crossing it in either direction raises an alert.</p>
    <div class="stage">
      <img id="feed" src="/video_feed" alt="Live camera feed">
      <canvas id="fence"></canvas>
    </div>
    <div class="controls">
      <button id="clear" class="secondary">Clear fence</button>
      <span id="status">Select two points.</span>
    </div>
  </main>
  <script>
    const feed = document.getElementById('feed');
    const canvas = document.getElementById('fence');
    const context = canvas.getContext('2d');
    const status = document.getElementById('status');
    let points = [];

    function resizeCanvas() {
      canvas.width = feed.clientWidth;
      canvas.height = feed.clientHeight;
      drawFence();
    }

    function drawFence() {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = '#facc15';
      context.strokeStyle = '#facc15';
      context.lineWidth = 3;
      points.forEach(point => {
        context.beginPath();
        context.arc(point.x, point.y, 6, 0, Math.PI * 2);
        context.fill();
      });
      if (points.length === 2) {
        context.beginPath();
        context.moveTo(points[0].x, points[0].y);
        context.lineTo(points[1].x, points[1].y);
        context.stroke();
      }
    }

    canvas.addEventListener('click', async event => {
      if (points.length === 2) return;
      const bounds = canvas.getBoundingClientRect();
      const point = { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
      points.push(point);
      drawFence();
      status.textContent = points.length === 1 ? 'Select the second point.' : 'Sending fence...';

      if (points.length === 2) {
        const normalized = points.map(point => [point.x / canvas.clientWidth, point.y / canvas.clientHeight]);
        const response = await fetch('/api/fence', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ points: normalized })
        });
        status.textContent = response.ok ? 'Fence active. Monitoring crossings.' : 'Could not set fence.';
      }
    });

    document.getElementById('clear').addEventListener('click', async () => {
      await fetch('/api/fence', { method: 'DELETE' });
      points = [];
      drawFence();
      status.textContent = 'Select two points.';
    });

    feed.addEventListener('load', resizeCanvas);
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
  </script>
</body>
</html>
"""
