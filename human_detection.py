import time
import threading
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)

# Use "0" for the laptop webcam.
# Replace with an RTSP URL for an IP/CCTV camera:
# CAMERA_SOURCE = "rtsp://username:password@ip-address:554/stream"
CAMERA_SOURCE = 0

# Lightweight pose model. It downloads automatically on the first run.
MODEL_PATH = "yolo11n-pose.pt"

# Minimum detection confidence.
CONFIDENCE = 0.45

# Average pixel displacement of visible joints used to classify movement.
STILL_THRESHOLD = 4.0
FAST_MOTION_THRESHOLD = 15.0

# COCO pose skeleton: zero-based keypoint indexes.
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

model = YOLO(MODEL_PATH)
camera = cv2.VideoCapture(CAMERA_SOURCE)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

latest_jpeg = None
latest_status = {
    "humans_detected": 0,
    "people": [],
    "timestamp": None
}

frame_lock = threading.Lock()
previous_keypoints = {}
motion_history = {}
stop_event = threading.Event()


def calculate_motion(track_id, points, valid):
    """Return average visible-joint displacement for one person."""
    valid_points = points[valid]

    if len(valid_points) == 0:
        return 0.0

    current = valid_points.astype(np.float32)
    previous = previous_keypoints.get(track_id)

    if previous is None or len(previous) != len(current):
        displacement = 0.0
    else:
        displacement = float(np.mean(np.linalg.norm(current - previous, axis=1)))

    previous_keypoints[track_id] = current.copy()

    if track_id not in motion_history:
        motion_history[track_id] = deque(maxlen=8)

    motion_history[track_id].append(displacement)
    return float(np.mean(motion_history[track_id]))


def get_motion_label(motion_score):
    if motion_score < STILL_THRESHOLD:
        return "STILL", (150, 150, 150)
    if motion_score < FAST_MOTION_THRESHOLD:
        return "MOVING", (0, 255, 0)
    return "FAST MOTION", (0, 165, 255)


def draw_pose(frame, points, confidence, track_id, bbox):
    """Draw green skeleton dots/lines and return movement information."""
    valid = confidence > 0.35
    motion_score = calculate_motion(track_id, points, valid)
    motion_label, motion_color = get_motion_label(motion_score)

    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(frame, (x1, y1), (x2, y2), motion_color, 2)

    label = f"ID {track_id} | {motion_label} | Motion: {motion_score:.1f}px"
    label_y = max(25, y1 - 10)

    cv2.rectangle(
        frame,
        (x1, label_y - 23),
        (min(frame.shape[1] - 5, x1 + 310), label_y + 5),
        (20, 20, 20),
        -1
    )
    cv2.putText(
        frame,
        label,
        (x1 + 5, label_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        motion_color,
        1,
        cv2.LINE_AA
    )

    # Draw skeleton lines first.
    for start_idx, end_idx in SKELETON:
        if valid[start_idx] and valid[end_idx]:
            start = tuple(points[start_idx].astype(int))
            end = tuple(points[end_idx].astype(int))
            cv2.line(frame, start, end, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw visible joints as green circles.
    for index, point in enumerate(points):
        if valid[index]:
            x, y = point.astype(int)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 7, (0, 70, 0), 1)

    return {
        "track_id": int(track_id),
        "motion_status": motion_label,
        "motion_score_px": round(motion_score, 2),
        "bbox": [x1, y1, x2, y2]
    }


def annotate_frame(frame):
    global latest_status

    # persist=True preserves object IDs over successive frames.
    # classes=[0] processes persons only.
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE,
        verbose=False
    )

    annotated = frame.copy()
    people = []

    for result in results:
        if result.boxes is None or result.keypoints is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id
        keypoints_xy = result.keypoints.xy.cpu().numpy()
        keypoints_conf = result.keypoints.conf.cpu().numpy()

        if ids is None:
            continue

        track_ids = ids.int().cpu().tolist()

        for bbox, track_id, points, confidence in zip(
            boxes,
            track_ids,
            keypoints_xy,
            keypoints_conf
        ):
            person_data = draw_pose(
                annotated,
                points,
                confidence,
                track_id,
                bbox
            )
            people.append(person_data)

    latest_status = {
        "humans_detected": len(people),
        "people": people,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    return annotated


def camera_worker():
    global latest_jpeg

    while not stop_event.is_set():
        success, frame = camera.read()

        if not success:
            time.sleep(0.5)
            continue

        processed_frame = annotate_frame(frame)

        success, buffer = cv2.imencode(
            ".jpg",
            processed_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 82]
        )

        if success:
            with frame_lock:
                latest_jpeg = buffer.tobytes()


def generate_mjpeg():
    while True:
        with frame_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )


@app.route("/")
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    worker = threading.Thread(target=camera_worker, daemon=True)
    worker.start()

    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        stop_event.set()
        camera.release()