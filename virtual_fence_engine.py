import cv2
import numpy as np
import logging

try:
    from ultralytics import YOLO
except ImportError:
    logging.error("ultralytics library is required. Please install it using: pip install ultralytics opencv-python")
    YOLO = None

class SurveillanceTracker:
    """
    Lightweight, deterministic spatial tracker for CCTV surveillance.
    Ensures persistent, low-integer IDs (#1, #2...) that never jump or explode.
    """
    def __init__(self, max_lost_frames=4, iou_thresh=0.2, dist_thresh=80):
        self.max_lost = max_lost_frames
        self.iou_thresh = iou_thresh
        self.dist_thresh = dist_thresh
        self.tracks = {}

    def reset(self):
        self.tracks.clear()

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return interArea / float(boxAArea + boxBArea - interArea)

    @staticmethod
    def _dist(boxA, boxB):
        cAx, cAy = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
        cBx, cBy = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2
        return ((cAx - cBx)**2 + (cAy - cBy)**2)**0.5

    @staticmethod
    def _nms(candidates, iou_thresh=0.35):
        if not candidates:
            return []
        sorted_dets = sorted(candidates, key=lambda x: x.get("conf", 0.0), reverse=True)
        keep = []
        for det in sorted_dets:
            suppress = False
            for k in keep:
                if SurveillanceTracker._iou(det["bbox"], k["bbox"]) > iou_thresh:
                    suppress = True
                    break
            if not suppress:
                keep.append(det)
        return keep

    def update(self, detected_boxes):
        # Apply NMS to eliminate overlapping sub-boxes on the same person
        clean_boxes = self._nms(detected_boxes, 0.35)
        matched = []
        unmatched_dets = list(range(len(clean_boxes)))
        unmatched_tracks = list(self.tracks.keys())

        for tid in list(unmatched_tracks):
            t_box = self.tracks[tid]["bbox"]
            best_det_idx = None
            best_score = -1.0

            for d_idx in unmatched_dets:
                d_box = clean_boxes[d_idx]["bbox"]
                iou = self._iou(t_box, d_box)
                dist = self._dist(t_box, d_box)
                score = iou if iou >= self.iou_thresh else (1.0 / (1.0 + dist) if dist <= self.dist_thresh else 0.0)

                if score > best_score and score > 0.15:
                    best_score = score
                    best_det_idx = d_idx

            if best_det_idx is not None:
                d = dict(clean_boxes[best_det_idx])
                d["track_id"] = tid
                self.tracks[tid] = {"bbox": d["bbox"], "lost": 0, "data": d}
                matched.append(d)
                unmatched_dets.remove(best_det_idx)
                unmatched_tracks.remove(tid)

        for d_idx in unmatched_dets:
            d = dict(clean_boxes[d_idx])
            used_ids = set(self.tracks.keys())
            cand = 1
            while cand in used_ids:
                cand += 1
            tid = cand
            d["track_id"] = tid
            self.tracks[tid] = {"bbox": d["bbox"], "lost": 0, "data": d}
            matched.append(d)

        for tid in list(unmatched_tracks):
            self.tracks[tid]["lost"] += 1
            if self.tracks[tid]["lost"] > self.max_lost:
                del self.tracks[tid]

        return matched


class VirtualFenceEngine:
    """
    A modular, drop-in engine for Virtual Fence Intrusion Detection in CCTV streams.
    Can be easily integrated into any existing web server, desktop app, or pipeline.
    """
    def __init__(self, model_path="yolo11n.pt", fence_type="polygon", fence_coords=None, target_classes=None):
        self.logger = logging.getLogger(__name__)
        if YOLO is None:
            raise RuntimeError("YOLO is not installed.")
            
        self.model = YOLO(model_path)
        self.fence_type = fence_type
        if fence_type == "line":
            if fence_coords is not None and len(fence_coords) != 2:
                raise ValueError("A line fence requires exactly two points")
            self.fence_coords = tuple(tuple(point) for point in fence_coords) if fence_coords else None
        elif fence_type == "polygon":
            self.fence_coords = np.array(fence_coords, np.int32) if fence_coords else np.array([[90, 125], [250, 125], [250, 235], [90, 235]], np.int32)
        else:
            raise ValueError("fence_type must be 'line' or 'polygon'")

        self.target_classes = target_classes if target_classes is not None else [0]
        self.tracked_objects = {}
        self.tracker = SurveillanceTracker(max_lost_frames=4, iou_thresh=0.2, dist_thresh=80)
        
    def _check_line_intersect(self, p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def set_fence(self, fence_coords):
        if len(fence_coords) != 2:
            raise ValueError("A line fence requires exactly two points")
        self.fence_coords = tuple(tuple(point) for point in fence_coords)

    def _crossed_line_fence(self, previous, current, fence_start, fence_end):
        def side_of_line(point):
            return (
                (fence_end[0] - fence_start[0]) * (point[1] - fence_start[1])
                - (fence_end[1] - fence_start[1]) * (point[0] - fence_start[0])
            )

        previous_side = side_of_line(previous)
        current_side = side_of_line(current)

        if previous_side == 0 or current_side == 0:
            side_changed = previous_side != current_side
        else:
            side_changed = (previous_side < 0) != (current_side < 0)

        return side_changed and self._check_line_intersect(
            previous,
            current,
            fence_start,
            fence_end,
        )

    def _check_polygon_inside(self, point):
        result = cv2.pointPolygonTest(self.fence_coords, (float(point[0]), float(point[1])), False)
        return result >= 0

    def process_frame(self, frame):
        alerts = []
        current_detections = []
        
        # 1. Detect target objects with confidence threshold
        try:
            results = self.model(frame, classes=self.target_classes, conf=0.40, verbose=False)[0]
        except Exception:
            results = None

        raw_candidates = []
        if results is not None and results.boxes is not None:
            for b in results.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                # Filter out tiny noise (e.g. wheels/pedals misdetected as persons)
                if (y2 - y1) >= 35:
                    raw_candidates.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": float(b.conf[0]),
                        "class_id": int(b.cls[0])
                    })

        # 2. Track with stable, low-number persistent IDs (#1, #2...)
        tracked = self.tracker.update(raw_candidates)

        # 3. Evaluate Intrusion Logic
        for d in tracked:
            x1, y1, x2, y2 = d["bbox"]
            track_id = d["track_id"]
            class_id = d.get("class_id", 0)
            conf = d.get("conf", 0.8)
            curr_pos = (int((x1 + x2) / 2), y2)
            is_intruder = False

            if self.fence_type == "line":
                if track_id in self.tracked_objects:
                    prev_pos = self.tracked_objects[track_id]
                    if self._crossed_line_fence(
                        prev_pos,
                        curr_pos,
                        self.fence_coords[0],
                        self.fence_coords[1],
                    ):
                        is_intruder = True
            elif self.fence_type == "polygon":
                if self._check_polygon_inside(curr_pos):
                    is_intruder = True

            if is_intruder:
                alerts.append({
                    "event_type": "INTRUSION_DETECTED",
                    "track_id": int(track_id),
                    "class_id": int(class_id),
                    "confidence": float(conf),
                    "location": curr_pos
                })

            current_detections.append({
                "bbox": (x1, y1, x2, y2),
                "track_id": int(track_id),
                "class_id": int(class_id),
                "confidence": float(conf),
                "is_intruder": is_intruder
            })

            self.tracked_objects[track_id] = curr_pos

        return frame, alerts, current_detections

    def reset(self):
        self.tracked_objects.clear()
        self.tracker.reset()

def select_fence_points(frame):
    """Let the user click two points on a frame to define a line fence."""
    points = []
    window_name = "Select Virtual Fence"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while len(points) < 2:
        display = frame.copy()
        cv2.putText(
            display,
            "Click two points for the fence; press ESC to cancel",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        for point in points:
            cv2.circle(display, point, 6, (0, 255, 255), -1)

        if len(points) == 2:
            cv2.line(display, points[0], points[1], (255, 0, 0), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("Fence selection cancelled")

    cv2.destroyWindow(window_name)
    return points


# ==============================================================================
# USAGE EXAMPLE (Can be run directly to test via webcam or a dummy feed)
# ==============================================================================
if __name__ == "__main__":
    print("Starting Virtual Fence Engine Demo...")

    # Open Webcam (0) or specify a video file e.g., 'border_feed.mp4'
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video feed or webcam.")
        exit()

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first video frame.")
        cap.release()
        exit()

    try:
        fence_points = select_fence_points(first_frame)
    except RuntimeError as error:
        print(error)
        cap.release()
        cv2.destroyAllWindows()
        exit()

    engine = VirtualFenceEngine(
        model_path="yolo11n.pt", 
        fence_type="line",
        fence_coords=fence_points,
        target_classes=[0],  # 0 = Person
    )

    print("Fence selected. Engine running! Press 'q' to quit.")
    
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break
            
        annotated_frame, frame_alerts = engine.process_frame(current_frame)
        
        # If alerts were detected, handle them (Send API request, play sound, write to DB)
        if frame_alerts:
            for alert in frame_alerts:
                print(f"[ALERT] Intrusion! Details: {alert}")
        
        # Display the output for the demo
        cv2.imshow("Intelligent Border Surveillance", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
