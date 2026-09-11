import cv2
import numpy as np
import logging

try:
    from ultralytics import YOLO
except ImportError:
    logging.error("ultralytics library is required. Please install it using: pip install ultralytics opencv-python")
    YOLO = None

class VirtualFenceEngine:
    """
    A modular, drop-in engine for Virtual Fence Intrusion Detection in CCTV streams.
    Can be easily integrated into any existing web server, desktop app, or pipeline.
    """
    def __init__(self, model_path="yolo11n.pt", fence_type="line", fence_coords=None, target_classes=None):
        """
        Initializes the analytics engine.
        
        :param model_path: Path to the YOLO weights (will download yolov8n.pt automatically if not found).
        :param fence_type: 'line' or 'polygon'
        :param fence_coords: 
            For 'line': tuple of two points ((x1, y1), (x2, y2))
            For 'polygon': list of points [(x1, y1), (x2, y2), (x3, y3), ...]
        :param target_classes: List of COCO class IDs to track (e.g., [0] for person, [2, 3, 5, 7] for vehicles).
        """
        self.logger = logging.getLogger(__name__)
        if YOLO is None:
            raise RuntimeError("YOLO is not installed.")
            
        # Load object detection & tracking model
        self.model = YOLO(model_path)
        
        self.fence_type = fence_type
        # Line mode accepts two user-selected points as the fence endpoints.
        if fence_type == "line":
            if fence_coords is not None and len(fence_coords) != 2:
                raise ValueError("A line fence requires exactly two points")
            self.fence_coords = tuple(tuple(point) for point in fence_coords) if fence_coords else None
        elif fence_type == "polygon":
            self.fence_coords = np.array(fence_coords, np.int32) if fence_coords else np.array([[100, 200], [500, 200], [600, 400], [50, 400]], np.int32)
        else:
            raise ValueError("fence_type must be 'line' or 'polygon'")

        # Default to detecting 'person' (0) if none provided
        self.target_classes = target_classes if target_classes is not None else [0]
        
        # Memory map to store previous positions of tracked objects
        self.tracked_objects = {}
        
    def _check_line_intersect(self, p1, p2, p3, p4):
        """Math helper: Checks if line segment (p1->p2) intersects with (p3->p4)."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def set_fence(self, fence_coords):
        """Set the two endpoints of the line fence."""
        if len(fence_coords) != 2:
            raise ValueError("A line fence requires exactly two points")
        self.fence_coords = tuple(tuple(point) for point in fence_coords)

    def _crossed_line_fence(self, previous, current, fence_start, fence_end):
        """Return whether movement crossed the selected fence segment."""
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
        """Math helper: Checks if a point is inside the defined polygon using OpenCV."""
        # cv2.pointPolygonTest returns >0 if inside, 0 if on edge, <0 if outside
        result = cv2.pointPolygonTest(self.fence_coords, (float(point[0]), float(point[1])), False)
        return result >= 0

    def process_frame(self, frame):
        """
        Main pipeline function to process a frame, run tracking, and evaluate logic.
        
        :param frame: A numpy array representing a BGR image (e.g., from cv2.VideoCapture or a camera stream).
        :return: (annotated_frame, list_of_alerts)
        """
        alerts = []
        current_detections = []
        
        # 1. Run YOLO Tracking (persist=True enables DeepSORT/ByteTrack internally)
        results = self.model.track(frame, persist=True, classes=self.target_classes, verbose=False)
        
        # 2. Evaluate Intrusion Logic
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # Bottom-center of the bounding box (object's "feet")
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

                # Record alert if intrusion detected
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

                # Update history for next frame
                self.tracked_objects[track_id] = curr_pos

        return frame, alerts, current_detections

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
