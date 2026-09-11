from ultralytics import YOLO
import cv2
import os

model = YOLO("yolo11n.pt")
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# Use webcam by default. For a file, set SOURCE to a real video or image path.
SOURCE = r"C:\Users\ishan\Desktop\SIH\Drishti\sample.jpeg"  # or r"C:\path\to\video.mp4" or r"C:\path\to\image.jpg"


def detect_vehicles(frame):
    results = model(frame, conf=0.4, verbose=False)
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            if cls_name.lower() not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append((cls_name, conf, (x1, y1, x2, y2)))

    return detections


# Work with webcam by default; if a file path is supplied and exists, use that instead.
source = SOURCE

if isinstance(source, str) and not os.path.isfile(source):
    print(f"File not found: {source}. Falling back to webcam.")
    source = 0

if isinstance(source, str):
    ext = os.path.splitext(source)[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {source}")

        detections = detect_vehicles(frame)
        for cls_name, conf, (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{cls_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Vehicle Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for cls_name, conf, (x1, y1, x2, y2) in detect_vehicles(frame):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{cls_name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Vehicle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
else:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check your camera permissions or device index.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for cls_name, conf, (x1, y1, x2, y2) in detect_vehicles(frame):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{cls_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Vehicle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()