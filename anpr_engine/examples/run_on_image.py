"""
examples/run_on_image.py
=========================
Minimal example showing how the Vehicle Detection Engine's output
(frame + vehicle_bbox) would be handed to the ANPR engine for a
single image, and how the result is consumed.

Usage:
    python examples/run_on_image.py path/to/image.jpg 120 80 640 480
                                     (x1  y1 x2  y2 of the vehicle bbox)
"""

import sys
import json
import cv2

from anpr import ANPREngine


def main():
    if len(sys.argv) < 6:
        print("Usage: python run_on_image.py <image_path> <x1> <y1> <x2> <y2>")
        sys.exit(1)

    image_path = sys.argv[1]
    vehicle_bbox = tuple(int(v) for v in sys.argv[2:6])

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    # In production this bbox (and vehicle_id/camera_id/timestamp) comes
    # from the Vehicle Detection Engine, not from argv.
    engine = ANPREngine()

    result = engine.process(
        frame=frame,
        vehicle_bbox=vehicle_bbox,
        vehicle_id="V_DEMO",
        camera_id="CAM_DEMO",
        timestamp="2026-09-10T10:00:00+05:30",
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
