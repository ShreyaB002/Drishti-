"""
examples/run_on_video.py
==========================
Minimal example showing how ANPR is driven frame-by-frame over a
video, using temporal fusion for a tracked vehicle.

NOTE: This demo uses a placeholder `dummy_vehicle_bbox()` in place of
the real Vehicle Detection Engine, since that module is out of scope
here. In production, `vehicle_bbox` and `vehicle_id` for each frame
come from the Vehicle Detection Engine's per-frame output.

Usage:
    python examples/run_on_video.py path/to/video.mp4
"""

import sys
import json
import cv2

from anpr import ANPREngine


def dummy_vehicle_bbox(frame):
    """
    Placeholder standing in for the Vehicle Detection Engine.
    Replace this call with the real bbox supplied by that module.
    """
    h, w = frame.shape[:2]
    return (int(w * 0.3), int(h * 0.4), int(w * 0.7), int(h * 0.9))


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_on_video.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)

    engine = ANPREngine()
    vehicle_id = "V_TRACK_1"  # would come from the Vehicle Detection Engine's tracker
    camera_id = "CAM_07"

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        vehicle_bbox = dummy_vehicle_bbox(frame)  # <- replace with real detection

        per_frame_result = engine.process(
            frame=frame,
            vehicle_bbox=vehicle_bbox,
            vehicle_id=vehicle_id,
            camera_id=camera_id,
            timestamp=f"frame_{frame_idx}",
            use_temporal_fusion=True,
        )

        if per_frame_result["status"] != "NO_PLATE_DETECTED":
            print(f"[frame {frame_idx}] {per_frame_result['plate_number']} "
                  f"(conf={per_frame_result['confidence']}, status={per_frame_result['status']})")

    cap.release()

    # Final, fully fused reading for the tracked vehicle once it has left the frame.
    final_result = engine.get_fused_result(vehicle_id)
    print("\nFinal fused result:")
    print(json.dumps(final_result, indent=2, default=str))


if __name__ == "__main__":
    main()
