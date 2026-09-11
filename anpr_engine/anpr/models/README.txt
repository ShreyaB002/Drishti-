Place your trained license-plate detection YOLO weights here as:

    plate_detector.pt

This must be a model trained on a SINGLE class: "license-plate" (or
similar). Do NOT use a general vehicle/COCO-pretrained model here —
this engine expects plate-only detections within an already-cropped
vehicle region.

Quick options to obtain weights:
1. Train your own: use Ultralytics YOLOv8/11 on an annotated Indian
   license-plate dataset (e.g. via Roboflow Universe "license plate"
   datasets), then export the best.pt checkpoint here.
2. Use an existing open license-plate YOLO checkpoint as a starting
   point and fine-tune on Indian plates for best accuracy.

Update `plate_detector_weights` in config.py if you store the weights
at a different path.
