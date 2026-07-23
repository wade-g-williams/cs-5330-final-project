# scripts/smoke_test_stage1.py
import argparse

import cv2

from collision_avoidance import config
from collision_avoidance.datasets.sunrgbd import SunRGBDLoader
from collision_avoidance.detection import Detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="which frame to run on")
    ap.add_argument("--model", default=config.DETECTOR_MODEL)
    args = ap.parse_args()

    frame = SunRGBDLoader(config.SUNRGBD_ROOT)[args.index]
    detector = Detector(args.model, conf=config.DETECTOR_CONF, class_map=config.COCO_TO_SUNRGBD)
    detections = detector.detect(frame)

    print(f"{len(detections)} detections on frame {frame.frame_id}:")
    for d in detections:
        tag = d.label if d.is_dataset_class else f"{d.coco_name} (not a scored class)"
        print(f"  {tag:24s} score={d.score:.2f}  bbox={tuple(round(v) for v in d.bbox)}")

    # Draw: green box = a scored SUN RGB-D class, orange box = an obstacle outside the 10 classes.
    bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        color = (0, 200, 0) if d.is_dataset_class else (0, 165, 255)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, f"{d.label} {d.score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imshow("stage 1 — detections (green=scored, orange=other obstacle)", bgr)
    print("press any key in the image window to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()