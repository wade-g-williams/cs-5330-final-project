"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Headless Stage 1 diagnostic. Runs the real Detector on several frames and
SAVES annotated jpgs (instead of cv2.imshow), so it works over SSH / without a GUI and
the images double as report figures. The point is to judge Stage 1 against stage1.md
§8 on representative, in-distribution frames -- not just the OOD showroom at index 0.
"""

import argparse
import os

import cv2

from collision_avoidance import config
from collision_avoidance.datasets.sunrgbd import SunRGBDLoader
from collision_avoidance.detection import Detector


def annotate(frame, detections):
    # Same convention as the smoke test: green box = a scored SUN RGB-D class,
    # orange box = an obstacle outside the 10 scored classes.
    bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        color = (0, 200, 0) if d.is_dataset_class else (0, 165, 255)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, f"{d.label} {d.score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 100, 1000, 3000],
                    help="frame indices to run (index = id - 1; e.g. 1000 -> 001001.jpg)")
    ap.add_argument("--model", default=config.DETECTOR_MODEL)
    ap.add_argument("--conf", type=float, default=config.DETECTOR_CONF)
    ap.add_argument("--out", default="out/stage1_diag")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    loader = SunRGBDLoader(config.SUNRGBD_ROOT)
    detector = Detector(args.model, conf=args.conf, class_map=config.COCO_TO_SUNRGBD)  # build ONCE
    model_stem = os.path.splitext(os.path.basename(args.model))[0]

    for idx in args.indices:
        frame = loader[idx]
        dets = detector.detect(frame)
        print(f"\nframe {frame.frame_id} (index {idx}): {len(dets)} detections")
        for d in dets:
            tag = d.label if d.is_dataset_class else f"{d.coco_name} (not a scored class)"
            print(f"  {tag:24s} score={d.score:.2f}  bbox={tuple(round(v) for v in d.bbox)}")
        path = os.path.join(args.out, f"{frame.frame_id}_{model_stem}.jpg")
        cv2.imwrite(path, annotate(frame, dets))
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
