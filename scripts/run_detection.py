"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Run the detector over a few frames and save annotated jpgs. Headless on purpose,
so it works over SSH and the images double as report figures.

    python scripts/run_detection.py --dataset sunrgbd --indices 0 100 1000 3000
"""

import argparse
import os

import cv2

from collision_avoidance import config, viz
from collision_avoidance.datasets import DATASETS, get_loader
from collision_avoidance.detection import Detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="sunrgbd")
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 100, 1000, 3000],
                    help="frame indices (index = id - 1; 1000 -> 001001.jpg)")
    ap.add_argument("--model", default=config.DETECTOR_MODEL)
    ap.add_argument("--conf", type=float, default=config.DETECTOR_CONF)
    ap.add_argument("--out", default="out/detections")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    loader = get_loader(args.dataset)
    detector = Detector(args.model, conf=args.conf,                     # build ONCE, reuse
                        class_map=config.CLASS_MAPS[args.dataset])
    model_stem = os.path.splitext(os.path.basename(args.model))[0]

    for index in args.indices:
        frame = loader[index]
        detections = detector.detect(frame)
        print(f"\nframe {frame.frame_id} (index {index}): {len(detections)} detections")
        for d in detections:
            tag = d.label if d.is_dataset_class else f"{d.coco_name} (not a scored class)"
            print(f"  {tag:24s} score={d.score:.2f}  bbox={tuple(round(v) for v in d.bbox)}")

        path = os.path.join(args.out, f"{args.dataset}_{frame.frame_id}_{model_stem}.jpg")
        cv2.imwrite(path, viz.draw_detections(frame, detections))
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
