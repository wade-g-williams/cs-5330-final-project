"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Empirically rule out the stage1.md §9 "silent RGB/BGR bug." Runs raw Ultralytics
two ways on one in-distribution frame -- the CORRECT BGR that detection.py feeds, and the
WRONG un-converted RGB -- and prints score-sorted detections for each. If CORRECT finds the
couch with higher confidence / saner classes than WRONG, the conversion matters AND the
current detection.py is on the right side of it. Proof, not assertion.
"""

import argparse

import cv2
from ultralytics import YOLO

from collision_avoidance import config
from collision_avoidance.datasets.sunrgbd import SunRGBDLoader


def summarize(result, title):
    names = result.names
    rows = sorted(
        ((names[int(c)], float(s))
         for c, s in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy())),
        key=lambda r: -r[1],
    )
    print(f"\n{title}: {len(rows)} detections")
    for name, score in rows:
        print(f"  {name:20s} {score:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=1000, help="in-distribution frame (1000 -> 001001.jpg, the couch)")
    ap.add_argument("--model", default="yolo11m.pt")
    args = ap.parse_args()

    frame = SunRGBDLoader(config.SUNRGBD_ROOT)[args.index]
    model = YOLO(args.model)

    correct = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)   # what detection.py feeds (BGR)
    wrong = frame.rgb                                      # RGB handed in as if it were BGR

    summarize(model(correct, conf=0.10, verbose=False)[0], "CORRECT (BGR, current code)")
    summarize(model(wrong,   conf=0.10, verbose=False)[0], "WRONG (RGB, the §9 bug)")


if __name__ == "__main__":
    main()
