"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: 
"""

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from .frame import Frame


@dataclass
class Detection:
    """One detected object(fusion)."""

    bbox: tuple[float, float, float, float]     # (x1, y1, x2, y2) in pixels, in the original image
    score: float                                # confidence in [0, 1]
    coco_name: str                              # the raw class the detector reported (e.g. "couch")
    label: str                                  # dataset label if COCO maps to one, else == coco_name
    is_dataset_class: bool                      # True if 'label' is one of the benchmark's scored classes


class Detector:
    def __init__(self, model_path: str = "yolo11n.pt", conf: float = 0.25, class_map: dict | None = None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_map = class_map or {}

    def detect(self, frame: Frame) -> list[Detection]:
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)

        result = self.model(bgr, conf=self.conf, verbose=False)[0]
        names = result.names

        detections: list[Detection] = []
        for box, score, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        ):
            coco_name = names[cls]
            label = self.class_map.get(coco_name, coco_name)
            detections.append(Detection(
                bbox=tuple(float(v) for v in box),
                score=float(score),
                coco_name=coco_name,
                label=label,
                is_dataset_class=coco_name in self.class_map,
            ))
        return detections
