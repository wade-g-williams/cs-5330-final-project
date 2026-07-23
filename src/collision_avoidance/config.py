"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose:
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUNRGBD_ROOT = PROJECT_ROOT / "data" / "sunrgbd" / "sunrgbd_trainval"

# COCO class name --> SUN RGB-D benchmark label. Only 5 of SUN-RGB-D's 10 clsses
# have a COCO equivalent; the other five (desk, dresser, night_stand, bookshelf,
# bathtub) have none and can only appear as "unknown" clusters.
COCO_TO_SUNRGBD = {
    "bed": "bed",
    "dining table": "table",
    "couch": "sofa",
    "chair": "chair",
    "toilet": "toilet",
}

DETECTOR_CONF = 0.25                # confidence threshold
DETECTOR_MODEL = "yolo11x.pt"       # x-large: best accuracy, fast on GPU; drop to m/s for faster iteration