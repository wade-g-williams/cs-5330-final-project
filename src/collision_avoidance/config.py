"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Single source of truth for paths, class maps, and model settings. Every path is
absolute (anchored to PROJECT_ROOT, not the cwd) so scripts behave the same from anywhere.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The datasets total ~47 GB. Set CA_DATA_DIR to keep them on an external drive or a shared
# mount instead, rather than editing this file and risking committing a machine-specific path.
DATA_DIR = Path(os.environ.get("CA_DATA_DIR") or PROJECT_ROOT / "data")
MODELS_DIR = Path(os.environ.get("CA_MODELS_DIR") or PROJECT_ROOT / "models")

SUNRGBD_ROOT = DATA_DIR / "sunrgbd" / "sunrgbd_trainval"   # image/ depth/ calib/, flat ids
KITTI_ROOT = DATA_DIR / "kitti"                            # training/{image_2,calib,label_2}

# COCO class name --> SUN RGB-D benchmark label. Only 5 of SUN RGB-D's 10 classes
# have a COCO equivalent; the other five (desk, dresser, night_stand, bookshelf,
# bathtub) have none and can only appear as "unknown" clusters.
COCO_TO_SUNRGBD = {
    "bed": "bed",
    "dining table": "table",
    "couch": "sofa",
    "chair": "chair",
    "toilet": "toilet",
}

# COCO class name --> KITTI scored class. KITTI also labels Van/Truck/Tram/Misc, but the
# benchmark scores only these three.
COCO_TO_KITTI = {
    "car": "Car",
    "person": "Pedestrian",
    "bicycle": "Cyclist",
}

# Which COCO->label map a dataset uses. Scripts look up --dataset here.
CLASS_MAPS = {"sunrgbd": COCO_TO_SUNRGBD, "kitti": COCO_TO_KITTI}

DETECTOR_CONF = 0.25                              # confidence threshold
DETECTOR_MODEL = str(MODELS_DIR / "yolo11x.pt")   # swap stem to yolo11m/s/n for faster iteration

# Depth Anything V2, metric variants (the transformers-native "-hf" builds). Domain-specific
# and NOT interchangeable -- crossing them breaks metric scale.
DEPTH_MODEL_OUTDOOR = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"   # VKITTI, 80 m
DEPTH_MODEL_INDOOR = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"     # Hypersim, 20 m
DEPTH_MAX_M = {"outdoor": 80.0, "indoor": 20.0}

# Stage 3 -- RANSAC ground-plane fit (ground.py). Distances are meters.
GROUND_DIST_THRESH = 0.04       # inlier band around the plane
GROUND_MAX_TILT_DEG = 30.0      # reject planes whose normal isn't ~vertical (Y axis)
GROUND_RANSAC_ITERS = 1000      # max iterations (adapts down once a good fit is found)
GROUND_RANSAC_SEED = 0          # seed the sampling so report numbers reproduce

# Stage 4 -- Euclidean clustering (cluster.py). Distances are meters.
CLUSTER_EPS = 0.3               # neighbour radius linking points into one obstacle
CLUSTER_MIN_POINTS = 30         # clusters smaller than this are dropped as noise
