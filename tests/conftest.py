"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Shared fixtures, and the guard that keeps pytest runnable without the datasets or a
YOLO checkpoint -- a fresh clone should SKIP those tests, not error out on them.
"""

from pathlib import Path

import numpy as np
import pytest

from collision_avoidance import config

# Loaders index thousands of samples, so these are session-scoped and built at most once.


def sunrgbd_is_available() -> bool:
    """True if the SUN RGB-D mirror is extracted where config expects it."""
    root = config.SUNRGBD_ROOT
    return all((root / sub).is_dir() and any((root / sub).iterdir())
               for sub in ("image", "depth", "calib"))


def kitti_is_available() -> bool:
    """True if KITTI's training split is extracted where config expects it."""
    root = config.KITTI_ROOT / "training"
    return all((root / sub).is_dir() and any((root / sub).iterdir())
               for sub in ("image_2", "calib", "label_2"))


@pytest.fixture(scope="session")
def sunrgbd_loader():
    """The real SUN RGB-D loader, or a skip with instructions if the data isn't downloaded."""
    if not sunrgbd_is_available():
        pytest.skip(f"SUN RGB-D not found at {config.SUNRGBD_ROOT} — "
                    f"run `python scripts/download_datasets.py sunrgbd` (~35 GB)")
    from collision_avoidance.datasets.sunrgbd import SunRGBDLoader
    return SunRGBDLoader(config.SUNRGBD_ROOT)


@pytest.fixture(scope="session")
def kitti_loader():
    """KITTI with a STUB depth model.

    Real depth means downloading a 1.4 GB checkpoint and ~1 s of GPU per frame. The calib,
    indexing, and label parsing are what these tests actually exercise, so we inject a
    constant-depth estimator and keep the suite fast. Depth quality is judged by eye via
    scripts/view_frame.py, not asserted here.
    """
    if not kitti_is_available():
        pytest.skip(f"KITTI not found at {config.KITTI_ROOT} — "
                    f"run `python scripts/download_datasets.py kitti` (~12 GB)")
    from collision_avoidance.datasets.kitti import KittiLoader

    def stub_depth(rgb: np.ndarray) -> np.ndarray:
        return np.full(rgb.shape[:2], 10.0, dtype=np.float32)

    return KittiLoader(config.KITTI_ROOT, depth_estimator=stub_depth)


@pytest.fixture(scope="session")
def detector():
    """The real Detector, or a skip if the checkpoint isn't on disk.

    Ultralytics would otherwise download the missing checkpoint (yolo11x.pt is 114 MB), so
    we check first rather than surprise whoever ran the test with a large download.
    """
    model_path = Path(config.DETECTOR_MODEL)
    if not model_path.exists():
        pytest.skip(f"{model_path} not found — fetch it with "
                    f"`yolo predict model=yolo11x.pt project={config.MODELS_DIR}`")
    from collision_avoidance.detection import Detector
    return Detector(config.DETECTOR_MODEL, conf=config.DETECTOR_CONF,
                    class_map=config.COCO_TO_SUNRGBD)
