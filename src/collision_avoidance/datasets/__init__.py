"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Dataset loader registry. Scripts resolve --dataset here, so adding a dataset means
adding one entry instead of editing every script.
"""

from .base import DatasetLoader

DATASETS = ("sunrgbd", "kitti")


def get_loader(name: str, **kwargs) -> DatasetLoader:
    """Build a loader by name. Imports are lazy -- KITTI pulls in torch, SUN RGB-D doesn't."""
    if name == "sunrgbd":
        from ..config import SUNRGBD_ROOT
        from .sunrgbd import SunRGBDLoader
        return SunRGBDLoader(SUNRGBD_ROOT, **kwargs)
    if name == "kitti":
        from ..config import KITTI_ROOT
        from .kitti import KittiLoader
        return KittiLoader(KITTI_ROOT, **kwargs)
    raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")


__all__ = ["DatasetLoader", "DATASETS", "get_loader"]
