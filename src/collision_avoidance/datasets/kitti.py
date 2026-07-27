"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: KITTI 3D object detection loader. KITTI ships no depth image, so Frame.depth comes
from Depth Anything V2 -- this is the pseudo-LiDAR branch the outdoor evaluation rests on.
Velodyne bins are optional and read separately for the LiDAR ablation.
"""

from pathlib import Path

import cv2
import numpy as np

from .base import DatasetLoader
from ..depth import MetricDepthEstimator
from ..frame import Frame


def lidar_to_camera(
    points: np.ndarray,
    R0_rect: np.ndarray,
    Tr_velo_to_cam: np.ndarray,
) -> np.ndarray:
    """Transform Velodyne XYZ into the rectified camera frame.

    Args:
        points: (N, 3) or (N, 4) in the Velodyne frame (x forward, y left, z up).
        R0_rect: (3, 3) rectification rotation from calib.
        Tr_velo_to_cam: (3, 4) Velodyne-to-camera transform from calib.

    Returns:
        (N, 3) float64 in the camera frame (X right, Y down, Z forward).
    """
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    homo = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)])
    cam = (Tr_velo_to_cam @ homo.T).T
    return (R0_rect @ cam.T).T


def project_lidar_to_image(
    points: np.ndarray,
    P2: np.ndarray,
    R0_rect: np.ndarray,
    Tr_velo_to_cam: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project Velodyne points into the left colour image.

    Returns:
        (u, v, z, in_bounds) -- pixel columns/rows, camera-forward depth in meters, and
        a boolean mask of points that land inside the image in front of the camera.
    """
    cam = lidar_to_camera(points, R0_rect, Tr_velo_to_cam)
    homo = np.hstack([cam, np.ones((cam.shape[0], 1), dtype=np.float64)])
    uvw = (P2 @ homo.T).T
    z = cam[:, 2]
    # Avoid divide-by-zero behind the camera; those rows are masked out below.
    front = z > 0
    u = np.zeros(len(cam), dtype=np.float64)
    v = np.zeros(len(cam), dtype=np.float64)
    u[front] = uvw[front, 0] / uvw[front, 2]
    v[front] = uvw[front, 1] / uvw[front, 2]

    h, w = image_shape
    in_bounds = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, z, in_bounds


class KittiLoader(DatasetLoader):
    """KITTI object detection, left colour camera (image_2).

    Depth is *predicted*, not measured. It is estimated lazily on the first frame access so
    that len(loader), indexing paths, and the tests that only need geometry never pay for
    loading a ~1.4 GB depth model.
    """

    def __init__(self, root: str | Path, split: str = "training", depth_estimator=None):
        self.root = Path(root) / split
        self.split = split
        self._depth = depth_estimator          # injectable, so tests can pass a fake
        self.samples = self._index_samples()
        if not self.samples:
            raise FileNotFoundError(f"No KITTI samples found under {self.root}")

    def _index_samples(self) -> list[dict]:
        samples = []
        for rgb_path in sorted(self.root.glob("image_2/*.png")):
            sample_id = rgb_path.stem
            label = self.root / "label_2" / f"{sample_id}.txt"       # absent in the testing split
            velo = self.root / "velodyne" / f"{sample_id}.bin"
            samples.append({
                "id": sample_id,
                "rgb": rgb_path,
                "calib": self.root / "calib" / f"{sample_id}.txt",
                "label": label if label.exists() else None,
                "velodyne": velo if velo.exists() else None,
            })
        return samples

    @property
    def depth_estimator(self) -> MetricDepthEstimator:
        if self._depth is None:
            self._depth = MetricDepthEstimator(domain="outdoor")     # VKITTI checkpoint, 80 m
        return self._depth

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Frame:
        s = self.samples[index]
        rgb = self._read_rgb(s["rgb"])
        K = self._read_calib(s["calib"])
        depth = self.depth_estimator(rgb).astype(np.float32)
        return Frame(
            rgb=rgb,
            depth=depth,
            K=K,
            frame_id=s["id"],
            meta={"dataset": "kitti", "split": self.split, "depth_source": "depth_anything_v2",
                  "paths": s},
        )

    # --- readers -----------------------------------------------------------

    def _read_rgb(self, path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)                  # BGR -> RGB, per Frame's contract

    def _parse_calib(self, path) -> dict:
        # calib/NNNNNN.txt holds one matrix per line, "KEY: f1 f2 ...".
        # P2 is the 3x4 for the left colour camera; its left 3x3 block is K.
        # R0_rect and Tr_velo_to_cam bring Velodyne points into that camera.
        rows = {}
        for line in Path(path).read_text().splitlines():
            if ":" in line:
                key, _, values = line.partition(":")
                rows[key.strip()] = np.fromstring(values, sep=" ")
        for key in ("P2", "R0_rect", "Tr_velo_to_cam"):
            if key not in rows:
                raise ValueError(f"no {key} matrix in {path}")
        P2 = rows["P2"].reshape(3, 4).astype(np.float64)
        return {
            "P2": P2,
            "K": P2[:, :3].copy(),
            "R0_rect": rows["R0_rect"].reshape(3, 3).astype(np.float64),
            "Tr_velo_to_cam": rows["Tr_velo_to_cam"].reshape(3, 4).astype(np.float64),
        }

    def _read_calib(self, path) -> np.ndarray:
        return self._parse_calib(path)["K"]

    def read_calib(self, index: int) -> dict:
        """Full calib matrices for one frame: P2, K, R0_rect, Tr_velo_to_cam."""
        return self._parse_calib(self.samples[index]["calib"])

    def read_labels(self, index: int) -> list[dict]:
        """Ground-truth boxes for one frame; empty on the testing split."""
        path = self.samples[index]["label"]
        if path is None:
            return []
        labels = []
        for line in Path(path).read_text().splitlines():
            f = line.split()
            if len(f) < 15:
                continue
            labels.append({
                "type": f[0],                                  # Car / Pedestrian / Cyclist / ...
                "truncated": float(f[1]),
                "occluded": int(f[2]),
                "bbox": tuple(float(v) for v in f[4:8]),       # 2D box (x1, y1, x2, y2) in pixels
                "dimensions": tuple(float(v) for v in f[8:11]),   # h, w, l in meters
                "location": tuple(float(v) for v in f[11:14]),    # x, y, z of the BOTTOM centre
                "rotation_y": float(f[14]),
            })
        return labels

    def read_lidar(self, index: int) -> np.ndarray:
        """Raw Velodyne sweep for one frame: (N, 4) float32 [x, y, z, reflectance].

        Coordinates are in the Velodyne frame (x forward, y left, z up), not the camera
        frame. Use lidar_to_camera / project_lidar_to_image with read_calib() to convert.
        Frame.depth is still Depth Anything V2 -- this is a side channel for the ablation.
        """
        path = self.samples[index]["velodyne"]
        if path is None:
            raise FileNotFoundError(
                f"no velodyne for frame {self.samples[index]['id']}; "
                f"run `python scripts/download_datasets.py kitti --with-lidar`"
            )
        pts = np.fromfile(str(path), dtype=np.float32)
        if pts.size % 4 != 0:
            raise ValueError(f"velodyne file length not divisible by 4: {path}")
        return pts.reshape(-1, 4)
