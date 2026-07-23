"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: 
"""

from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

from .base import DatasetLoader
from ..frame import Frame


class SunRGBDLoader(DatasetLoader):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        # Build an index of samples once. Each entry holds the file paths for
        # one frame. This mirror is a flat layout: image/NNNNNN.jpg, depth/NNNNNN.mat,
        # calib/NNNNNN.txt, all sharing the same zero-padded numeric id.
        self.samples = self._index_samples()
        if not self.samples:
            raise FileNotFoundError(f"No SUN RGB-D samples found under {self.root}")

    def _index_samples(self) -> list[dict]:
        samples = []
        for rgb_path in sorted(self.root.glob("image/*.jpg")):
            sample_id = rgb_path.stem
            samples.append({
                "id": sample_id,
                "rgb": rgb_path,
                "depth": self.root / "depth" / f"{sample_id}.mat",
                "intr": self.root / "calib" / f"{sample_id}.txt",
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Frame:
        s = self.samples[index]
        rgb = self._read_rgb(s["rgb"])
        K = self._read_intrinsics(s["intr"])
        depth = self._read_depth(s["depth"], K, rgb.shape[:2])
        return Frame(
            rgb=rgb,
            depth=depth,
            K=K,
            frame_id=s["id"],
            meta={"dataset": "sunrgbd", "paths": s},
        )


    # --- readers -----------------------------------------------------------

    def _read_rgb(self, path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)   # OpenCV loads as BGR
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)     # BGR --> RGB, per our Frame convention

    def _read_depth(self, path, K: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        # depth/NNNNNN.mat holds an (N, 6) point cloud [x, y, z, r, g, b] in SUN RGB-D's "upright" frame:
        # x = right, y = forward (depth), z = up. Already in meters.
        mat = sio.loadmat(str(path))
        if mat is None or "instance" not in mat:
            raise FileNotFoundError(f"could not read depth points: {path}")
        pts = mat["instance"].astype(np.float64)

        # Convert upright (x, y_forward, z_up) --> pinhole camera frame
        # (Xc = right, Yc = down, Zc = forward), then reproject through K to
        # rasterize the point cloud back onto the pixel grid it came from.
        Xc, Yc, Zc = pts[:, 0], -pts[:, 2], pts[:, 1]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = np.round(fx * Xc / Zc + cx).astype(np.int64)
        v = np.round(fy * Yc / Zc + cy).astype(np.int64)

        h, w = shape
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (Zc > 0)
        u, v, Zc = u[in_bounds], v[in_bounds], Zc[in_bounds]

        depth_m = np.zeros((h, w), dtype=np.float32)
        # Where multiple points land on the same pixel, keep the nearest one.
        order = np.argsort(-Zc)                               # far first, near last -> near wins
        depth_m[v[order], u[order]] = Zc[order].astype(np.float32)
        depth_m[depth_m > 8.0] = 0.0                          # these sensors max ~8m; treat > 8m as invalid
        return depth_m

    def _read_intrinsics(self, path) -> np.ndarray:
        # calib/NNNNNN.txt has two lines: Rtilt (room-tilt rotation, unused here)
        # and K stored transposed (fx 0 0 / 0 fy 0 / cx cy 1) -- transpose it back.
        vals = np.loadtxt(str(path))
        K_transposed = vals[1].reshape(3, 3)
        return K_transposed.T.astype(np.float64)