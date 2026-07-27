"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Load the SUN RGB-D mirror (flat image/depth/calib layout) as Frames. Depth is stored
as a tilt-rotated point cloud, not a depth image, so it must be projected back onto the pixels.
"""

from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

from .base import DatasetLoader
from ..frame import Frame


def _project_to_pixels(xyz: np.ndarray, rtilt: np.ndarray, K: np.ndarray,
                       shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project stored SUN RGB-D cloud points onto the pixel grid they came from.

    read3dPoints.m stores points as `Rtilt @ [X, Z, -Y]`, so the room tilt must be undone
    before the columns are un-permuted. Skipping the undo still yields a full-looking depth
    map with the wrong value at every pixel, which is why tests/test_datasets.py scores this
    against the photo rather than just checking its shape.

    Args:
        xyz: (N, 3) stored point positions, in meters.
        rtilt: (3, 3) room-tilt rotation, from line 1 of calib/*.txt.
        K: (3, 3) camera intrinsics, already un-transposed.
        shape: (H, W) of the image the points must land on.

    Returns:
        (u, v, z, in_bounds) -- integer pixel columns/rows, forward distance in meters, and
        the boolean mask of points landing inside the frame in front of the camera.
    """
    p = xyz @ rtilt.T                          # for row-vectors, Rtilt^-1 is `@ rtilt.T`
    Xc, Yc, Zc = p[:, 0], -p[:, 2], p[:, 1]    # -> camera frame: X right, Y down, Z forward

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.round(fx * Xc / Zc + cx).astype(np.int64)
    v = np.round(fy * Yc / Zc + cy).astype(np.int64)

    h, w = shape
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (Zc > 0)
    return u, v, Zc, in_bounds


class SunRGBDLoader(DatasetLoader):
    def __init__(self, root: str | Path, label_version: str = "v2"):
        if label_version not in ("v1", "v2"):
            raise ValueError(f"label_version must be 'v1' or 'v2', got {label_version!r}")
        self.root = Path(root)
        self.label_version = label_version
        self._label_dir = self.root / ("label_v1" if label_version == "v1" else "label")
        # Build an index of samples once. Each entry holds the file paths for
        # one frame. This mirror is a flat layout: image/NNNNNN.jpg, depth/NNNNNN.mat,
        # calib/NNNNNN.txt, all sharing the same zero-padded numeric id.
        self.samples = self._index_samples()
        if not self.samples:
            raise FileNotFoundError(f"No SUN RGB-D samples found under {self.root}")
        self._id_to_index = {s["id"]: i for i, s in enumerate(self.samples)}

    def _index_samples(self) -> list[dict]:
        samples = []
        for rgb_path in sorted(self.root.glob("image/*.jpg")):
            sample_id = rgb_path.stem
            label = self._label_dir / f"{sample_id}.txt"
            samples.append({
                "id": sample_id,
                "rgb": rgb_path,
                "depth": self.root / "depth" / f"{sample_id}.mat",
                "intr": self.root / "calib" / f"{sample_id}.txt",
                "label": label if label.exists() else None,
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Frame:
        s = self.samples[index]
        rgb = self._read_rgb(s["rgb"])
        rtilt, K = self._read_calib(s["intr"])
        depth = self._read_depth(s["depth"], rtilt, K, rgb.shape[:2])
        return Frame(
            rgb=rgb,
            depth=depth,
            K=K,
            frame_id=s["id"],
            meta={"dataset": "sunrgbd", "paths": s},
        )

    def split_indices(self, split: str) -> list[int]:
        """Loader indices for the official train or val split.

        The mirror's *_data_idx.txt files store 1-based sample numbers (1 = frame 000001).
        """
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        path = self.root / f"{split}_data_idx.txt"
        if not path.exists():
            raise FileNotFoundError(f"split file not found: {path}")
        indices = []
        for token in path.read_text().split():
            sample_id = f"{int(token):06d}"
            if sample_id not in self._id_to_index:
                raise KeyError(f"split id {sample_id} not in indexed samples")
            indices.append(self._id_to_index[sample_id])
        return indices

    def read_labels(self, index: int) -> list[dict]:
        """Ground-truth boxes for one frame.

        Each label/*.txt line is:
            classname x y w h cx cy cz sx sy sz ox oy
        where (x, y, w, h) is the 2D box, (cx, cy, cz) the 3D centroid, (sx, sy, sz) the
        size coeffs as stored in the mirror, and (ox, oy) the heading vector.
        """
        path = self.samples[index]["label"]
        if path is None:
            return []
        labels = []
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) < 13:
                continue
            x, y, w, h = (float(v) for v in parts[1:5])
            ox, oy = float(parts[11]), float(parts[12])
            labels.append({
                "type": parts[0],
                "bbox": (x, y, x + w, y + h),                 # (x1, y1, x2, y2) in pixels
                "location": tuple(float(v) for v in parts[5:8]),   # centroid cx, cy, cz
                "dimensions": tuple(float(v) for v in parts[8:11]),  # size coeffs sx, sy, sz
                "orientation": (ox, oy),
                "rotation_y": float(np.arctan2(oy, ox)),
            })
        return labels

    # --- readers -----------------------------------------------------------

    def _read_rgb(self, path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)   # OpenCV loads as BGR
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)     # BGR --> RGB, per our Frame convention

    def _read_depth(self, path, rtilt: np.ndarray, K: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        # depth/NNNNNN.mat holds an (N, 6) point cloud [x, y, z, r, g, b], already in meters.
        mat = sio.loadmat(str(path))
        if mat is None or "instance" not in mat:
            raise FileNotFoundError(f"could not read depth points: {path}")
        pts = mat["instance"].astype(np.float64)

        # Project the cloud back onto the pixel grid, then rasterize distance into it.
        u, v, Zc, in_bounds = _project_to_pixels(pts[:, :3], rtilt, K, shape)
        u, v, Zc = u[in_bounds], v[in_bounds], Zc[in_bounds]

        h, w = shape
        depth_m = np.zeros((h, w), dtype=np.float32)
        # Where multiple points land on the same pixel, keep the nearest one.
        order = np.argsort(-Zc)                               # far first, near last -> near wins
        depth_m[v[order], u[order]] = Zc[order].astype(np.float32)
        depth_m[depth_m > 8.0] = 0.0                          # these sensors max ~8m; treat > 8m as invalid
        return depth_m

    def _read_calib(self, path) -> tuple[np.ndarray, np.ndarray]:
        # calib/NNNNNN.txt has two lines: Rtilt (the room-tilt rotation, needed to bring
        # the stored point cloud back into the camera frame -- see _project_to_pixels) and K
        # stored transposed (fx 0 0 / 0 fy 0 / cx cy 1) -- transpose it back.
        vals = np.loadtxt(str(path))
        rtilt = vals[0].reshape(3, 3).astype(np.float64)
        K_transposed = vals[1].reshape(3, 3)
        return rtilt, K_transposed.T.astype(np.float64)
