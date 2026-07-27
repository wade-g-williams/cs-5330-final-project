"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Shared records that later stages pass between each other. Detection stays in
detection.py; point clouds stay plain (N, 3) float64 arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .detection import Detection


@dataclass
class Cluster:
    """One obstacle blob from clustering. Fusion consumes these."""

    points: np.ndarray      # (M, 3) float64, meters, camera frame X right / Y down / Z forward
    centroid: np.ndarray    # (3,) float64
    id: int

    def __post_init__(self):
        assert self.points.ndim == 2 and self.points.shape[1] == 3, (
            f"points must be Mx3, got {self.points.shape}"
        )
        assert self.centroid.shape == (3,), f"centroid must be (3,), got {self.centroid.shape}"


@dataclass
class Obstacle:
    """One fused obstacle for the BEV map."""

    label: str                          # dataset class or "unknown"
    centroid: np.ndarray                # (3,) float64, meters, camera frame
    footprint: np.ndarray               # (K, 2) float64 BEV x-z corners; empty (0, 2) until filled
    detection: Detection | None = None  # matched 2D box, if any

    def __post_init__(self):
        assert self.centroid.shape == (3,), f"centroid must be (3,), got {self.centroid.shape}"
        assert self.footprint.ndim == 2 and self.footprint.shape[1] == 2, (
            f"footprint must be Kx2, got {self.footprint.shape}"
        )
