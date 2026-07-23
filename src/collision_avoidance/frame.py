"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: 
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Frame:
    """
    One RGB-D frame in a standard form.

    Every dataset loader returns a Frame.
    """

    rgb: np.ndarray     # (H, W, 3) uint8, RGB channel order
    depth: np.ndarray   # (H, W)    float32, metric depth in meters; 0.0 = no reading
    K: np.ndarray       # (3, 3)    float64 camera intrinsics
    frame_id: str = ""  # dataset-specific identifier for logging/debugging
    meta: dict = field(default_factory=dict)    # extras (sensor type, file paths, etc.)

    def __post_init__(self):
        assert self.rgb.ndim == 3 and self.rgb.shape[2] == 3, f"rgb must be HxWx3, got {self.rgb.shape}"
        assert self.rgb.dtype == np.uint8, f"rgb must be uint8, got {self.rgb.dtype}"
        assert self.depth.ndim == 2, f"depth must be HxW, got {self.depth.shape}"
        assert self.rgb.shape[:2] == self.depth.shape, (f"rgb {self.rgb.shape[:2]} and depth {self.depth.shape} must be the same size")
        assert self.K.shape == (3, 3), f"K must be 3x3, got {self.K.shape}"

    # Named access to the intrinsics
    @property
    def fx(self) -> float: return float(self.K[0, 0])
    @property
    def fy(self) -> float: return float(self.K[1, 1])
    @property
    def cx(self) -> float: return float(self.K[0, 2])
    @property
    def cy(self) -> float: return float(self.K[1, 2])


