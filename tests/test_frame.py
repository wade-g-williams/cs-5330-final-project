"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Lock the Frame contract that every loader produces and every stage assumes. Frame
validates itself in __post_init__; these prove those guards fire. No dataset needed.
"""

import numpy as np
import pytest

from collision_avoidance.frame import Frame


def make_frame(h: int = 4, w: int = 6, **overrides) -> Frame:
    """A minimal valid Frame; pass overrides to break exactly one part of the contract."""
    fields = {
        "rgb": np.zeros((h, w, 3), dtype=np.uint8),
        "depth": np.ones((h, w), dtype=np.float32),
        "K": np.array([[500.0, 0.0, w / 2],
                       [0.0, 500.0, h / 2],
                       [0.0, 0.0, 1.0]]),
    }
    fields.update(overrides)
    return Frame(**fields)


def test_valid_frame_constructs():
    frame = make_frame()
    assert frame.rgb.shape[:2] == frame.depth.shape
    assert frame.frame_id == ""
    assert frame.meta == {}


def test_intrinsics_properties_read_the_right_cells():
    # fx/fy on the diagonal, cx/cy in the last column -- a transposed K would put cx/cy
    # in the bottom row instead, which is exactly the SUN RGB-D calib gotcha that
    # sunrgbd._read_calib has to undo.
    K = np.array([[525.0, 0.0, 319.5],
                  [0.0, 524.0, 239.5],
                  [0.0, 0.0, 1.0]])
    frame = make_frame(rgb=np.zeros((480, 640, 3), np.uint8),
                       depth=np.ones((480, 640), np.float32), K=K)
    assert (frame.fx, frame.fy) == (525.0, 524.0)
    assert (frame.cx, frame.cy) == (319.5, 239.5)
    assert all(isinstance(v, float) for v in (frame.fx, frame.fy, frame.cx, frame.cy))


def test_meta_is_not_shared_between_frames():
    # `meta: dict = field(default_factory=dict)` exists precisely to avoid the classic
    # mutable-default bug, where every Frame would share one dict.
    a, b = make_frame(), make_frame()
    a.meta["dataset"] = "sunrgbd"
    assert b.meta == {}


@pytest.mark.parametrize("bad", [
    pytest.param({"rgb": np.zeros((4, 6, 4), np.uint8)}, id="rgb-not-3-channel"),
    pytest.param({"rgb": np.zeros((4, 6), np.uint8)}, id="rgb-is-2D"),
    pytest.param({"rgb": np.zeros((4, 6, 3), np.float32)}, id="rgb-not-uint8"),
    pytest.param({"depth": np.ones((4, 6, 1), np.float32)}, id="depth-is-3D"),
    pytest.param({"depth": np.ones((8, 6), np.float32)}, id="depth-size-mismatch"),
    pytest.param({"K": np.eye(4)}, id="K-not-3x3"),
])
def test_invalid_frame_is_rejected(bad):
    with pytest.raises(AssertionError):
        make_frame(**bad)
