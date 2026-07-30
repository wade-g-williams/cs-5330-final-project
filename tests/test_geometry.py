"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Back-projection contract tests -- a known pixel round-trips through back_project /
project_to_image to the same pixel, depth==0 pixels are dropped rather than projected to the
camera origin, and the output shape/dtype matches what ground.py and cluster.py expect.
"""

import numpy as np
import pytest

from collision_avoidance.geometry import back_project, project_to_image


def _synthetic_K() -> np.ndarray:
    # fx=fy=500, principal point at the center of a 64x48 image.
    return np.array([
        [500.0, 0.0, 32.0],
        [0.0, 500.0, 24.0],
        [0.0, 0.0, 1.0],
    ])


def test_back_project_drops_zero_depth_pixels():
    depth = np.zeros((48, 64), dtype=np.float32)
    depth[24, 32] = 2.0    # exactly one valid reading; the rest are "no reading"

    points = back_project(depth, _synthetic_K())

    assert points.shape == (1, 3)
    assert points.dtype == np.float64


def test_back_project_matches_the_pinhole_formula_at_the_principal_point():
    # A pixel exactly at (cx, cy) must back-project to X=0, Y=0, Z=depth.
    depth = np.zeros((48, 64), dtype=np.float32)
    depth[24, 32] = 3.0

    (point,) = back_project(depth, _synthetic_K())
    assert point == pytest.approx([0.0, 0.0, 3.0])


def test_back_project_and_project_to_image_round_trip():
    K = _synthetic_K()
    h, w = 48, 64
    depth = np.zeros((h, w), dtype=np.float32)
    depth[10, 20] = 4.5
    depth[30, 50] = 1.2

    points = back_project(depth, K)
    u, v, z, in_bounds = project_to_image(points, K, (h, w))

    assert in_bounds.all()
    # Recover the same (u, v) pixel coordinates and depth we started from.
    recovered = sorted(zip(np.round(v).astype(int), np.round(u).astype(int), z))
    expected = sorted([(10, 20, 4.5), (30, 50, 1.2)])
    for (rv, ru, rz), (ev, eu, ez) in zip(recovered, expected):
        assert (rv, ru) == (ev, eu)
        assert rz == pytest.approx(ez)


def test_project_to_image_masks_points_behind_the_camera():
    K = _synthetic_K()
    points = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 2.0]])

    _u, _v, _z, in_bounds = project_to_image(points, K, (48, 64))

    assert list(in_bounds) == [False, True]
