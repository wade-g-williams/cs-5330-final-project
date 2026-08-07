"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: RANSAC ground-removal contract tests on synthetic clouds -- a near-horizontal plane
plus a floating blob. Verifies the plane is found with a vertical normal, ground points are the
inliers, obstacle points survive removal, the fit reproduces under a fixed seed, and degenerate
inputs are handled. No dataset required.
"""

import numpy as np
import pytest

from collision_avoidance.ground import fit_ground_plane, remove_ground


def synthetic_scene(seed: int = 0) -> tuple[np.ndarray, int, int]:
    """A flat floor at y = 1.5 (Y is down) plus a compact obstacle blob above it.

    Returns (points, n_ground, n_obstacle) so tests can check the split by count.
    """
    rng = np.random.default_rng(seed)

    n_ground = 500
    gx = rng.uniform(-2.0, 2.0, n_ground)
    gz = rng.uniform(1.0, 5.0, n_ground)
    gy = 1.5 + rng.normal(0.0, 0.005, n_ground)      # thin noise about the plane
    ground = np.stack([gx, gy, gz], axis=1)

    n_obstacle = 120
    obstacle = np.array([0.0, 0.5, 3.0]) + rng.normal(0.0, 0.05, (n_obstacle, 3))

    points = np.vstack([ground, obstacle]).astype(np.float64)
    return points, n_ground, n_obstacle


def test_plane_normal_is_vertical():
    points, _, _ = synthetic_scene()
    plane, _ = fit_ground_plane(points)
    normal = plane[:3]
    # the floor normal must be (near) parallel to the Y axis
    assert abs(normal[1]) == pytest.approx(1.0, abs=1e-2)
    assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-6)


def test_ground_inliers_match_the_floor():
    points, n_ground, n_obstacle = synthetic_scene()
    _, mask = fit_ground_plane(points)

    # the first n_ground points are the floor; almost all should be inliers
    assert mask[:n_ground].mean() > 0.99
    # essentially none of the obstacle points should be called ground
    assert mask[n_ground:].sum() == 0


def test_remove_ground_keeps_the_obstacle():
    points, n_ground, n_obstacle = synthetic_scene()
    nonground, mask = remove_ground(points)

    assert nonground.shape[1] == 3
    assert nonground.dtype == np.float64
    # what's left is the obstacle blob (allow a tiny margin for stray floor noise)
    assert n_obstacle <= len(nonground) <= n_obstacle + 5
    # mask length matches the input and removal is its complement
    assert mask.shape == (len(points),)
    assert len(nonground) == int((~mask).sum())


def test_fit_is_reproducible_with_a_fixed_seed():
    points, _, _ = synthetic_scene()
    plane_a, mask_a = fit_ground_plane(points, seed=42)
    plane_b, mask_b = fit_ground_plane(points, seed=42)
    assert np.allclose(plane_a, plane_b)
    assert np.array_equal(mask_a, mask_b)


def test_too_few_points_returns_no_ground():
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])   # only 2 points
    plane, mask = fit_ground_plane(points)
    assert not mask.any()
    assert mask.shape == (2,)
