"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Fit and remove the ground plane, so the floor or road stops being reported as an
obstacle. RANSAC here is written from scratch -- that is an assignment requirement.

Contract: takes and returns (N, 3) float64 point clouds in meters, camera frame X right /
Y down / Z forward. Seed the random sampling so the report's numbers reproduce.
"""

from __future__ import annotations

import numpy as np

from . import config


def plane_from_three_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Plane through three points as (unit_normal, offset) with n·p + d = 0.

    Returns None if the three points are (near) collinear, in which case the
    caller should just draw another sample.
    """
    n = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(n)
    if norm < 1e-8:                       
        return None
    n = n / norm
    d = -float(n @ p0)
    return n, d


def refit_plane(inliers: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane fit to a set of inliers via SVD (total least squares).

    The best-fit plane passes through the centroid; its normal is the direction
    of least variance -- the last right-singular vector of the centered points.
    """
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid)
    n = vt[-1]
    d = -float(n @ centroid)
    return n, d


def fit_ground_plane(
    points: np.ndarray,
    dist_thresh: float = config.GROUND_DIST_THRESH,
    max_tilt_deg: float = config.GROUND_MAX_TILT_DEG,
    num_iters: int = config.GROUND_RANSAC_ITERS,
    seed: int = config.GROUND_RANSAC_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Self-implemented RANSAC fit of the dominant (near-horizontal) plane.

    Each iteration samples three points, forms a candidate plane, rejects it if
    its normal is not roughly vertical (the floor's normal is +/-Y in this camera
    frame), and otherwise counts inliers within ``dist_thresh`` meters. The plane
    with the most inliers wins and is refined by a least-squares fit over all its
    inliers. Iterations adapt down as a good model is found.

    Args:
        points: (N, 3) float64, meters, camera frame X right / Y down / Z forward.
        dist_thresh: point-to-plane inlier band, meters.
        max_tilt_deg: reject candidate planes whose normal tilts more than this
            from the vertical (Y) axis -- stops walls/table-tops from winning.
        num_iters: maximum RANSAC iterations (an upper cap; may stop earlier).
        seed: RNG seed so the fit reproduces run to run.

    Returns:
        (plane, inlier_mask) where plane is (a, b, c, d) with a·x+b·y+c·z+d=0 and
        unit normal (a, b, c), and inlier_mask is a (N,) bool array marking ground
        points. If no vertical plane is found, plane is zeros and the mask is all
        False (nothing is treated as ground).
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    n_points = len(points)
    empty_plane = np.zeros(4, dtype=np.float64)
    if n_points < 3:
        return empty_plane, np.zeros(n_points, dtype=bool)

    rng = np.random.default_rng(seed)
    cos_tol = np.cos(np.radians(max_tilt_deg))
    y_axis = np.array([0.0, 1.0, 0.0])

    best_count = 0
    best_normal = None
    best_d = 0.0

    i = 0
    iters = num_iters
    while i < iters:
        i += 1
        idx = rng.choice(n_points, size=3, replace=False)
        plane = plane_from_three_points(points[idx[0]], points[idx[1]], points[idx[2]])
        if plane is None:
            continue
        n, d = plane

        # orientation gate: keep only near-horizontal planes (normal ~ +/-Y)
        if abs(float(n @ y_axis)) < cos_tol:
            continue

        dists = np.abs(points @ n + d)
        count = int(np.count_nonzero(dists < dist_thresh))
        if count > best_count:
            best_count = count
            best_normal, best_d = n, d

            # adaptively lower the iteration budget as the model improves
            w = best_count / n_points
            if 0.0 < w < 1.0:
                denom = np.log(1.0 - w ** 3)
                if denom < 0.0:
                    needed = int(np.log(1.0 - 0.99) / denom) + 1
                    iters = min(iters, max(needed, 1))

    if best_normal is None:               # no vertical plane found
        return empty_plane, np.zeros(n_points, dtype=bool)

    # refine on the winning inlier set, then recompute the final mask
    inlier_mask = np.abs(points @ best_normal + best_d) < dist_thresh
    n, d = refit_plane(points[inlier_mask])
    inlier_mask = np.abs(points @ n + d) < dist_thresh

    plane = np.array([n[0], n[1], n[2], d], dtype=np.float64)
    return plane, inlier_mask


def remove_ground(
    points: np.ndarray,
    dist_thresh: float = config.GROUND_DIST_THRESH,
    max_tilt_deg: float = config.GROUND_MAX_TILT_DEG,
    num_iters: int = config.GROUND_RANSAC_ITERS,
    seed: int = config.GROUND_RANSAC_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the ground plane and return the cloud with the ground removed.

    Args:
        points: (N, 3) float64 point cloud (see module contract).
        dist_thresh, max_tilt_deg, num_iters, seed: forwarded to fit_ground_plane.

    Returns:
        (nonground_points, inlier_mask) -- the (M, 3) obstacle cloud (ground
        dropped) and the (N,) ground mask over the input (kept so callers/viz can
        color the removed floor).
    """
    _, inlier_mask = fit_ground_plane(
        points, dist_thresh=dist_thresh, max_tilt_deg=max_tilt_deg,
        num_iters=num_iters, seed=seed)
    return points[~inlier_mask], inlier_mask
