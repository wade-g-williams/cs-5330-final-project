"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Euclidean clustering contract tests on synthetic clouds -- well-separated blobs become
one cluster each with centroids at the right places, sub-threshold blobs are dropped as noise,
ids come out contiguous, and empty input is handled. No dataset required.
"""

import numpy as np
import pytest

from collision_avoidance.cluster import cluster_points
from collision_avoidance.types import Cluster


def blob(center, n, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return np.array(center, dtype=np.float64) + rng.normal(0.0, spread, (n, 3))


def test_two_separated_blobs_give_two_clusters():
    a = blob([0.0, 0.0, 3.0], 80, seed=1)
    b = blob([2.0, 0.0, 3.0], 80, seed=2)     # 2 m away -> far beyond eps
    points = np.vstack([a, b])

    clusters = cluster_points(points, eps=0.3, min_points=20)

    assert len(clusters) == 2
    assert all(isinstance(c, Cluster) for c in clusters)
    # every input point is accounted for across the two clusters
    assert sum(len(c.points) for c in clusters) == len(points)

    # centroids sit at the two blob centers (order not guaranteed -> sort by x)
    centroids = sorted((c.centroid for c in clusters), key=lambda p: p[0])
    assert centroids[0] == pytest.approx([0.0, 0.0, 3.0], abs=0.05)
    assert centroids[1] == pytest.approx([2.0, 0.0, 3.0], abs=0.05)


def test_small_blobs_are_dropped_as_noise():
    big = blob([0.0, 0.0, 3.0], 80, seed=1)
    tiny = blob([3.0, 0.0, 3.0], 5, seed=2)   # only 5 points, below min_points
    points = np.vstack([big, tiny])

    clusters = cluster_points(points, eps=0.3, min_points=20)

    assert len(clusters) == 1
    assert len(clusters[0].points) == 80


def test_ids_are_contiguous_over_kept_clusters():
    a = blob([0.0, 0.0, 3.0], 40, seed=1)
    tiny = blob([1.0, 0.0, 3.0], 3, seed=2)   # dropped
    b = blob([2.0, 0.0, 3.0], 40, seed=3)
    points = np.vstack([a, tiny, b])

    clusters = cluster_points(points, eps=0.3, min_points=20)

    ids = sorted(c.id for c in clusters)
    assert ids == list(range(len(clusters)))   # 0..k-1, no gap from the dropped blob


def test_one_dense_blob_is_a_single_cluster():
    points = blob([0.0, 0.0, 2.0], 100, seed=4)
    clusters = cluster_points(points, eps=0.3, min_points=20)
    assert len(clusters) == 1
    assert len(clusters[0].points) == 100


def test_empty_input_returns_no_clusters():
    points = np.empty((0, 3), dtype=np.float64)
    assert cluster_points(points) == []
