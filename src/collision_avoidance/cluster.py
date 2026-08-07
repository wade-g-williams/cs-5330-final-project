"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Group the ground-free point cloud into 3D obstacle clusters, so objects the detector
never saw are still found. The clustering is written from scratch -- an assignment requirement.

Contract: takes (N, 3) float64 points in meters, camera frame X right / Y down / Z forward.
scipy's KD-tree is fine for the neighbour search; the clustering logic itself must be ours.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import config
from .types import Cluster


def cluster_points(
    points: np.ndarray,
    eps: float = config.CLUSTER_EPS,
    min_points: int = config.CLUSTER_MIN_POINTS,
) -> list[Cluster]:
    """Self-implemented Euclidean clustering (radius-based region growing).

    Two points belong to the same obstacle if they are within ``eps`` meters of
    each other; connectivity is transitive, so each cluster is one connected
    component under that radius graph. We use a KD-tree only to answer the
    "who is within eps of this point" query fast -- the region-growing (a
    breadth-first flood fill over unvisited points) is done by hand.

    Args:
        points: (N, 3) float64, meters, camera frame X right / Y down / Z forward.
        eps: neighbour radius in meters; larger merges more, smaller splits more.
        min_points: clusters with fewer points than this are discarded as noise.

    Returns:
        A list of Cluster records with contiguous ids 0..k-1, each carrying its
        member points and their centroid. Empty input yields an empty list.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    n_points = len(points)
    if n_points == 0:
        return []

    tree = cKDTree(points)
    visited = np.zeros(n_points, dtype=bool)
    clusters: list[Cluster] = []

    for start in range(n_points):
        if visited[start]:
            continue

        # breadth-first flood fill from this seed point
        members: list[int] = []
        stack = [start]
        visited[start] = True
        while stack:
            current = stack.pop()
            members.append(current)
            neighbours = tree.query_ball_point(points[current], eps)
            for nb in neighbours:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        if len(members) < min_points:
            continue                       # too small -> noise, drop it

        member_points = points[np.asarray(members, dtype=np.int64)]
        clusters.append(
            Cluster(
                points=member_points,
                centroid=member_points.mean(axis=0),
                id=len(clusters),          # contiguous ids over the KEPT clusters
            )
        )

    return clusters
