"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Cluster and Obstacle shape contracts. No dataset required.
"""

import numpy as np

from collision_avoidance.types import Cluster, Obstacle


def test_cluster_contract():
    points = np.zeros((5, 3), dtype=np.float64)
    centroid = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    cluster = Cluster(points=points, centroid=centroid, id=0)
    assert cluster.points.shape == (5, 3)
    assert cluster.centroid.shape == (3,)
    assert cluster.id == 0


def test_obstacle_contract():
    centroid = np.array([0.5, 0.0, 3.0], dtype=np.float64)
    footprint = np.zeros((0, 2), dtype=np.float64)
    obs = Obstacle(label="unknown", centroid=centroid, footprint=footprint)
    assert obs.label == "unknown"
    assert obs.centroid.shape == (3,)
    assert obs.footprint.shape == (0, 2)
    assert obs.detection is None
