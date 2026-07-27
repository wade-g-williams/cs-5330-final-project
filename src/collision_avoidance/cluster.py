"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Group the ground-free point cloud into 3D obstacle clusters, so objects the detector
never saw are still found. The clustering is written from scratch -- an assignment requirement.

Contract: takes (N, 3) float64 points in meters, camera frame X right / Y down / Z forward.
scipy's KD-tree is fine for the neighbour search; the clustering logic itself must be ours.
"""
