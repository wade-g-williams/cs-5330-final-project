"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Back-project a depth map through the camera intrinsics into a metric 3D point cloud
-- the "pseudo-LiDAR" representation that ground removal, clustering, and fusion all read.

Contract: point clouds are (N, 3) float64 in METERS, camera frame X right / Y down / Z
forward. A depth of 0.0 means "no reading" and those pixels must be dropped, not projected.
"""
