"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Fit and remove the ground plane, so the floor or road stops being reported as an
obstacle. RANSAC here is written from scratch -- that is an assignment requirement.

Contract: takes and returns (N, 3) float64 point clouds in meters, camera frame X right /
Y down / Z forward. Seed the random sampling so the report's numbers reproduce.
"""
