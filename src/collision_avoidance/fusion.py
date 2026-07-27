"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Match 3D clusters to 2D detections by projecting the clusters onto the image, so every
obstacle ends up with a class label or "unknown", a 3D centroid, and a box.

Contract: consumes the Detection dataclass from detection.py, which is already fixed. Matching
needs to know which pixels a cluster occupies -- whether back-projection hands that over or it
gets recomputed here is a decision to settle with whoever owns geometry.py.
"""
