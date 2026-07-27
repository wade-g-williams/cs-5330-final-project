"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Rasterize fused obstacles into the top-down Bird's-Eye-View occupancy map with
real-world distances and collision warnings -- the project's headline output.

Contract: takes the fused obstacles, returns a BGR image. Drawing helpers that other modules
would also want belong in viz.py; grid and rasterization logic stays here.
"""
