"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Back-project a depth map through the camera intrinsics into a metric 3D point cloud,
the pseudo-LiDAR representation that ground removal, clustering, and fusion all read.

Contract: point clouds are (N, 3) float in METERS, camera frame X right / Y down / Z
forward. A depth of 0.0 means no reading and those pixels must be dropped, not projected.
"""

import numpy as np


def back_project(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Lift a metric depth map into a 3D point cloud via the pinhole model.

    X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = depth, for every pixel with a
    reading. Pixels where depth == 0.0 ("no reading") are dropped rather than projected
    to the camera origin.

    Args:
        depth: (H, W) float metric depth in meters; 0.0 = no reading.
        K: (3, 3) camera intrinsics, un-transposed (fx, fy, cx, cy read off the diagonal
            and last column, per the Frame contract).

    Returns:
        (N, 3) float points in meters, camera frame X right / Y down / Z forward.
        N is the count of pixels with depth > 0; order is not meaningful.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    v, u = np.indices(depth.shape)  # v = row (y), u = column (x)
    z = depth.astype(np.float64)

    valid = z > 0.0
    u, v, z = u[valid].astype(
        np.float64), v[valid].astype(np.float64), z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def project_to_image(
    points: np.ndarray, K: np.ndarray, image_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame 3D points back onto the pixel grid - the inverse of back_project.

    Fusion uses this to project 3D clusters onto the image and match them to 2D detections.

    Args:
        points: (N, 3) float meters, camera frame X right / Y down / Z forward.
        K: (3, 3) camera intrinsics, un-transposed.
        image_shape: (H, W) of the image to project onto.

    Returns:
        (u, v, z, in_bounds) - float pixel columns/rows, camera-forward depth in meters,
        and a boolean mask of points that land inside the image in front of the camera.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    front = z > 0.0
    u = np.zeros(len(points), dtype=np.float64)
    v = np.zeros(len(points), dtype=np.float64)
    u[front] = fx * x[front] / z[front] + cx
    v[front] = fy * y[front] / z[front] + cy

    h, w = image_shape
    in_bounds = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, z, in_bounds
