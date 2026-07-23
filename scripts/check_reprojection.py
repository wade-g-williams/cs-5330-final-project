"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Settle which transform correctly maps the SUN RGB-D mirror's depth/*.mat point
clouds back onto the image, so datasets/sunrgbd.py can build a correctly aligned depth map.

The clouds carry their OWN per-point colour, which gives us an objective score: reproject
the points, and compare each point's colour against the real photo's pixel at the location
it lands on. The correct transform makes those agree (low colour MAE) and puts nearly every
point inside the frame; wrong transforms smear colours and fling points off-frame.

Columns of the stored cloud are a permutation of the pinhole camera frame
(X right, Y down, Z forward): col0 = X, col1 = Z, col2 = -Y. What's ambiguous is whether
Rtilt (the room-tilt rotation on line 1 of calib/*.txt) still needs undoing, and in which
direction -- so we test all three and let the numbers decide.
"""

import argparse
import os

import cv2
import numpy as np
import scipy.io as sio

from collision_avoidance import config

# name -> how to map stored rows into the camera-frame permutation
VARIANTS = [
    ("A: no Rtilt (current loader)", None),
    ("B: @ Rtilt", "R"),
    ("C: @ Rtilt.T", "RT"),
]


def load_raw(root, sample_id):
    pts = sio.loadmat(str(root / "depth" / f"{sample_id}.mat"))["instance"].astype(np.float64)
    calib = np.loadtxt(str(root / "calib" / f"{sample_id}.txt"))
    rtilt = calib[0].reshape(3, 3)
    k = calib[1].reshape(3, 3).T                 # calib stores K transposed
    rgb = cv2.imread(str(root / "image" / f"{sample_id}.jpg"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"could not read image for sample {sample_id}")
    return pts, rtilt, k, rgb


def to_camera_frame(xyz, rtilt, mode):
    """Map stored points into the pinhole camera frame (X right, Y down, Z forward)."""
    if mode == "R":
        p = xyz @ rtilt
    elif mode == "RT":
        p = xyz @ rtilt.T
    else:
        p = xyz
    return p[:, 0], -p[:, 2], p[:, 1]            # X, Y(down), Z(forward)


def evaluate(pts, rtilt, k, rgb, mode):
    """Reproject, paint each point's own colour, and score against the real photo."""
    x, y, z = to_camera_frame(pts[:, :3], rtilt, mode)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * x / z + cx
        v = fy * y / z + cy

    h, w = rgb.shape[:2]
    ok = np.isfinite(u) & np.isfinite(v) & (z > 0)
    u = np.where(ok, np.round(u), -1).astype(np.int64)
    v = np.where(ok, np.round(v), -1).astype(np.int64)
    ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # point colours: columns 3..5, normalised [0, 1], RGB order -> BGR for OpenCV
    colors = (np.clip(pts[:, 3:6], 0.0, 1.0) * 255).astype(np.uint8)[:, ::-1]

    uu, vv, zz, cc = u[ok], v[ok], z[ok], colors[ok]
    order = np.argsort(-zz)                       # far first, so nearer points overwrite
    canvas = np.zeros((h, w, 3), np.uint8)
    canvas[vv[order], uu[order]] = cc[order]

    # THE decisive score: does each point's own colour match the photo where it landed?
    in_frame = float(ok.mean())
    color_mae = float(np.abs(rgb[vv, uu].astype(np.float64) - cc.astype(np.float64)).mean()) \
        if len(uu) else float("inf")
    return canvas, in_frame, color_mae


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=1000, help="frame index (index = id - 1)")
    ap.add_argument("--out", default="out/reprojection_check")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sample_id = f"{args.index + 1:06d}"
    pts, rtilt, k, rgb = load_raw(config.SUNRGBD_ROOT, sample_id)

    print(f"frame {sample_id}: {len(pts)} points, image {rgb.shape[1]}x{rgb.shape[0]}")
    print(f"{'variant':32s} {'in frame':>10s} {'colour MAE':>12s}   (lower MAE = correct)")

    panels, best = [label(rgb, "REAL PHOTO")], None
    for name, mode in VARIANTS:
        canvas, in_frame, mae = evaluate(pts, rtilt, k, rgb, mode)
        print(f"{name:32s} {in_frame:9.1%} {mae:12.1f}")
        panels.append(label(canvas, f"{name}  {in_frame:.0%} in, MAE {mae:.0f}"))
        if best is None or mae < best[1]:
            best = (name, mae)

    path = os.path.join(args.out, f"{sample_id}_reprojection.jpg")
    cv2.imwrite(path, np.hstack(panels))
    print(f"\nbest (lowest colour MAE): {best[0]}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
