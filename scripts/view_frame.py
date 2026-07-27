"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Eyeball one RGB-D frame -- image, depth colormap, and printed stats. Whether the
depth *looks* like the scene is a judgement no assertion makes; the checkable half is in tests.

    python scripts/view_frame.py --dataset sunrgbd --index 0            # GUI window
    python scripts/view_frame.py --dataset kitti --index 0 --save out/  # headless, for SSH
"""

import argparse
import os

import cv2
import numpy as np

from collision_avoidance import viz
from collision_avoidance.datasets import DATASETS, get_loader


def summarize(frame, n_frames: int) -> None:
    """Print the frame's shapes, intrinsics, and metric depth stats."""
    valid = frame.depth[frame.depth > 0]
    h, w = frame.depth.shape
    print(f"dataset  : {frame.meta.get('dataset', '?')}, {n_frames} frames")
    print(f"frame_id : {frame.frame_id}")
    print(f"rgb      : {frame.rgb.shape} {frame.rgb.dtype}")
    print(f"depth    : {frame.depth.shape} {frame.depth.dtype} "
          f"({frame.meta.get('depth_source', 'sensor')})")
    print(f"K:\n{frame.K}")
    print(f"fx={frame.fx:.1f} fy={frame.fy:.1f} cx={frame.cx:.1f} cy={frame.cy:.1f}")
    if valid.size:
        print(f"depth (m): min={valid.min():.2f}  median={np.median(valid):.2f}  max={valid.max():.2f}")
        print(f"valid depth: {valid.size / frame.depth.size:.1%} of pixels")
        print(f"center pixel depth: {frame.depth[h // 2, w // 2]:.2f} m")
    else:
        print("depth (m): no valid readings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="sunrgbd")
    ap.add_argument("--index", type=int, default=0, help="which frame to load (index = id - 1)")
    ap.add_argument("--save", metavar="DIR",
                    help="write a side-by-side jpg here instead of opening a window")
    args = ap.parse_args()

    loader = get_loader(args.dataset)
    frame = loader[args.index]
    summarize(frame, len(loader))

    rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    depth_vis = viz.depth_colormap(frame.depth)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        path = os.path.join(args.save, f"{args.dataset}_{frame.frame_id}_rgbd.jpg")
        cv2.imwrite(path, viz.side_by_side(rgb_bgr, depth_vis))
        print(f"wrote {path}")
        return

    cv2.imshow("rgb", rgb_bgr)
    cv2.imshow("depth (jet: near=blue, far=red, holes=black)", depth_vis)
    print("press any key in an image window to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
