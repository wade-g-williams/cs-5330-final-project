# scripts/smoke_test_stage0.py
import argparse

import cv2
import numpy as np

from collision_avoidance import config
from collision_avoidance.datasets.sunrgbd import SunRGBDLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="which frame to load")
    args = ap.parse_args()

    loader = SunRGBDLoader(config.SUNRGBD_ROOT)
    print(f"dataset: {len(loader)} frames")

    frame = loader[args.index]
    print(f"frame_id : {frame.frame_id}")
    print(f"rgb      : {frame.rgb.shape} {frame.rgb.dtype}")
    print(f"depth    : {frame.depth.shape} {frame.depth.dtype}")
    print(f"K:\n{frame.K}")
    print(f"fx={frame.fx:.1f} fy={frame.fy:.1f} cx={frame.cx:.1f} cy={frame.cy:.1f}")

    valid = frame.depth[frame.depth > 0]
    print(f"depth (m): min={valid.min():.2f}  median={np.median(valid):.2f}  max={valid.max():.2f}")
    print(f"valid depth: {valid.size / frame.depth.size:.1%} of pixels")

    h, w = frame.depth.shape
    print(f"center pixel depth: {frame.depth[h // 2, w // 2]:.2f} m")

    # Visualize: RGB, and depth as a colormap (near = blue, far = red).
    depth_norm = np.clip(frame.depth / max(valid.max(), 1e-6), 0, 1)
    depth_vis = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_vis[frame.depth == 0] = 0                       # paint holes black

    cv2.imshow("rgb", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow("depth (jet: near=blue, far=red, holes=black)", depth_vis)
    print("press any key in an image window to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()