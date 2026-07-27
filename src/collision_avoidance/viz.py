"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Shared drawing helpers. Everything here returns a BGR image ready for
cv2.imwrite/imshow -- keep drawing out of the pipeline stages so they stay testable.
"""

import cv2
import numpy as np

from .detection import Detection
from .frame import Frame

SCORED_COLOR = (0, 200, 0)      # BGR green  -- a class the benchmark scores
UNSCORED_COLOR = (0, 165, 255)  # BGR orange -- an obstacle outside the scored classes


def draw_detections(frame: Frame, detections: list[Detection]) -> np.ndarray:
    """RGB frame + boxes -> annotated BGR image."""
    bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        color = SCORED_COLOR if d.is_dataset_class else UNSCORED_COLOR
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, f"{d.label} {d.score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return bgr


def depth_colormap(depth: np.ndarray, max_m: float | None = None) -> np.ndarray:
    """Metric depth -> jet BGR image. Near = blue, far = red, no-reading = black."""
    valid = depth[depth > 0]
    ceiling = max_m or (valid.max() if valid.size else 1.0)
    norm = np.clip(depth / max(ceiling, 1e-6), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth == 0] = 0
    return vis


def side_by_side(*images: np.ndarray) -> np.ndarray:
    """Horizontally stack images, padding to the tallest so mismatched sizes still stack."""
    height = max(im.shape[0] for im in images)
    padded = [cv2.copyMakeBorder(im, 0, height - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
              for im in images]
    return np.hstack(padded)
