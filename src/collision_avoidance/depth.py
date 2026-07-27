"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Metric depth from a single RGB image, via Depth Anything V2. This is what stands in
for a depth sensor on KITTI, and what we compare against real sensor depth on SUN RGB-D.
"""

import numpy as np

from . import config


class MetricDepthEstimator:
    """Depth Anything V2 (metric variants) wrapped to return meters, same contract as a sensor.

    The checkpoints are domain-specific and NOT interchangeable -- Hypersim is trained to
    20 m indoors, VKITTI to 80 m outdoors. Crossing them silently breaks metric scale, so
    the domain is a required argument rather than a default.
    """

    def __init__(self, domain: str = "outdoor", device: str | None = None):
        if domain not in config.DEPTH_MAX_M:
            raise ValueError(f"domain must be 'indoor' or 'outdoor', got {domain!r}")

        import torch                                                  # local: keeps the core install torch-free
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.domain = domain
        self.max_depth = config.DEPTH_MAX_M[domain]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_id = config.DEPTH_MODEL_OUTDOOR if domain == "outdoor" else config.DEPTH_MODEL_INDOOR
        self.processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=config.MODELS_DIR)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, cache_dir=config.MODELS_DIR)
        self.model.to(self.device).eval()
        self._torch = torch

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB -> (H, W) float32 metric depth in meters."""
        torch = self._torch
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        # The model predicts at its own working resolution; resample back to the image grid.
        depth = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = depth.float().cpu().numpy().astype(np.float32)
        depth[depth > self.max_depth] = 0.0        # beyond the checkpoint's range = no reading
        depth[depth <= 0] = 0.0
        return depth
