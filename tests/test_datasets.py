"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: 
"""

import numpy as np
from collision_avoidance import config
from collision_avoidance.datasets.sunrgbd import SunRGBDLoader


def test_sunrgbd_frame_contract():
    frame = SunRGBDLoader(config.SUNRGBD_ROOT)[0]
    assert frame.rgb.dtype == np.uint8
    assert frame.depth.dtype == np.float32
    assert frame.rgb.shape[:2] == frame.depth.shape     # aligned
    valid = frame.depth[frame.depth > 0]
    assert 0.1 < np.median(valid) < 8.0                 # plausible indoor distance range in meters