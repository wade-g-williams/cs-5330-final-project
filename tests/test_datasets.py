"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Prove the loaders produce a *correct* Frame, not merely a well-shaped one -- shape
checks alone miss a wrong depth transform, which is a bug we already shipped once (74f62f8).
Needs the datasets downloaded; skips otherwise.
"""

import numpy as np
import pytest
import scipy.io as sio

from collision_avoidance.datasets.sunrgbd import _project_to_pixels

# SUN RGB-D's sensors top out around 8 m, and the loader zeroes anything beyond that.
MAX_SENSOR_RANGE_M = 8.0

# Frames spanning the three different camera intrinsics present in the mirror, so a
# hard-coded K can't accidentally pass. Index i is sample id i+1 (000001 is index 0).
REPROJECTION_FRAMES = [0, 1000, 3000]

pytestmark = pytest.mark.needs_data


def test_sunrgbd_frame_contract(sunrgbd_loader):
    frame = sunrgbd_loader[0]
    assert frame.rgb.dtype == np.uint8
    assert frame.depth.dtype == np.float32
    assert frame.rgb.shape[:2] == frame.depth.shape     # aligned
    valid = frame.depth[frame.depth > 0]
    assert 0.1 < np.median(valid) < 8.0                 # plausible indoor distance range in meters


def test_depth_map_is_metric_and_well_populated(sunrgbd_loader):
    """The machine-checkable half of what scripts/view_frame.py prints for a human to eyeball."""
    frame = sunrgbd_loader[0]
    depth = frame.depth

    assert np.isfinite(depth).all()                     # no NaN/inf leaking out of the projection
    assert (depth >= 0).all()                           # 0.0 means "no reading", never negative
    assert depth.max() <= MAX_SENSOR_RANGE_M            # the >8 m clamp actually ran

    # Rasterizing a cloud leaves holes, but a majority of pixels should still get a reading;
    # a badly wrong projection scatters points and craters this number.
    valid_fraction = float((depth > 0).mean())
    assert valid_fraction > 0.5, f"only {valid_fraction:.1%} of pixels have depth"


def test_intrinsics_are_untransposed(sunrgbd_loader):
    """calib/*.txt stores K transposed; _read_calib must undo it."""
    frame = sunrgbd_loader[0]
    h, w = frame.depth.shape

    # The giveaway of a transposed K: the principal point sits in the bottom row
    # (fx 0 0 / 0 fy 0 / cx cy 1) instead of the last column.
    assert np.allclose(frame.K[2], [0.0, 0.0, 1.0]), f"K looks transposed:\n{frame.K}"
    assert frame.fx > 0 and frame.fy > 0
    assert 0.25 * w < frame.cx < 0.75 * w               # principal point near the image centre
    assert 0.25 * h < frame.cy < 0.75 * h


@pytest.mark.parametrize("index", REPROJECTION_FRAMES)
def test_depth_reprojection_is_aligned_to_the_image(sunrgbd_loader, index):
    """
    Score the loader's projection against ground truth that ships with the data.

    The stored cloud carries its OWN per-point colour, which is an objective oracle: push
    the points through the loader's real projection, then compare each point's colour to
    the photo pixel it lands on. The correct transform makes those agree; a wrong one
    smears colours across the scene and flings points off-frame.

    Measured margins (see the three candidate transforms we compared before fixing this):
        correct   99.6-99.8% in frame, MAE  2.3-8.7
        no tilt   31-84%     in frame, MAE 44-108
        inverted  0-67%      in frame, MAE 31-129
    The thresholds below sit ~3x clear of either side, so this fails loudly on a
    regression without being flaky.
    """
    frame = sunrgbd_loader[index]
    paths = frame.meta["paths"]

    pts = sio.loadmat(str(paths["depth"]))["instance"].astype(np.float64)
    rtilt, K = sunrgbd_loader._read_calib(paths["intr"])

    # The SAME projection the loader uses for depth -- so this can't drift out of sync
    # with the code it is testing.
    u, v, _z, in_bounds = _project_to_pixels(pts[:, :3], rtilt, K, frame.depth.shape)

    # Point colours live in columns 3..5, normalised to [0, 1], already in RGB order --
    # which is Frame's convention too, so no channel swap here.
    colors = (np.clip(pts[:, 3:6], 0.0, 1.0) * 255).astype(np.uint8)

    in_frame = float(in_bounds.mean())
    sampled = frame.rgb[v[in_bounds], u[in_bounds]].astype(np.float64)
    color_mae = float(np.abs(sampled - colors[in_bounds].astype(np.float64)).mean())

    assert in_frame > 0.99, f"only {in_frame:.1%} of points reproject into the image"
    assert color_mae < 20.0, (f"colour MAE {color_mae:.1f} -- the cloud does not line up "
                              f"with the photo, so the tilt transform is wrong")


# --- KITTI ---------------------------------------------------------------------------

def test_kitti_frame_contract(kitti_loader):
    frame = kitti_loader[0]
    assert frame.rgb.dtype == np.uint8
    assert frame.depth.dtype == np.float32
    assert frame.rgb.shape[:2] == frame.depth.shape
    assert frame.meta["dataset"] == "kitti"
    assert frame.meta["depth_source"] == "depth_anything_v2"   # KITTI has no depth sensor


@pytest.mark.parametrize("index", [0, 1, 3000])
def test_kitti_intrinsics_come_from_p2(kitti_loader, index):
    """P2 is the left colour camera's 3x4 projection; K is its left 3x3 block."""
    frame = kitti_loader[index]
    h, w = frame.depth.shape

    assert np.allclose(frame.K[2], [0.0, 0.0, 1.0])       # bottom row -> not transposed
    assert frame.fx > 0 and frame.fy > 0
    assert frame.fx == pytest.approx(frame.fy, rel=1e-6)  # rectified KITTI has square pixels
    assert 0.25 * w < frame.cx < 0.75 * w
    assert 0.25 * h < frame.cy < 0.75 * h


def test_kitti_labels_parse(kitti_loader):
    """Ground-truth boxes must come back with sane geometry, since eval scores against them."""
    labels = kitti_loader.read_labels(0)
    assert labels, "frame 000000 has annotations"
    for lab in labels:
        x1, y1, x2, y2 = lab["bbox"]
        assert x2 > x1 and y2 > y1
        assert all(d > 0 for d in lab["dimensions"])      # h, w, l in meters
        assert lab["location"][2] > 0                     # Z forward, in front of the camera


def test_sunrgbd_labels_parse(sunrgbd_loader):
    """SUN label/*.txt lines must parse into 2D boxes that sit on the image."""
    frame = sunrgbd_loader[0]
    labels = sunrgbd_loader.read_labels(0)
    assert labels, "frame 000001 has annotations"
    h, w = frame.rgb.shape[:2]
    for lab in labels:
        x1, y1, x2, y2 = lab["bbox"]
        assert x2 > x1 and y2 > y1
        assert 0 <= x1 < w and 0 <= y1 < h
        assert x2 <= w + 1 and y2 <= h + 1          # boxes can clip the border slightly
        assert len(lab["location"]) == 3
        assert len(lab["dimensions"]) == 3
        assert len(lab["orientation"]) == 2
        assert np.isfinite(lab["rotation_y"])


def test_sunrgbd_labels_cover_benchmark_classes(sunrgbd_loader):
    """Across a few frames we should see some of the 10 VoteNet class names."""
    benchmark = {
        "bed", "table", "sofa", "chair", "toilet",
        "desk", "dresser", "night_stand", "bookshelf", "bathtub",
    }
    seen = set()
    for index in (0, 100, 1000, 3000):
        for lab in sunrgbd_loader.read_labels(index):
            if lab["type"] in benchmark:
                seen.add(lab["type"])
    assert seen, "expected at least one VoteNet class in the sampled frames"


def test_sunrgbd_split_indices(sunrgbd_loader):
    train = sunrgbd_loader.split_indices("train")
    val = sunrgbd_loader.split_indices("val")
    assert len(train) == 5285
    assert len(val) == 5050
    assert set(train).isdisjoint(set(val))
    assert sunrgbd_loader.samples[val[0]]["id"] == "000001"
    assert sunrgbd_loader.samples[train[0]]["id"] == "005051"


def test_kitti_lidar_and_projection(kitti_loader):
    """Velodyne is optional; skip if --with-lidar was not used."""
    if kitti_loader.samples[0]["velodyne"] is None:
        pytest.skip("KITTI velodyne/ not present — "
                    "run `python scripts/download_datasets.py kitti --with-lidar`")

    from collision_avoidance.datasets.kitti import project_lidar_to_image

    pts = kitti_loader.read_lidar(0)
    assert pts.dtype == np.float32
    assert pts.ndim == 2 and pts.shape[1] == 4
    assert pts.shape[0] > 10_000

    frame = kitti_loader[0]
    calib = kitti_loader.read_calib(0)
    u, v, z, in_bounds = project_lidar_to_image(
        pts, calib["P2"], calib["R0_rect"], calib["Tr_velo_to_cam"], frame.rgb.shape[:2],
    )
    assert in_bounds.sum() > 1_000
    depth = z[in_bounds]
    assert depth.min() > 1.0
    assert depth.max() < 80.0
