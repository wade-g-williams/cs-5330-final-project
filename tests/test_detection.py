"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Stage 1 contract test -- lock the Detection output shape that Stage 5
fusion will consume, so a later refactor can't silently break it. This is the
stage1.md §8 contract check.
"""

from collision_avoidance import config
from collision_avoidance.datasets.sunrgbd import SunRGBDLoader
from collision_avoidance.detection import Detection, Detector


def test_detector_returns_detections():
    # Frame 1000 is the in-distribution couch scene: reliably yields detections.
    frame = SunRGBDLoader(config.SUNRGBD_ROOT)[1000]
    detector = Detector(config.DETECTOR_MODEL, conf=config.DETECTOR_CONF,
                        class_map=config.COCO_TO_SUNRGBD)
    detections = detector.detect(frame)

    assert len(detections) > 0                       # the couch / tables must be found
    for d in detections:
        assert isinstance(d, Detection)
        x1, y1, x2, y2 = d.bbox
        assert x2 > x1 and y2 > y1                    # a real, non-degenerate box
        assert 0.0 <= d.score <= 1.0                  # a valid confidence
        # A mapped COCO class must carry the dataset label, not the raw COCO name;
        # everything else keeps its COCO name and is flagged non-scored.
        if d.coco_name in config.COCO_TO_SUNRGBD:
            assert d.label == config.COCO_TO_SUNRGBD[d.coco_name]
            assert d.is_dataset_class
        else:
            assert d.label == d.coco_name
            assert not d.is_dataset_class
