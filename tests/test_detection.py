"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Detector contract tests -- that it hands the model BGR (feeding it RGB doesn't raise,
it just quietly degrades every detection), and that Detection has the shape fusion needs.
"""

import numpy as np
import pytest

# detection.py imports ultralytics at module scope, and ultralytics is deliberately NOT a
# core dependency (see pyproject.toml), so skip this whole module rather than error on a
# torch-free install.
pytest.importorskip("ultralytics", reason="ultralytics not installed (see docs/implementation.md §5)")

from collision_avoidance import config                             # noqa: E402
from collision_avoidance import detection as detection_module      # noqa: E402
from collision_avoidance.detection import Detection                # noqa: E402
from collision_avoidance.frame import Frame                        # noqa: E402


class _FakeTensor:
    """Stands in for a torch tensor: Detector only ever calls .cpu().numpy() on these."""

    def __init__(self, array):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeBoxes:
    def __init__(self):
        self.xyxy = _FakeTensor(np.array([[10.0, 20.0, 30.0, 40.0]]))
        self.conf = _FakeTensor(np.array([0.9]))
        self.cls = _FakeTensor(np.array([57.0]))          # 57 == "couch" in COCO


class _FakeResult:
    def __init__(self):
        self.names = {57: "couch", 0: "person"}
        self.boxes = _FakeBoxes()


class _RecordingModel:
    """Captures the image Detector passes in, so the test can inspect channel order."""

    def __init__(self):
        self.seen_image = None

    def __call__(self, image, conf=None, verbose=None):
        self.seen_image = image
        return [_FakeResult()]


def _synthetic_frame(rgb: np.ndarray) -> Frame:
    h, w = rgb.shape[:2]
    return Frame(rgb=rgb, depth=np.ones((h, w), np.float32), K=np.eye(3))


def test_detector_feeds_bgr_to_the_model(monkeypatch):
    recorder = _RecordingModel()
    monkeypatch.setattr(detection_module, "YOLO", lambda path: recorder)

    # A pure-red image in our RGB convention. If Detector forwards it unconverted, the
    # model sees red in channel 0; converted correctly, red lands in channel 2 (BGR).
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[:, :, 0] = 255

    detector = detection_module.Detector("unused.pt", conf=0.25, class_map=config.COCO_TO_SUNRGBD)
    detector.detect(_synthetic_frame(rgb))

    seen = recorder.seen_image
    assert seen is not None, "Detector never called the model"
    assert seen.shape == rgb.shape and seen.dtype == np.uint8
    assert (seen[:, :, 2] == 255).all(), "red must arrive in the BGR red channel (index 2)"
    assert (seen[:, :, 0] == 0).all(), "channel 0 must be blue -- the RGB image was not converted"


def test_detector_maps_classes_without_touching_the_image(monkeypatch):
    """COCO name -> dataset label mapping, checked independently of any real inference."""
    monkeypatch.setattr(detection_module, "YOLO", lambda path: _RecordingModel())

    detector = detection_module.Detector("unused.pt", conf=0.25, class_map=config.COCO_TO_SUNRGBD)
    (det,) = detector.detect(_synthetic_frame(np.zeros((8, 8, 3), np.uint8)))

    assert isinstance(det, Detection)
    assert det.coco_name == "couch"
    assert det.label == "sofa"                # remapped to the SUN RGB-D benchmark label
    assert det.is_dataset_class
    assert det.bbox == (10.0, 20.0, 30.0, 40.0)
    assert det.score == pytest.approx(0.9)


@pytest.mark.needs_data
def test_detector_returns_detections(sunrgbd_loader, detector):
    # Frame 1000 is the in-distribution couch scene: reliably yields detections.
    frame = sunrgbd_loader[1000]
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
