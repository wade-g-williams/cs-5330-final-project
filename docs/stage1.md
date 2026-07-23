# Stage 1 — 2D Object Detection

**Goal of this stage:** take the `frame.rgb` image that [Stage 0](stage0.md) produces, run a **YOLO11**
object detector on it, and get back a list of objects — each with a **2D bounding box**, a **class label**
(remapped from COCO into the dataset's vocabulary), and a **confidence score**. Then draw those boxes on the
image and confirm they land on real objects.

This is the **semantic** half of the pipeline: it answers *"what is that, and where is it in the image?"*
It runs in parallel with the **geometry** half (Stages 2–4, which turn depth into 3D obstacle clusters), and
the two meet in Stage 5 (fusion). You can build and test Stage 1 completely independently of the depth side
— all it needs is `frame.rgb`.

> This assumes [Stage 0](stage0.md) works: `SunRGBDLoader(config.SUNRGBD_ROOT)[i]` returns a `Frame` with
> `rgb` (RGB, uint8), `depth` (meters), and `K`. We only touch `frame.rgb` here.

> **Prerequisite:** YOLO needs PyTorch. Install it **before** `ultralytics` so you don't get the CPU-only
> torch wheel — see [implementation.md §5](implementation.md#5-environment-setup). For a first pass, CPU is
> fine (YOLO11n runs at ~18 fps on CPU); the GPU just makes it faster.

---

## 1. Where this sits in the pipeline

```
frame.rgb ──► [ STAGE 1: YOLO11 ] ──► list[Detection]  ─┐
                                                         │  (semantic branch)
                                                         ▼
frame.depth + frame.K ──► [ STAGE 2–4: geometry ] ──► 3D clusters ──► [ STAGE 5: FUSION ]
```

Detection gives you **labels**; geometry gives you **completeness** (obstacles the detector was never
trained on). Stage 5 fuses them so each obstacle ends up with both a class and a metric 3D position. Keep
that end goal in mind — it's *why* Stage 1's output is shaped the way it is.

---

## 2. Concepts you need

### What an object detector produces

For each object it finds, a detector outputs three things:

- a **bounding box** — an axis-aligned rectangle in **pixel coordinates**. We use the `xyxy` format:
  `(x1, y1, x2, y2)` = top-left corner and bottom-right corner.
- a **class label** — which of its known categories the object is.
- a **confidence score** — how sure it is, in `[0, 1]`.

(Contrast with *classification*, which labels the whole image, and *segmentation*, which labels every
pixel. Detection is the middle ground: per-object boxes.)

**YOLO** ("You Only Look Once") predicts every box in the image in a **single forward pass**, which is why
it runs in real time. YOLO11 is the current Ultralytics generation; we use it **pretrained on COCO** and do
no training of our own for the core pipeline.

### COCO, and why we remap classes

YOLO11 is pretrained on **COCO**, an 80-class everyday-object dataset. Its indoor classes include `chair`,
`couch`, `bed`, `dining table`, `toilet`, `potted plant`, `tv`, and so on. But our benchmark, **SUN
RGB-D**, scores a *different* 10-class vocabulary (`bed, table, sofa, chair, toilet, desk, dresser,
night_stand, bookshelf, bathtub`). The names don't line up — COCO says `couch`, SUN RGB-D says `sofa`; COCO
says `dining table`, SUN RGB-D says `table`.

So Stage 1 needs a **class map**: a translation from COCO names to the dataset's labels. Crucially, the map
is **partial** — some SUN RGB-D classes have no COCO equivalent at all (§4). That's expected and it's the
whole reason clustering (Stage 4) exists.

### Confidence threshold and NMS — handled for you

Two standard post-processing steps happen *inside* Ultralytics, so you don't implement them, but you should
know they're there:

- **Confidence threshold** (`conf`, default 0.25): boxes below this score are dropped. Raise it for fewer,
  surer boxes; lower it to catch more at the cost of false positives.
- **Non-Maximum Suppression (NMS):** when the detector fires several overlapping boxes for the *same*
  object, NMS keeps the highest-scoring one and suppresses the rest. Without it you'd get three boxes on one
  chair.

### Why detection alone isn't enough

The detector only knows its 80 COCO classes. A cardboard box, a pallet, a trash can, a backpack on the
floor — anything outside those 80 — gets **no box**. For a collision-avoidance system that's dangerous: an
unlabeled obstacle is still an obstacle. That gap is exactly what the **geometry branch** fills: Stage 4
clusters raw 3D points into blobs regardless of class, so unknown obstacles are still caught (and labeled
`"unknown"`). Detection supplies semantics; geometry supplies completeness. Neither is sufficient alone.

### PyTorch vs ONNX

Two ways to run the same model:

- **PyTorch** (Ultralytics' native API) — the simplest to start with: `YOLO("yolo11n.pt")`. Flexible, one
  line, auto-downloads weights.
- **ONNX** — a portable, optimized format for **deployment** (fast, framework-independent CPU inference).
  The proposal targets ONNX for the ~18 fps CPU number.

We build on PyTorch first (§5) and export to ONNX later (§7). **Note:** the known metric-depth *ONNX
accuracy bug* is a **Depth Anything V2** problem (Stage 0's KITTI branch), **not** a YOLO problem — YOLO's
ONNX export is safe to use.

---

## 3. The `Detection` dataclass

Just as `Frame` is the contract Stage 0 hands downstream, `Detection` is the contract Stage 1 hands to
Stage 5. Define it in `detection.py`.

```python
# src/collision_avoidance/detection.py  (part 1)
from dataclasses import dataclass


@dataclass
class Detection:
    """One detected object, produced by Stage 1 and consumed by Stage 5 (fusion)."""

    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2) in pixels, in the ORIGINAL image
    score: float                              # confidence in [0, 1]
    coco_name: str                            # the raw class the detector reported (e.g. "couch")
    label: str                                # dataset label if COCO maps to one, else == coco_name
    is_dataset_class: bool                    # True if `label` is one of the benchmark's scored classes
```

**Why both `coco_name` and `label`?** `coco_name` is the ground truth of what the *detector* said; `label`
is what *we* call it after remapping. Keeping both means evaluation can filter to the scored classes
(`is_dataset_class`), while the collision-avoidance map can still show *every* detected object as an
obstacle (using `label`, which falls back to the COCO name when there's no dataset equivalent). This is the
plan's "keep all detections as obstacles" decision made concrete.

---

## 4. The COCO → SUN RGB-D class map

Here's the honest mapping. Of SUN RGB-D's 10 scored classes, **only 5 have a COCO equivalent**:

| SUN RGB-D class | COCO class | Mappable? |
|-----------------|------------|-----------|
| bed | `bed` | ✅ |
| table | `dining table` | ✅ |
| sofa | `couch` | ✅ |
| chair | `chair` | ✅ |
| toilet | `toilet` | ✅ |
| desk | — | ❌ (no COCO class) |
| dresser | — | ❌ |
| night_stand | — | ❌ |
| bookshelf | — | ❌ |
| bathtub | — | ❌ |

The five unmappable classes will never get a detection box — they can only surface as **"unknown" clusters**
from the geometry branch (Stage 4). This is precisely the proposal's stated protocol: *"evaluate the classes
that are scoreable, with the rest appearing as unknown obstacles."* Being upfront about this in the report
is part of an honest evaluation.

Put the map (and a couple of related constants) in `config.py`:

```python
# src/collision_avoidance/config.py   (add to the Stage 0 constants)
# COCO class name -> SUN RGB-D benchmark label. Only 5 of SUN RGB-D's 10 classes
# have a COCO equivalent; the other five (desk, dresser, night_stand, bookshelf,
# bathtub) have none and can only appear as "unknown" clusters from Stage 4.
COCO_TO_SUNRGBD = {
    "bed": "bed",
    "dining table": "table",
    "couch": "sofa",
    "chair": "chair",
    "toilet": "toilet",
}

DETECTOR_CONF = 0.25          # confidence threshold
DETECTOR_MODEL = "yolo11n.pt" # nano; scale up to yolo11s/m/l/x for accuracy (you have the GPU for it)
```

When the KITTI loader arrives later, it gets its own map (`COCO_TO_KITTI` = `{"car": "Car", "person":
"Pedestrian", "bicycle": "Cyclist"}`), and the `Detector` takes whichever map matches the dataset — the
detection code itself doesn't change.

---

## 5. Implementing `detection.py`

A thin `Detector` class wraps Ultralytics, applies the class map, and returns `list[Detection]`.

```python
# src/collision_avoidance/detection.py  (part 2)
import cv2
import numpy as np
from ultralytics import YOLO

from .frame import Frame


class Detector:
    def __init__(self, model_path: str = "yolo11n.pt", conf: float = 0.25, class_map: dict | None = None):
        # YOLO() loads a .pt or .onnx model; a .pt is auto-downloaded on first use.
        # It uses the GPU automatically if one is available.
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_map = class_map or {}          # COCO name -> dataset label

    def detect(self, frame: Frame) -> list[Detection]:
        # ⚠️ Ultralytics expects a BGR numpy array; our Frame.rgb is RGB. Convert, or every
        #    detection is run on colour-swapped pixels and quietly gets worse. (See §9.)
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)

        result = self.model(bgr, conf=self.conf, verbose=False)[0]   # one image in, one Result out
        names = result.names                                         # {coco_id: coco_name}

        detections: list[Detection] = []
        for box, score, cls in zip(
            result.boxes.xyxy.cpu().numpy(),        # (N, 4) pixel coords in the ORIGINAL image
            result.boxes.conf.cpu().numpy(),        # (N,)
            result.boxes.cls.cpu().numpy().astype(int),   # (N,) COCO class ids
        ):
            coco_name = names[cls]
            label = self.class_map.get(coco_name, coco_name)   # remap, or fall back to the COCO name
            detections.append(Detection(
                bbox=tuple(float(v) for v in box),
                score=float(score),
                coco_name=coco_name,
                label=label,
                is_dataset_class=coco_name in self.class_map,
            ))
        return detections
```

**Read it in plain English.** `__init__` loads the model once (loading is slow; inference is fast — never
reload per frame). `detect` converts the frame to BGR, runs one forward pass, then walks the results:
Ultralytics hands back boxes in `xyxy` **original-image pixel coordinates** (it letterboxes to 640 px
internally and rescales the boxes back for you), plus a confidence and a COCO class id per box. For each, we
look up the COCO name, remap it through the class map (falling back to the COCO name if unmapped), and pack
it into a `Detection`.

**The one line that bites people** is `cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)`. Stage 0 deliberately
stores `Frame.rgb` in **RGB** order, but Ultralytics — built around OpenCV — expects **BGR** numpy arrays.
Skip the conversion and detection still "works," just worse and inconsistently, which is the nastiest kind
of bug. Stage 1 owns this conversion so nothing downstream has to think about it.

---

## 6. The smoke test — your verification loop

```python
# scripts/smoke_test_stage1.py
import argparse

import cv2

from collision_avoidance import config
from collision_avoidance.datasets.sunrgbd import SunRGBDLoader
from collision_avoidance.detection import Detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="which frame to run on")
    ap.add_argument("--model", default=config.DETECTOR_MODEL)
    args = ap.parse_args()

    frame = SunRGBDLoader(config.SUNRGBD_ROOT)[args.index]
    detector = Detector(args.model, conf=config.DETECTOR_CONF, class_map=config.COCO_TO_SUNRGBD)
    detections = detector.detect(frame)

    print(f"{len(detections)} detections on frame {frame.frame_id}:")
    for d in detections:
        tag = d.label if d.is_dataset_class else f"{d.coco_name} (not a scored class)"
        print(f"  {tag:24s} score={d.score:.2f}  bbox={tuple(round(v) for v in d.bbox)}")

    # Draw: green box = a scored SUN RGB-D class, orange box = an obstacle outside the 10 classes.
    bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        color = (0, 200, 0) if d.is_dataset_class else (0, 165, 255)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, f"{d.label} {d.score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imshow("stage 1 — detections (green=scored, orange=other obstacle)", bgr)
    print("press any key in the image window to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

Run it:

```bash
python scripts/smoke_test_stage1.py --index 0
```

The first run downloads `yolo11n.pt` (a few MB). You should see the RGB frame with labeled boxes on the
furniture, and a printout of each detection.

---

## 7. ONNX export (deployment / speed)

Once detection works in PyTorch, export the model to ONNX for the portable, fast-on-CPU path the proposal
targets:

```python
from ultralytics import YOLO
YOLO("yolo11n.pt").export(format="onnx")     # writes yolo11n.onnx next to the .pt
```

Then point the **same** `Detector` at the ONNX file — no other code changes, because Ultralytics wraps ONNX
inference behind the identical Results interface:

```python
detector = Detector("yolo11n.onnx", conf=config.DETECTOR_CONF, class_map=config.COCO_TO_SUNRGBD)
```

Move the export into `models/` (`models/yolo11n.onnx`) to match the repo layout. For GPU ONNX inference,
install `onnxruntime-gpu` instead of `onnxruntime`. Unlike Depth Anything V2, **YOLO's ONNX export has no
accuracy issue** — it's safe to use for evaluation as well as demos.

---

## 8. Verification checklist

Stage 1 passes when **all** of these hold — look, don't assume:

- [ ] **Boxes land on real objects** — the green/orange rectangles sit on actual furniture/objects, not on
      blank wall.
- [ ] **The class map works** — a sofa in the scene is labeled **`sofa`** (not `couch`), a table is
      **`table`** (not `dining table`). If you see the COCO names on those, the map isn't wired in.
- [ ] **Unmapped obstacles are kept** — a detected `person` or `potted plant` still appears (in orange) as
      an obstacle, tagged "not a scored class." Nothing an obstacle-avoider cares about is silently dropped.
- [ ] **Confidences are sane** — clear objects score high (> 0.5); borderline ones sit near the threshold.
      If *everything* is 0.9+ or *nothing* clears 0.25, revisit `conf` and the RGB/BGR conversion.
- [ ] **Boxes are in original-image pixels** — a box's corners match where the object is in the full-res
      `frame.rgb` (Ultralytics rescales from its internal 640 px back to your resolution).
- [ ] **A couple of frames run** without error and in a fraction of a second each.

A minimal test locks in the contract:

```python
# tests/test_detection.py
from collision_avoidance import config
from collision_avoidance.datasets.sunrgbd import SunRGBDLoader
from collision_avoidance.detection import Detector, Detection


def test_detector_returns_detections():
    frame = SunRGBDLoader(config.SUNRGBD_ROOT)[0]
    dets = Detector(config.DETECTOR_MODEL, conf=0.25, class_map=config.COCO_TO_SUNRGBD).detect(frame)
    assert all(isinstance(d, Detection) for d in dets)
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        assert x2 > x1 and y2 > y1          # valid box
        assert 0.0 <= d.score <= 1.0
        # a mapped class must carry a dataset label, not the raw COCO name
        if d.coco_name in config.COCO_TO_SUNRGBD:
            assert d.label == config.COCO_TO_SUNRGBD[d.coco_name]
```

---

## 9. Common pitfalls

- **RGB vs BGR (the big one).** `Frame.rgb` is RGB; Ultralytics expects BGR numpy arrays. Convert inside
  `detect()` (`cv2.COLOR_RGB2BGR`). Symptom if you forget: detections still appear but accuracy quietly
  drops and behaves oddly on color-sensitive classes — a silent degradation, not a crash. (Alternatively you
  could pass the image *file path* to Ultralytics, which handles color itself — but our pipeline works from
  in-memory `Frame`s, so convert.)
- **Reloading the model every frame.** `YOLO(...)` is slow to construct; build the `Detector` **once** and
  reuse it across frames. Constructing it inside a per-frame loop will crawl.
- **Where the weights live.** `YOLO("yolo11n.pt")` downloads to the current directory on first use. Keep
  weights under `models/` (they're gitignored) and pass the path, so runs are reproducible and you're not
  re-downloading.
- **Confidence threshold mis-set.** Too high → real obstacles vanish (dangerous for collision avoidance);
  too low → false-positive boxes on clutter. Start at 0.25 and tune while watching the smoke test.
- **COCO id vs name confusion.** Map by **name** (`result.names[id]`), not by hard-coded integer id —
  clearer, and it won't silently break if an id ordering ever differs between exports.
- **torch-before-ultralytics install order.** If you installed `ultralytics` first, you likely pulled the
  CPU-only torch and lost the GPU. Fix per [implementation.md §5](implementation.md#5-environment-setup):
  install CUDA `torch` first, then `ultralytics`.

---

## 10. Handoff to the next stages

Stage 1 now turns any `Frame` into a `list[Detection]`, each with a pixel box, a score, and a label
(dataset class or kept-as-obstacle fallback). Two threads pick this up:

- **Stage 5 (fusion)** is where detections *matter*: each 3D cluster from the geometry branch is projected
  into the image and matched against these boxes, so a cluster that overlaps a `sofa` box becomes a labeled
  `sofa` obstacle, and a cluster that overlaps nothing becomes `"unknown"`.
- **Stage 2 (back-projection)** is **independent** of Stage 1 — it needs only `frame.depth` and `frame.K` —
  and it's the next thing to build in the roadmap. Detection and geometry proceed on separate tracks until
  they meet in fusion.

**Next:** `stage2.md` — back-projecting the depth map through `K` into a metric 3D point cloud (the
"pseudo-LiDAR" scan), the first step of the geometry branch.
