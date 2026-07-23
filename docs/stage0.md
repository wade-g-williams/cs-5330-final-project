# Stage 0 — Data Foundation

**Goal of this stage:** load one frame of a dataset and get three things into memory in a standard form:

1. the **RGB image** (what the scene looks like),
2. the **depth map**, in **meters** (how far every pixel is),
3. the **camera intrinsics** `K` (the numbers that connect pixels to real-world geometry),

then **visualize** the frame and **sanity-check** that the depth is actually in meters. When
`scripts/smoke_test_stage0.py` shows you a sensible image, a sensible depth map, and prints plausible
distances, Stage 0 is done.

We build this on **SUN RGB-D** first because it ships **real sensor depth** — so you can build and test the
whole geometry side of the pipeline (Stages 2–4) on true depth, *before* standing up Depth Anything V2 for
the KITTI branch.

> New to this? Read it top to bottom — each section motivates the next. Already comfortable with cameras and
> depth? Skip to [§4 The `Frame` dataclass](#4-the-frame-dataclass) for the code.

---

## 1. Why this is the foundation

Look back at the [pipeline diagram](implementation.md#2-the-pipeline-at-a-glance). **Every** downstream
stage consumes the same three things:

- Stage 1 (detection) needs the **RGB image**.
- Stage 2 (back-projection) needs the **depth map** and the **intrinsics `K`**.
- Stages 3–6 build on Stage 2's output.

So until we can reliably produce `(rgb, depth-in-meters, K)` for a frame, there is nothing to detect on,
nothing to back-project, and nothing to test the rest of the pipeline against. Stage 0 is the contract the
whole system is built on. Get it right and everything after it has solid ground; get the depth units wrong
here and every distance in your final BEV map is silently off by a factor of 1000.

The other reason to nail Stage 0 first: **KITTI and SUN RGB-D store their data completely differently**
(different depth formats, different intrinsics files, different folder layouts), but the *rest* of the
pipeline shouldn't care. We solve that with one small abstraction — a `Frame` type and a `DatasetLoader`
interface — so Stage 1 onward is written **once** and every dataset plugs in behind it.

---

## 2. Concepts you need

### Depth map

A depth map is an image where each pixel value is a **distance**, not a color. Pixel `(u, v)` in the depth
map tells you how far the surface seen at color pixel `(u, v)` is from the camera.

The critical, error-prone detail is **units**. Depth sensors almost never store meters directly — they
store **millimeters as 16-bit integers** (a value of `2000` means 2.000 m), because integers are exact and
compact. Our `Frame` will always hold depth as **float32 meters**, so *every* loader is responsible for
converting into that one convention. A pixel with **no reading** (too far, too close, reflective surface,
or shadow) is stored as **0** — we treat 0 as "unknown," not "0 meters away."

SUN RGB-D adds a twist covered in [§5](#5-implementing-the-sun-rgb-d-loader): its 16-bit depth is
**bit-rotated** before it's in millimeters. This is exactly the kind of format quirk Stage 0 exists to
absorb.

### Camera intrinsics — the matrix `K`

Intrinsics are the four numbers that describe *this particular camera's* geometry, arranged in a 3×3 matrix:

```
K = ┌ fx   0   cx ┐
    │  0  fy   cy │
    └  0   0    1 ┘
```

- **fx, fy** — the focal length in **pixels** (horizontal and vertical). Bigger fx = narrower field of
  view = more "zoomed in."
- **cx, cy** — the **principal point**: the pixel where the optical axis hits the image, usually near the
  image center (≈ width/2, height/2).

Why we need them *now*: intrinsics are the dictionary that translates between **pixels** (what the image
gives you) and **metric 3D geometry** (what the BEV map needs). Without `K`, a depth map is just a
grayscale picture; with `K`, it becomes a 3D scan. Every camera has its own `K`, and SUN RGB-D even mixes
several sensor types, so we read `K` **per frame**, never hard-code it.

### Aligned (registered) depth

The RGB camera and the depth camera are physically separate lenses a few centimeters apart, so their raw
images don't line up pixel-for-pixel. **Aligned** (or *registered*) depth has already been warped so that
depth pixel `(u, v)` corresponds to color pixel `(u, v)`. SUN RGB-D provides aligned depth out of the box,
and Depth Anything V2's output is aligned by construction (it's predicted *from* the color image). Raw
RealSense needs alignment via its SDK — a Stage-0 concern for the extension, not now. **For this stage,
assume aligned depth and verify it** (see the checklist).

### Forward pointer: the pinhole model (Stage 2)

Here's *why* meters + `K` are the two things Stage 0 must deliver. In Stage 2, each pixel `(u, v)` with
depth `Z` becomes a 3D point in the camera's frame using the **pinhole camera model**:

```
X = (u − cx) · Z / fx
Y = (v − cy) · Z / fy
Z =  Z
```

Notice the ingredients: pixel coordinates `(u, v)`, the intrinsics `(fx, fy, cx, cy)`, and the metric depth
`Z`. That's *exactly* the `Frame`. Stage 0 exists to hand Stage 2 those ingredients in clean, correct
units. (You don't implement this yet — it just explains the design.)

---

## 3. The plan for Stage 0

Three small pieces, built in this order:

1. `frame.py` — the `Frame` dataclass: the standard container every loader returns.
2. `datasets/base.py` — the `DatasetLoader` interface every dataset implements.
3. `datasets/sunrgbd.py` — the first concrete loader.

Then `scripts/smoke_test_stage0.py` proves it works.

---

## 4. The `Frame` dataclass

This is the single most important type in the codebase: the **contract** between "data" and "pipeline."
Write it first.

```python
# src/collision_avoidance/frame.py
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Frame:
    """One RGB-D frame in a standard form that every pipeline stage understands.

    This is the contract: every dataset loader returns a Frame, and every stage
    (detection, back-projection, …) consumes one. Units and conventions are fixed
    here so no downstream code ever has to guess.
    """

    rgb: np.ndarray          # (H, W, 3) uint8, RGB channel order (NOT OpenCV's BGR)
    depth: np.ndarray        # (H, W)   float32, metric depth in METERS; 0.0 = no reading
    K: np.ndarray            # (3, 3)   float64 camera intrinsics
    frame_id: str = ""       # dataset-specific identifier, for logging/debugging
    meta: dict = field(default_factory=dict)   # anything extra (sensor type, file paths, …)

    def __post_init__(self):
        # Cheap invariants that catch the most common loader bugs immediately.
        assert self.rgb.ndim == 3 and self.rgb.shape[2] == 3, f"rgb must be HxWx3, got {self.rgb.shape}"
        assert self.rgb.dtype == np.uint8, f"rgb must be uint8, got {self.rgb.dtype}"
        assert self.depth.ndim == 2, f"depth must be HxW, got {self.depth.shape}"
        assert self.rgb.shape[:2] == self.depth.shape, (
            f"rgb {self.rgb.shape[:2]} and depth {self.depth.shape} must be the same size (aligned)"
        )
        assert self.K.shape == (3, 3), f"K must be 3x3, got {self.K.shape}"

    # Convenient named access to the intrinsics — reads better than K[0, 0] everywhere.
    @property
    def fx(self) -> float: return float(self.K[0, 0])
    @property
    def fy(self) -> float: return float(self.K[1, 1])
    @property
    def cx(self) -> float: return float(self.K[0, 2])
    @property
    def cy(self) -> float: return float(self.K[1, 2])
```

**Why a `dataclass`?** It gives you `__init__`, a readable `repr`, and named fields for free — the point of
Stage 0 is a clean, self-documenting container, and a dataclass is exactly that with no boilerplate.

**Why the asserts?** They encode the three conventions that, if broken, cause the nastiest bugs — wrong
channel order, wrong depth units, misaligned depth. A loader with a bug trips the assert *at load time* with
a clear message, instead of producing a garbage BEV map five stages later. This is the "fix the root cause,
provide a feedback loop" habit baked into the type itself.

**The conventions, stated once:** RGB order (not BGR), depth in float32 **meters** with 0 = no reading, `K`
as a 3×3 float64 matrix, and `rgb`/`depth` the **same H×W** (i.e., aligned). Every loader obeys these; no
stage ever re-checks them.

---

## 5. The `DatasetLoader` interface

Now the abstraction that lets KITTI, nuScenes, or your own RealSense recordings drop in later **without
touching a single line of the pipeline**. An *abstract base class* (ABC) defines the shape every loader must
have; the pipeline programs against the shape, not the specific dataset.

```python
# src/collision_avoidance/datasets/base.py
from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..frame import Frame


class DatasetLoader(ABC):
    """Interface every dataset loader implements.

    Behaves like a read-only sequence of Frames: len(loader) frames, indexable
    with loader[i], and iterable with `for frame in loader`.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Number of frames available."""

    @abstractmethod
    def __getitem__(self, index: int) -> Frame:
        """Load and return frame `index` as a Frame (rgb, depth-in-meters, K)."""

    def __iter__(self) -> Iterator[Frame]:
        # Free iteration for any subclass that implements __len__ and __getitem__.
        for i in range(len(self)):
            yield self[i]
```

**Why bother with an interface?** Because the moment `run_frame.py` says `loader = SunRGBDLoader(...)` and
later `loader = KittiLoader(...)` and *nothing else changes*, you've made the pipeline dataset-agnostic. The
proposal evaluates on two datasets and lists three extensions; this five-line ABC is what keeps that from
becoming five copies of the pipeline. It's also why the smoke test and every `scripts/` entry point can take
a `--dataset` flag.

**`__getitem__` loads lazily** — one frame at a time, on demand — so you never hold 10,000 frames in RAM.

---

## 6. Implementing the SUN RGB-D loader

This is the concrete work. **Before writing code, look at what you actually downloaded** — the exact
on-disk layout of the HuggingFace mirror determines your paths, and guessing here is how you waste an hour.

### 6a. Get the data and inspect it

`scripts/download_sunrgbd.py` will pull the mirror; the one-liner it wraps:

```python
from huggingface_hub import snapshot_download
path = snapshot_download("youdaoyzbx/processed_sunrgbd", repo_type="dataset")
print(path)   # cache location of the extracted dataset
```

Then **inspect it** — do not trust any layout from memory, including this doc:

```bash
# What's actually in there?
ls -R "<path from above>" | head -50

# Pick one depth file and check its dtype and value range — this tells you the units.
python - <<'PY'
import cv2, numpy as np
raw = cv2.imread("<path to one depth image>", cv2.IMREAD_UNCHANGED)
print("dtype:", raw.dtype, "shape:", raw.shape, "min:", raw.min(), "max:", raw.max())
PY
```

What you're checking:
- Is depth stored as **16-bit PNGs** (`dtype uint16`), or already-converted floats, or a `.bin`/`.npy`
  point cloud? mmdetection3d-style mirrors sometimes preprocess into point clouds — if so, your loader
  reads *that* instead of converting depth yourself, and this section simplifies. **Verify first.**
- Where do the **intrinsics** live — a per-frame `intrinsics.txt`, a shared calib file, or a metadata
  pickle/JSON?
- Are RGB and depth the **same resolution**? (They must be, for aligned depth.)

Fill the exact paths into the loader below once you've seen them.

### 6b. The loader

```python
# src/collision_avoidance/datasets/sunrgbd.py
from pathlib import Path

import cv2
import numpy as np

from .base import DatasetLoader
from ..frame import Frame


class SunRGBDLoader(DatasetLoader):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        # Build an index of samples ONCE. Each entry holds the file paths for one
        # frame. Adjust the globbing to the actual mirror layout you inspected in 6a.
        self.samples = self._index_samples()
        if not self.samples:
            raise FileNotFoundError(f"No SUN RGB-D samples found under {self.root}")

    def _index_samples(self) -> list[dict]:
        # EXAMPLE structure — replace with the real layout from step 6a.
        # Goal: for each frame, know its rgb / depth / intrinsics paths.
        samples = []
        for sample_dir in sorted(self.root.glob("<per-frame-dir-glob>")):
            samples.append({
                "id": sample_dir.name,
                "rgb": next(sample_dir.glob("image/*")),          # ← confirm subfolder names
                "depth": next(sample_dir.glob("depth/*")),        # ← confirm
                "intr": sample_dir / "intrinsics.txt",            # ← confirm
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Frame:
        s = self.samples[index]
        return Frame(
            rgb=self._read_rgb(s["rgb"]),
            depth=self._read_depth(s["depth"]),   # returns METERS
            K=self._read_intrinsics(s["intr"]),
            frame_id=s["id"],
            meta={"dataset": "sunrgbd", "paths": s},
        )

    # --- readers -----------------------------------------------------------------

    def _read_rgb(self, path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)   # OpenCV loads as BGR
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)     # → RGB, per our Frame convention

    def _read_depth(self, path) -> np.ndarray:
        # SUN RGB-D packs depth into 16-bit with a 3-bit rotation, THEN it's millimeters.
        # This mirrors the dataset's official read3dPoints.m. If your mirror already
        # provides converted depth or a point cloud (check in 6a), skip the bit rotate.
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)   # 16-bit, unaltered
        if raw is None:
            raise FileNotFoundError(f"could not read depth: {path}")
        assert raw.dtype == np.uint16, f"expected uint16 depth, got {raw.dtype}"

        raw = raw.astype(np.uint32)                          # avoid shift overflow ambiguity
        d = ((raw >> 3) | (raw << 13)) & 0xFFFF              # 16-bit rotate-right by 3
        depth_m = d.astype(np.float32) / 1000.0              # millimeters → meters
        depth_m[depth_m > 8.0] = 0.0                         # these sensors max ~8 m; treat far as invalid
        return depth_m

    def _read_intrinsics(self, path) -> np.ndarray:
        # SUN RGB-D stores a 3x3 camera matrix per frame (9 numbers, row-major).
        vals = np.loadtxt(str(path))
        return vals.reshape(3, 3).astype(np.float64)
```

**Read it in plain English.** `__init__` scans the dataset once and records, for each frame, where its RGB,
depth, and intrinsics files live. `__getitem__` reads those three files and packs them into a `Frame`. The
three `_read_*` helpers each handle one file and one convention: BGR→RGB for the image, the bit-rotate +
mm→m for depth, reshape-to-3×3 for intrinsics. The `Frame`'s own asserts then double-check the result.

**The one line that matters most** is inside `_read_depth`: `((raw >> 3) | (raw << 13)) & 0xFFFF`. SUN
RGB-D rotated the 16 depth bits right by 3 before storing; this rotates them back. Forget it and your depth
is scrambled — not merely scaled, but nonsense. This is the canonical Stage-0 gotcha, which is exactly why
Stage 0 owns it and no other stage ever sees it. **But** — if step 6a showed the mirror already gives you
converted depth or a point cloud, don't apply the rotate; adapt the reader to what's actually on disk.

---

## 7. The smoke test — your verification loop

A stage isn't done until you can *see* it working. This script is Stage 0's feedback loop.

```python
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
```

(`config.SUNRGBD_ROOT` is a path constant you set in `src/collision_avoidance/config.py` — point it at your
`data/sunrgbd/` extraction.)

Run it:

```bash
python scripts/smoke_test_stage0.py --index 0
```

---

## 8. Verification checklist

Stage 0 passes when **all** of these hold — check them, don't assume them:

- [ ] **RGB looks right** — recognizable indoor scene, natural colors (a blue-tinted image means you skipped
      BGR→RGB somewhere).
- [ ] **Depth is aligned** — the depth colormap's shapes line up with the RGB's objects (a near table is
      blue in the same place the table sits in the RGB).
- [ ] **Depth is in meters** — for an indoor scene, `min`/`median`/`max` land roughly in **0.5–8 m**. If you
      see values like 500–8000, you're still in millimeters (missing the `/1000`); if you see scrambled
      noise, the bit-rotate is wrong.
- [ ] **The center pixel reads a believable distance** — point the check at a frame where you can eyeball it
      (a wall a few meters back should read a few meters).
- [ ] **Intrinsics are sane** — `cx ≈ width/2`, `cy ≈ height/2`, and `fx`, `fy` are positive focal lengths
      in the low hundreds for VGA-scale images (they vary by sensor — SUN RGB-D mixes Kinect/Xtion/RealSense
      — so treat this as a plausibility check, not an exact value).
- [ ] **Holes are handled** — a modest fraction of pixels are 0 (no reading); those show black, and your
      depth stats were computed on `depth > 0` only.
- [ ] **A couple of random indices load** without tripping any `Frame` assert.

A tiny unit test locks in the contract so a future refactor can't quietly break it:

```python
# tests/test_datasets.py
import numpy as np
from collision_avoidance import config
from collision_avoidance.datasets.sunrgbd import SunRGBDLoader


def test_sunrgbd_frame_contract():
    frame = SunRGBDLoader(config.SUNRGBD_ROOT)[0]
    assert frame.rgb.dtype == np.uint8
    assert frame.depth.dtype == np.float32
    assert frame.rgb.shape[:2] == frame.depth.shape        # aligned
    valid = frame.depth[frame.depth > 0]
    assert 0.1 < np.median(valid) < 8.0                    # plausible indoor meters
```

---

## 9. Common pitfalls

- **BGR vs RGB.** OpenCV's `imread` returns **BGR**; our `Frame` wants **RGB**. Convert in the loader
  (`cv2.cvtColor(..., COLOR_BGR2RGB)`), not scattered through the pipeline. Symptom if you forget: everyone
  looks like a Smurf.
- **Depth units.** The two failure modes: forgetting `/1000` (depth comes out ~1000× too big, in mm), or
  skipping the SUN RGB-D bit-rotate (depth comes out as scrambled noise). The checklist's meters range
  catches both.
- **Zero / missing depth.** Never divide by, or take statistics over, the 0 pixels — they mean "no reading,"
  not "zero distance." Always mask with `depth > 0`. Left unmasked, they'll later back-project to a fake
  wall of points at the camera origin.
- **Wrong-resolution intrinsics.** `K` is only valid for the resolution it was measured at. If you ever
  resize the RGB or depth, you must scale `fx, fy, cx, cy` by the same factor. Best practice for Stage 0:
  keep the native resolution and don't resize yet (Stage 1 handles the detector's 640 px resize
  internally).
- **Assuming alignment.** Confirm depth and RGB line up (checklist item 2) rather than trusting a label. If
  a dataset gives unregistered depth, aligning it is a Stage-0 job — solve it here, once.
- **Hard-coding one camera's `K`.** SUN RGB-D mixes sensors; read intrinsics per frame. Hard-coding works on
  frame 0 and silently corrupts geometry on frame 1.

---

## 10. Handoff to Stage 1

You now produce a `Frame` on demand for any SUN RGB-D index. The rest of the pipeline consumes it:

- **Stage 1 (detection)** takes `frame.rgb` → 2D boxes + class labels.
- **Stage 2 (back-projection)** takes `frame.depth` + `frame.K` → a metric 3D point cloud, using the pinhole
  equations previewed in [§2](#forward-pointer-the-pinhole-model-stage-2).

When the KITTI branch comes later, you'll add `datasets/kitti.py` (same interface) and a Depth Anything V2
depth source in `depth.py` — and because they both produce a `Frame`, **Stages 1–6 won't change at all.**
That payoff is the whole reason Stage 0 looks the way it does.

**Next:** `stage1.md` — running YOLO11 on `frame.rgb` and mapping COCO's classes onto the dataset's labels.
