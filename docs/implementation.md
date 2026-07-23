# Implementation Guide — RGB-D Object Detection for Collision Avoidance

**Start here.** This is the map for turning the [proposal](../report/proposal.pdf) into working code. It
lays out the pipeline stages, the repository layout, what every file does, how to set up the environment
and data, and the order to build things in. Each stage then gets its own deep-dive walkthrough:
[`stage0.md`](stage0.md) is written; `stage1.md` … `stage6.md` follow as we build.

> **A note on numbering.** The proposal describes a **six-stage** pipeline (Stage 1 = detection … Stage 6 =
> BEV map). But before any of those stages can run, you need a frame's RGB image, its depth map, and its
> camera intrinsics loaded into memory. That groundwork is **Stage 0 — the data foundation**. Numbering it
> "0" keeps the proposal's Stage 1–6 exactly as written while making the build order honest: you implement
> Stage 0 first. So "the first thing you implement" and "the proposal's first pipeline stage" are two
> different things, and that is deliberate.

---

## 1. What we're building

Every frame of an RGB-D (or RGB-plus-estimated-depth) video stream goes in; a **top-down Bird's-Eye-View
(BEV) occupancy map** with real-world distances comes out. On that map, each obstacle carries a class label
(or "unknown"), a metric 3D position, and a collision warning if it's too close.

The pipeline does this by combining a 2D object detector with geometry:

1. Detect and classify objects in the RGB image (YOLO).
2. Turn the depth map into a metric 3D point cloud (a "pseudo-LiDAR" scan).
3. Delete the ground plane so the floor/road isn't treated as an obstacle.
4. Cluster the remaining points into obstacle blobs — so objects the detector was **never trained on** are
   still caught.
5. Fuse detections with clusters, so each obstacle gets a class + a 3D centroid + a box.
6. Render it all onto a BEV occupancy grid with distance rings and a near-field warning zone.

**The headline experiment** is a scientific question, not just an engineering one: *can a monocular depth
estimate substitute for a real depth sensor?* We answer it by running the identical pipeline two ways on the
same indoor frames — once on real sensor depth, once on [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
predictions — and comparing the results. Outdoors on KITTI (which has no depth camera) we only have the
estimated-depth branch.

Background and rationale live in [`../report/proposal.pdf`](../report/proposal.pdf) (authoritative scope)
and [`project-ideas.md`](project-ideas.md) → *Idea 1* (the richest implementation detail: exact model
checkpoints, class mappings, and per-stage sub-steps). Dataset logistics and gotchas are in
[`datasets-and-simulators.md`](datasets-and-simulators.md).

---

## 2. The pipeline at a glance

```
                         ┌─────────────────────────────────────────────────────────────┐
                         │  STAGE 0 — DATA FOUNDATION                                   │
   dataset files  ─────► │  loader → Frame{ rgb, depth (meters), K (intrinsics) }       │
   (SUN RGB-D,           │  depth source = sensor (indoor)  OR  Depth Anything V2 (KITTI)│
    KITTI, …)            └───────────────┬─────────────────────────────┬───────────────┘
                                         │ rgb                          │ depth + K
                                         ▼                              ▼
                         ┌───────────────────────────┐    ┌───────────────────────────────┐
                         │ STAGE 1 — 2D DETECTION     │    │ STAGE 2 — BACK-PROJECTION      │
                         │ YOLO11 → boxes + classes   │    │ pinhole model → 3D point cloud │
                         │ (COCO → dataset labels)    │    │ X=(u−cx)·Z/fx  Y=(v−cy)·Z/fy   │
                         └─────────────┬──────────────┘    └───────────────┬───────────────┘
                                       │ detections                        │ point cloud
                                       │                                   ▼
                                       │                   ┌───────────────────────────────┐
                                       │                   │ STAGE 3 — GROUND REMOVAL       │
                                       │                   │ self-impl RANSAC plane fit     │
                                       │                   └───────────────┬───────────────┘
                                       │                                   ▼
                                       │                   ┌───────────────────────────────┐
                                       │                   │ STAGE 4 — CLUSTERING           │
                                       │                   │ self-impl Euclidean / DBSCAN   │
                                       │                   └───────────────┬───────────────┘
                                       │                                   │ 3D clusters
                                       └──────────────┬────────────────────┘
                                                      ▼
                                       ┌───────────────────────────────────┐
                                       │ STAGE 5 — FUSION                   │
                                       │ project clusters → image, match to │
                                       │ boxes → {class|"unknown", centroid,│
                                       │ box}                               │
                                       └─────────────────┬─────────────────┘
                                                         ▼
                                       ┌───────────────────────────────────┐
                                       │ STAGE 6 — BEV RENDER               │
                                       │ occupancy grid, class colors,      │
                                       │ distance rings, <1.5 m red zone    │
                                       └───────────────────────────────────┘
```

| Stage | Name | Input | Output | Core method | Module |
|-------|------|-------|--------|-------------|--------|
| **0** | Data foundation | dataset files | `Frame` = RGB + metric depth + `K` | dataset loaders + depth handling | `datasets/`, `frame.py`, `depth.py` |
| **1** | 2D detection | RGB image | 2D boxes + class labels | YOLO11 (COCO-pretrained, ONNX) | `detection.py` |
| **2** | Back-projection | depth + `K` | metric 3D point cloud | pinhole model (pseudo-LiDAR) | `geometry.py` |
| **3** | Ground removal | point cloud | cloud minus floor/road | **self-implemented** RANSAC plane fit | `ground.py` |
| **4** | Clustering | ground-free cloud | 3D obstacle clusters | **self-implemented** Euclidean / DBSCAN | `cluster.py` |
| **5** | Fusion | detections + clusters | per-obstacle `{class or "unknown", 3D centroid, box}` | project clusters → image, match to boxes | `fusion.py` |
| **6** | BEV render | fused obstacles | top-down occupancy map + warnings | grid raster, class colors, distance rings | `bev.py` |

**Why clustering matters (the design insight):** the detector only knows its 80 COCO classes. Clustering
works on raw geometry, so a cardboard box, a pallet, or a trash can with no COCO label still becomes a solid
3D blob and gets flagged as an "unknown obstacle." Detection gives you *semantics*; geometry gives you
*completeness*. Fusion (Stage 5) is where the two meet.

---

## 3. Repository & file structure

The code is an **installable Python package** (`collision_avoidance`) under a `src/` layout. This keeps
imports clean (`from collision_avoidance import geometry`), stops accidental "it works from this folder but
not that one" bugs, and scales cleanly across three people.

```
cs-5330-final-project/
├── pyproject.toml              # package metadata + dependencies (managed with uv)
├── README.md                   # public-facing project summary + links (already exists)
├── .gitignore                  # add: .venv/  data/  models/  __pycache__/  *.onnx  *.pt
│
├── data/                       # (gitignored) datasets live here — never committed
│   ├── sunrgbd/                #   SUN RGB-D (HuggingFace mirror extraction)
│   └── kitti/                  #   KITTI object (images + calib + labels; no LiDAR) — later
├── models/                     # (gitignored) model weights — never committed
│   ├── yolo11n.onnx            #   YOLO11 detector export (Stage 1)
│   └── depth_anything_v2/      #   DA-V2 metric checkpoints (Stage 0 depth for KITTI) — later
│
├── docs/                       # planning + implementation docs
│   ├── implementation.md       #   ← this file (the index)
│   ├── stage0.md               #   ← Stage 0 walkthrough (data foundation)
│   ├── stage1.md … stage6.md   #   ← added as each pipeline stage is built
│   ├── project-ideas.md        #   the 7-variant brainstorm; Idea 1 is our proposal, in depth
│   ├── research.md             #   hub doc: datasets/simulators/papers considered, with verdicts
│   ├── datasets-and-simulators.md  # download logistics + verified gotchas
│   ├── papers.md               #   venue-verified reading list (peer-reviewed only)
│   └── assignment.md           #   the course's final-project spec
│
├── report/                     # LaTeX — proposal now, IEEE 2-column final report later
│   ├── proposal.tex / .pdf     #   the submitted proposal (fphw class)
│   ├── fphw.cls                #   proposal class file (vendored)
│   └── IEEEtran.cls            #   final-report class file (vendored)
│
├── src/collision_avoidance/    # THE PACKAGE — one module per pipeline stage
│   ├── __init__.py             #   marks the package; may expose a top-level API
│   ├── config.py               #   constants + paths: data roots, grid resolution, warning
│   │                           #     distance (1.5 m), COCO→dataset class maps
│   ├── frame.py                #   Frame dataclass — the contract every loader returns and
│   │                           #     every stage consumes (rgb, depth in meters, K, meta)
│   ├── datasets/               #   Stage 0 — one loader per dataset, all returning Frame
│   │   ├── __init__.py
│   │   ├── base.py             #     DatasetLoader abstract base class (the interface)
│   │   ├── sunrgbd.py          #     SUN RGB-D loader (built first — real sensor depth)
│   │   └── kitti.py            #     KITTI loader (later — pairs with DA-V2 depth)
│   ├── depth.py                #   depth sources: sensor passthrough (now) + Depth Anything V2
│   │                           #     wrapper for KITTI's estimated-depth branch (later)
│   ├── detection.py            #   Stage 1 — YOLO11 wrapper + COCO→dataset class remap
│   ├── geometry.py             #   Stage 2 — pinhole back-projection, intrinsics math
│   ├── ground.py               #   Stage 3 — RANSAC plane fit + inlier (ground) removal
│   ├── cluster.py              #   Stage 4 — Euclidean / DBSCAN clustering
│   ├── fusion.py               #   Stage 5 — cluster ↔ detection matching
│   ├── bev.py                  #   Stage 6 — BEV occupancy-map rasterization + warnings
│   ├── pipeline.py             #   orchestrates Stage 0 → 6 for a single Frame
│   └── viz.py                  #   shared drawing helpers (boxes, depth colormap, BEV overlay)
│
├── scripts/                    # runnable command-line entry points (thin wrappers on the package)
│   ├── download_sunrgbd.py     #   fetch the SUN RGB-D mirror into data/
│   ├── smoke_test_stage0.py    #   Stage 0 verification: load a frame, visualize, print stats
│   ├── run_frame.py            #   run the full pipeline on one frame → save visualizations
│   └── run_sequence.py         #   run over a sequence → annotated RGB | BEV video
│
├── tests/                      # pytest unit tests (grow alongside the modules)
│   ├── test_frame.py
│   ├── test_datasets.py
│   └── test_geometry.py
│
└── eval/                       # evaluation harnesses (added with the later stages)
    ├── kitti_eval.py           #   KITTI 3D AP + centroid error binned by distance
    └── sunrgbd_eval.py         #   SUN RGB-D 10-class mAP @ 3D IoU 0.25 (VoteNet protocol)
```

**Separation of concerns.** The `src/collision_avoidance/` package holds *library* code — importable,
testable, side-effect-free functions and classes. `scripts/` holds *executables* — the thin
argument-parsing wrappers you actually run. `tests/` verifies the library. `eval/` scores the pipeline
against dataset ground truth. Keeping these apart is what makes the code reusable across the core pipeline
*and* the extensions (Habitat navigation, live RealSense demo) without rewrites.

**Never commit `data/` or `models/`.** Datasets are gigabytes and licensed (KITTI is CC BY-NC-SA); weights
are large binaries. Both are gitignored and reproduced from the download scripts / links, not from git.

---

## 4. File structure per stage

Which files you touch when you implement each stage. Early stages create files; later stages mostly *add*
to `pipeline.py`, `viz.py`, and `config.py`.

| Stage | Primary file(s) | Also touches | Verify with |
|-------|-----------------|--------------|-------------|
| **0 — Data foundation** | `frame.py`, `datasets/base.py`, `datasets/sunrgbd.py`, `depth.py` (sensor passthrough) | `config.py` (data paths), `scripts/download_sunrgbd.py`, `viz.py` | `scripts/smoke_test_stage0.py`, `tests/test_frame.py`, `tests/test_datasets.py` |
| **1 — Detection** | `detection.py` | `config.py` (class maps), `viz.py` (draw boxes) | run YOLO on a loaded frame; count/label boxes |
| **2 — Back-projection** | `geometry.py` | `pipeline.py`, `viz.py` (point-cloud view) | `tests/test_geometry.py` — round-trip a known pixel↔3D point |
| **3 — Ground removal** | `ground.py` | `pipeline.py`, `viz.py` (color inliers) | visualize: floor points removed, obstacles kept |
| **4 — Clustering** | `cluster.py` | `pipeline.py`, `viz.py` (color clusters) | visualize: each obstacle = one cluster |
| **5 — Fusion** | `fusion.py` | `pipeline.py` | each cluster gets a label or "unknown"; centroids sane |
| **6 — BEV render** | `bev.py` | `pipeline.py`, `viz.py` | `scripts/run_frame.py` → RGB + BEV side by side |
| **Eval** | `eval/kitti_eval.py`, `eval/sunrgbd_eval.py` | — | AP / mAP numbers vs published baselines |

Stage 0 is documented in full in [`stage0.md`](stage0.md). Each later `stageN.md` follows the same shape:
goal → concepts → the code → a runnable verification.

---

## 5. Environment setup

Python, with a **`uv`-managed virtual environment** in the repo root. (`uv` is the fast installer;
`uv pip install …` and `uv venv` mirror the usual commands.)

```bash
cd "cs-5330-final-project"
uv venv                      # creates .venv/ using your default Python (3.10–3.12 all fine)
source .venv/bin/activate
uv pip install -e .          # editable install of the collision_avoidance package
```

`pyproject.toml` declares the dependencies. The core set:

| Package | Used for | Introduced in |
|---------|----------|---------------|
| `numpy` | arrays, geometry, RANSAC, clustering | Stage 0 |
| `opencv-python` | image + depth I/O, drawing, colormaps | Stage 0 |
| `matplotlib` | quick plots, depth visualization | Stage 0 |
| `huggingface-hub` | download the SUN RGB-D mirror | Stage 0 |
| `scipy` | KD-tree for clustering neighbor queries | Stage 4 |
| `ultralytics` | YOLO11 detection + ONNX export | Stage 1 |
| `onnxruntime` | run the exported YOLO ONNX model | Stage 1 |
| `torch`, `torchvision` | Depth Anything V2; also YOLO's backend | Stage 0 (KITTI branch) |

Two things installed **outside** `pip`:
- **Depth Anything V2** is a GitHub repo, not a PyPI package — clone it (or add it as a git dependency) and
  download the metric checkpoints into `models/depth_anything_v2/`. Needed only when the KITTI branch comes
  online.
- **Habitat-Sim** (the navigation extension, if we reach it) installs via **conda**, not `uv`. When that
  time comes, make a *separate* conda environment for it — don't try to force it into the `uv` venv.

> **GPU:** run Depth Anything V2 and any fine-tuning on the GPU. The team has ample GPU access, so choose
> models for accuracy, not memory frugality.

---

## 6. Data setup

Everything goes under `data/` (gitignored). Get datasets in the order you need them.

**SUN RGB-D (first — indoor, real sensor depth).** Skip the official Princeton release: its label
extraction requires MATLAB (the #1 pain point in VoteNet issues). Use the pre-extracted HuggingFace mirror
[`youdaoyzbx/processed_sunrgbd`](https://huggingface.co/datasets/youdaoyzbx/processed_sunrgbd)
(mmdetection3d layout). `scripts/download_sunrgbd.py` will fetch it into `data/sunrgbd/`. **Spot-check a few
frames** against the raw data — confirm the depth-unit convention and intrinsics before trusting the loader
(see [`stage0.md`](stage0.md) §Pitfalls).

**KITTI (later — outdoor, estimated depth).** No registration needed: the files are on
[AWS Open Data](https://registry.opendata.aws/kitti/) (`s3://avg-kitti`) and a
[Kaggle mirror](https://www.kaggle.com/datasets/klemenko/kitti-dataset). Grab **images + calib + labels
(~12 GB) and skip the 29 GB LiDAR** — we use Depth Anything V2 for depth, not LiDAR. License CC BY-NC-SA 3.0.

**Own RealSense recordings (extension #1).** `pyrealsense2` ships Ubuntu-24.04 wheels; record short clips
(bags grow ~1 GB/min at 640×480@30) with obstacles at tape-measured distances — the only data where ground
truth is exactly controlled.

**Model weights** go under `models/` (also gitignored): the YOLO11 ONNX export, and the DA-V2 metric
checkpoints. Remember DA-V2's checkpoints are **domain-specific** — Indoor/Hypersim (max 20 m) for indoor
frames, Outdoor/VKITTI (max 80 m) for KITTI. Mixing them breaks metric scale.

---

## 7. Build roadmap & milestones

Today is **2026-07-22**. The progress check-in email is due **2026-07-31** and everything is due
**2026-08-13**. Build in dependency order; get an end-to-end vertical slice working on SUN RGB-D before
adding the KITTI branch.

**Week 1 (now → Jul 31) — a working indoor slice, on real sensor depth.**
- Stage 0: `Frame` + `DatasetLoader` + SUN RGB-D loader; smoke test passes. → [`stage0.md`](stage0.md)
- Stage 1: YOLO11 detecting on loaded frames; COCO→SUN RGB-D class map.
- Stage 2: back-project depth → point cloud; sanity-check with a known distance.
- **Jul 31:** send the check-in email (1-paragraph description + names + progress — reuse the README
  paragraph). Aim to report "Stages 0–2 working indoors."

**Week 2 (Aug 1 → Aug 7) — close the loop, add the outdoor branch.**
- Stages 3–6: RANSAC ground removal → clustering → fusion → BEV map, all on SUN RGB-D.
- First full BEV video from `scripts/run_sequence.py`.
- Bring in KITTI + Depth Anything V2 (outdoor checkpoint); get the same pipeline running outdoors.
- Stand up `eval/` harnesses.

**Week 3 (Aug 8 → Aug 13) — evaluate, ablate, present.**
- KITTI: 3D AP + centroid error by distance bin. SUN RGB-D: 10-class mAP@0.25.
- **The headline ablation:** sensor depth vs DA-V2 depth on identical SUN RGB-D frames.
- One extension if time allows, in priority order: own RealSense recordings → Habitat-Sim navigation.
- Record the demo video (host on Drive/YouTube — *not* Gradescope), write the IEEE 2-column report
  (≤ 8 pages), update the README with links.
- **Aug 13, 11:59 pm:** submit report + code + README to Gradescope. No extensions.

**Training add-ons are never on the critical path.** Each (detector fine-tune, depth-head fine-tune, learned
frustum head) is an *ablation against the zero-shot baseline*, trained only on train splits. A failed
training run costs an experiment, not the project. Pick at most one, and only if the core is solid.

---

## 8. Running & verifying

Each stage ships with a way to see it work — never call a stage done without one.

```bash
# Stage 0 — load a frame and eyeball it
python scripts/smoke_test_stage0.py --dataset sunrgbd --index 0

# Full pipeline on one frame → saves annotated RGB + BEV image
python scripts/run_frame.py --dataset sunrgbd --index 0 --out out/frame0/

# Full pipeline over a sequence → annotated RGB | BEV video
python scripts/run_sequence.py --dataset sunrgbd --range 0:200 --out out/seq.mp4

# Unit tests
pytest
```

The golden rule from the course spec and from good practice: **every stage has a concrete feedback loop.**
Geometry stages get a unit test (round-trip a known point); perception stages get a visualization you can
look at. If you can't see or measure a stage working, it isn't done.

---

## 9. Evaluation plan (brief)

Full detail lands with the `eval/` harnesses; the shape:

- **KITTI (outdoor):** 3D **centroid error binned by range** (0–10 / 10–20 / 20–30 m) via the
  [distance-binned fork](https://github.com/xiazhiyi99/kitti_object_eval_python_by_distance) of
  [`kitti-object-eval-python`](https://github.com/traveller59/kitti-object-eval-python), plus **3D AP** at
  relaxed IoU. Context: published [pseudo-LiDAR](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Pseudo-LiDAR_From_Visual_Depth_Estimation_Bridging_the_Gap_in_3D_CVPR_2019_paper.html)
  scores ≈ 12/10/10 AP₃D (E/M/H @ IoU 0.7). A classical RANSAC+clustering head is expected to land below
  that — which is **fine, if** we also report centroid error by distance. Honest, defensible protocol.
- **SUN RGB-D (indoor):** standard **10-class mAP @ 3D IoU 0.25** (the
  [VoteNet](https://github.com/facebookresearch/votenet) protocol; learned-method ceiling ≈ 57). COCO covers
  only some of the 10 classes, so evaluate the scoreable classes and let the rest appear as "unknown"
  obstacles.
- **Own recordings:** absolute distance error vs the tape measure.
- **Cross-cutting headline:** sensor depth vs Depth Anything V2 on identical frames — the "can estimated
  depth replace a sensor" question, in one table + one chart.

---

## 10. Conventions & gotchas

**Code conventions**
- Clean, readable Python — descriptive names, small functions, this is coursework meant to *show
  understanding*, not production infra.
- **RANSAC (Stage 3) and clustering (Stage 4) are self-implemented** — that's a proposal commitment and part
  of the learning goal. Don't reach for a library's one-liner for these two.
- Library code in `src/collision_avoidance/`; runnable entry points in `scripts/`; keep them separate.
- Stage specific files when committing (`git add <file>`), never `git add .` or `-A`.
- Don't commit `data/` or `models/` — they're gigabytes and reproducible from the download scripts.

**Gotchas to respect (sourced in [`datasets-and-simulators.md`](datasets-and-simulators.md))**
- **Depth Anything V2 metric checkpoints are domain-specific** — Indoor/Hypersim (20 m) vs Outdoor/VKITTI
  (80 m). Crossing them silently breaks metric scale.
- **DA-V2's metric ONNX export has a known accuracy bug** ([issue #49](https://github.com/DepthAnything/Depth-Anything-V2/issues/49))
  — run DA-V2 in **PyTorch on GPU** for evaluation, or verify ONNX parity on a few frames first.
- **SUN RGB-D's trap is MATLAB** — use the HuggingFace mirror, spot-check the depth units and intrinsics.
- **KITTI vs nuScenes box centers differ** — KITTI's label location is the box *bottom* center; nuScenes'
  `Box.center` is the geometric center (add h/2 when comparing across datasets). Relevant only if we swap in
  nuScenes.
- **Detector citation note:** Ultralytics YOLOv5/v8/v11 have **no peer-reviewed papers**. Cite the original
  YOLO (Redmon et al., CVPR 2016) + YOLOv10 (NeurIPS 2024) for the report's Related Work — the professor
  requires true venues, not arXiv-only.
- **`pyrealsense2` replay:** set `playback.set_real_time(False)` or frames silently drop.

---

## Next

Open **[`stage0.md`](stage0.md)** and build the data foundation — the `Frame` type, the loader interface,
and the SUN RGB-D loader — then run the smoke test. Once a frame loads with RGB + depth-in-meters +
intrinsics and passes the checklist, you're ready for Stage 1 (detection).
