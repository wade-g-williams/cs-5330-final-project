# Datasets & Simulators — Comparison

Research compiled 2026-07-21 for the CS 5330 final project (RGB-D Object Detection for Collision Avoidance).
All venue/status/size claims below were verified against primary sources (dataset sites, GitHub repos, challenge pages) on this date.

**Decisions made (2026-07-21):** core evaluation on KITTI + SUN RGB-D; own RealSense recordings = extension #1; Habitat-Sim navigation = extension #2; live hardware demo = extension #3. SemanticKITTI dropped.

## Datasets

| Dataset | Sensor / domain | Download | 3D box labels | Role for this project | Verdict |
|---|---|---|---|---|---|
| **KITTI object** (CVPR 2012) | Stereo RGB + LiDAR-projected depth, outdoor driving | **12 GB** images + calib/labels | ✓ 7,481 frames, Car/Ped/Cyclist | Outdoor eval of the *monocular-depth* branch (Depth Anything V2 → pseudo-LiDAR) | ✓ **core** |
| **SUN RGB-D** (CVPR 2015) | Real RGB-D (Kinect/Xtion/RealSense), indoor | **6.4 GB** | ✓ 10,335 frames, 64,595 boxes | Indoor eval with *real sensor depth*; standard 10-class mAP@0.25 protocol | ✓ **core** |
| **Own RealSense recordings** | RealSense D435-class, indoor | — | Tape-measure GT | Only place we control ground truth exactly; demo footage | ✓ **extension #1** |
| NYU Depth v2 (ECCV 2012) | Kinect, indoor | ~4 GB (Kaggle mirror) | ✗ (2D seg only) | Sanity-check DA-V2 indoor metric depth | optional |
| KITTI depth prediction | Outdoor | moderate | ✗ (dense depth GT) | Quantify DA-V2 depth error before it enters the pipeline | optional |
| TUM RGB-D (IROS 2012) | Kinect, indoor | small | ✗ (camera trajectories) | Only useful for SLAM/ego-motion variants | ✗ skip |
| SemanticKITTI (ICCV 2019) | **LiDAR only** | ~80 GB | ✗ (point labels) | No RGB-D alignment; one metric via extra plumbing | ✗ **dropped** |
| ScanNet (CVPR 2017) | RGB-D, indoor | huge + signed TOS | ✓ | Heavier substitute for SUN RGB-D | ✗ skip |
| BDD100K / Cityscapes | RGB driving | large | ✗ (2D/seg) | Only for freespace/robustness project variants | ✗ (variant only) |
| nuScenes-mini (CVPR 2020) | 6 surround cameras (1600×900) + LiDAR/radar; incl. night + rain | **3.9 GB**, direct URL, no login | ✓ (10 classes, 2 Hz keyframes) | Stretch: 360° Bird's Eye View (BEV) map; official center-distance mAP fits our centroid pipeline; night/rain figures | ~ stretch |
| nuScenes full / Waymo / Lyft | Full AV suites, LiDAR-centric | 314 GB (nuScenes; keyframes-only ~60 GB) / larger | ✓ | Redundant with KITTI for the monocular branch | ✗ skip |
| nuImages | 6 cameras, **2D boxes/masks only** | large | ✗ no 3D, no depth | Detector training/robustness only — we don't train | ✗ skip |

### Key practical findings

- **KITTI without registration**: the official cvlibs site wants an approved institutional account, but the identical files are on AWS Open Data — `aws s3 cp --no-sign-request s3://avg-kitti/data_object_image_2.zip .` (https://registry.opendata.aws/kitti/) — and the Kaggle mirror (`klemenko/kitti-dataset`) allows per-folder download, so grab images + calib + labels (~12 GB) and **skip the 29 GB of LiDAR**. License CC BY-NC-SA 3.0.
- **Ready-made KITTI eval**: `traveller59/kitti-object-eval-python` (de-facto standard, full val split in seconds; AP11/AP40, 2D/BEV/3D). A distance-binned fork exists (`xiazhiyi99/kitti_object_eval_python_by_distance`) that matches our "3D centroid error vs range" metric. Standard split: 3,712 train / 3,769 val ("Chen split"); AP|R40 is standard since 2019.
- **SUN RGB-D's known trap is MATLAB**: official label extraction runs MATLAB scripts (the #1 pain point in VoteNet issues). Mitigation: community **pre-extracted mirrors on Hugging Face** (`youdaoyzbx/processed_sunrgbd`, mmdetection3d layout; `yxchng/processed_sunrgbd` as backup) — spot-check frames against raw data and skip MATLAB entirely. Direct HTTP download from Princeton needs no registration (https://rgbd.cs.princeton.edu/).
- **SUN RGB-D protocol**: VoteNet 10-class (bed, table, sofa, chair, toilet, desk, dresser, night_stand, bookshelf, bathtub), mAP @ 3D IoU 0.25; VoteNet reference ≈ 57 mAP@0.25. The 37-class protocol is 2D segmentation — not relevant here.
- **Comparison numbers exist**: a Jan-2026 systematic study of pseudo-LiDAR depth backbones reports Car AP₃D ≈ 11.9/10.5/10.0 (E/M/H @ IoU 0.7) on val (https://arxiv.org/html/2601.03617). A classical RANSAC+clustering head will land below that — fine, *if* we also report centroid error by distance bin. Honest, defensible protocol.
- **Depth Anything V2 metric checkpoints are domain-specific**: Indoor/Hypersim (max_depth 20 m) for SUN RGB-D, Outdoor/VKITTI (max_depth 80 m) for KITTI — crossing them breaks scale. Caution: a known issue reports **accuracy degradation in ONNX exports of the metric variants** (DepthAnything/Depth-Anything-V2#49) — run PyTorch (GPU) for eval, or verify ONNX parity on a few frames first.
- **Detector**: Ultralytics ONNX CPU speeds at 640 px — YOLO11n ≈ 18 fps, YOLO11s ≈ 11 fps. COCO classes cover both domains (car/truck/person/bicycle outdoors; chair/couch/bed/toilet/table indoors) — map COCO→dataset classes explicitly and state which are evaluable.
- **Using nuScenes (verified 2026-07-21)**: `v1.0-mini.tgz` (3.88 GiB, 10 scenes, all sensors + annotations) downloads from a direct public URL (`https://www.nuscenes.org/data/v1.0-mini.tgz`) with no account; full trainval is 10 blobs ≈ 314 GB with **no camera-only option**, but the download page offers "keyframe blobs only" ≈ 60 GB total. Devkit: `uv pip install nuscenes-devkit` (≥ Py3.9). `nusc.get_sample_data(cam_token)` returns image path + GT boxes *already in the camera frame* + intrinsics — drop-in for our pipeline; `nusc.explorer.map_pointcloud_to_image()` gives LiDAR-projected GT depth. Gotcha: nuScenes `Box.center` is the geometric 3D center, KITTI's location is the *bottom* center — add h/2 when comparing across datasets. Official eval (`detection_cvpr_2019` config) runs on `mini_val`, but needs results in the *global* frame (chain camera→ego via `calibrated_sensor`, ego→global via `ego_pose`). COCO→nuScenes class map covers car, truck, bus, pedestrian, motorcycle, bicycle (6 of 10 classes). License CC BY-NC-SA — coursework fine.
- **pyrealsense2 now ships Ubuntu-24.04-compatible wheels** (v2.58.3, 2026-07-19) — `uv pip install pyrealsense2` works. Record with `rs.config.enable_record_to_file()`; replay with `playback.set_real_time(False)` (without it, playback silently drops frames). Bags grow ~1 GB/min at 640×480@30 — keep clips short.

## Simulators

| Simulator | Status (mid-2026) | Install on our setup | GPU load | Path to "navigate using OUR obstacle map" | Effort | Verdict |
|---|---|---|---|---|---|---|
| **Habitat-Sim** | Active | conda one-liner; **ReplicaCAD scenes instant**, HM3D needs a free access request | Easy | Python loop: `sim.step()` → RGB-D → our pipeline → BEV map → A* → discrete actions. Built-in `GreedyGeodesicFollower` = free oracle baseline; report SR/SPL | ~5–8 person-days | ✓ **chosen** |
| **Gazebo Harmonic + ROS 2 Jazzy + TurtleBot 4 + Nav2** | Official Jazzy support (Clearpath 2024) | `apt install`, zero env risk (Wade's native stack) | Fine | Publish obstacle clusters as PointCloud2 → Nav2 costmap with LiDAR disabled → Nav2 navigates on our perception alone; RTAB-Map binaries for RGB-D SLAM comparison | ~6–10 person-days, ROS-member-heavy | runner-up — synergizes with hardware ext (same node runs on the real camera) |
| AI2-THOR | Maintenance mode (5.0.0, Dec 2022) | `pip install ai2thor`; ProcTHOR-10K ungated | Fine | Same loop as Habitat via step API + GT metadata | ~4–6 person-days | fallback if Habitat install fights back |
| CARLA | Active (0.9.16 / 0.10.0-UE5) | Ubuntu 24.04 unsupported; ~20 GB | Heavy | Great GT-depth data generator, expensive nav platform | High | ✗ skip |
| Flightmare / Colosseum (AirSim) | Unmaintained / archived | — | — | — | — | ✗ skip |

**Habitat plan sketch** (if extension #2 is reached): day 1 conda install + ReplicaCAD + submit HM3D request; days 2–3 control loop feeding our pipeline and accumulating a world-frame BEV grid from GT pose; days 4–5 A*/greedy planner → discrete actions toward PointNav goals; days 6–8 SR/SPL over N episodes vs the geodesic oracle + videos. Free ablation: sim's perfect depth vs DA-V2 on sim RGB.

## Challenges (NeurIPS / ICRA / CVPR) — verified status

| Challenge | Venue / years | Status for us |
|---|---|---|
| Habitat Challenge (PointNav/ObjectNav) | CVPR EAI Workshop 2019–2023 | Ended; **frozen benchmark fully reusable** — adopt SR/SPL |
| BARN (cluttered navigation) | ICRA 2022–2026 | 2026 edition concluded June; scoring formula citable/reusable |
| DodgeDrone (vision obstacle avoidance) | ICRA 2022 | One edition; starter kit public |
| AI Driving Olympics (Duckietown) | NeurIPS 2018–2021 | Dormant — closest thing to a NeurIPS collision-avoidance challenge |
| Waymo / Argoverse / nuScenes / CARLA | CVPR WAD yearly | No 2026 cycles; leaderboards stay open year-round |
| **RoboWorld** (RGB-D social navigation) | **NeurIPS 2026** | **Active now**, deadlines ~Oct — after the course, but relevant beyond it |
| CMU VLN Challenge | IROS 2026 | Registration closes Jul 25, submission Aug 25 — after our due date; not practical |

Takeaway: nothing aligns with Aug 13; the value is citing and reusing *protocols* — KITTI AP for detection, SPL/Success Rate (Habitat, Savva et al. ICCV 2019) for navigation, BARN's success×time score for obstacle-course demos.

## Kaggle — verified

No active competition on driving perception, depth, or obstacle detection as of July 2026. Historical ones (Lyft 3D Detection 2019 ~85 GB LiDAR-centric; Peking/Baidu 6-DoF pose 2020; CVPR 2018 WAD segmentation) are poor fits. Current serious challenges run on non-Kaggle platforms (Waymo eval server, EvalAI, CodaBench). **Use Kaggle purely as a mirror**: `klemenko/kitti-dataset` (selective download) and NYU Depth v2 (~4 GB, `soumikrakshit` mirror).
