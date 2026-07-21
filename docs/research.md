Thomas Kulch, Darshan Kedari, Wade Williams  
CS 5330: Computer Vision & Pattern Recognition  
Final Project Proposal  
Summer 2026

Final Project Proposal Research

Everything we considered for the project — datasets, simulators, challenges, and references — with a
verdict for each. Deep dives live in [datasets-and-simulators.md](datasets-and-simulators.md) (logistics +
gotchas), [papers.md](papers.md) (full venue-verified paper list), and [project-ideas.md](project-ideas.md)
(the seven project variants). Facts verified against primary sources 2026-07-21.

# Datasets

| Dataset | Sensor / domain | Size | 3D labels | Role for us | Verdict |
|---|---|---|---|---|---|
| [KITTI](https://www.cvlibs.net/datasets/kitti/) (object) | Stereo RGB + LiDAR, driving | 12 GB | ✓ 3D boxes | Outdoor eval (monocular-depth branch) | ✓ **core** |
| [SUN RGB-D](https://rgbd.cs.princeton.edu/) | Real RGB-D, indoor | 6.4 GB | ✓ 3D boxes | Indoor eval (sensor-depth branch) | ✓ **core** |
| Our own RealSense recordings | RGB-D, indoor | — | tape-measure GT | Metric accuracy + demo footage | ✓ **extension #1** |
| [nuScenes-mini](https://www.nuscenes.org/nuscenes) | 6 cameras + LiDAR, driving | 3.9 GB | ✓ | Stretch: 360° map, center-distance metric, night/rain | ~ stretch |
| [nuScenes full](https://www.nuscenes.org/nuscenes) | Full AV suite | 314 GB | ✓ | Redundant with KITTI | ✗ skip |
| [nuImages](https://www.nuscenes.org/nuimages) | 6 cameras, 2D only | large | ✗ | Detector fine-tuning only | ~ training only |
| [COCO](https://cocodataset.org/#home) | RGB, everyday scenes | — | ✗ | YOLO's pretraining source — not an eval set | (pretraining) |
| [KITTI tracking](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) | Driving video, track IDs | moderate | ✓ | Only for the dynamic-obstacle variant (idea 6) | ~ variant |
| [MOT17](https://motchallenge.net/) | Pedestrian video | moderate | ✗ 2D IDs | Same — tracking variant only | ~ variant |
| [NYU Depth v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) | Kinect, indoor | ~4 GB | ✗ | Optional depth-model sanity check | optional |
| [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) | Kinect + mocap poses | small | ✗ | Only for mapping/SLAM variants | ✗ skip |
| [ScanNet](http://www.scan-net.org/) | RGB-D room scans | huge + signed TOS | ✓ (mesh) | Only for the persistent-map variant (idea 3) | ✗ skip |
| [SemanticKITTI](http://semantic-kitti.org/) | LiDAR only | ~80 GB | ✗ | No RGB-D alignment | ✗ dropped |
| [Amazon AOT](https://registry.opendata.aws/airborne-object-tracking/) | Grayscale air-to-air | **13.4 TB** | ✗ 2D | Airborne detect-and-avoid — wrong domain | ✗ drop |

- **KITTI** — the standard autonomous-driving benchmark suite (CVPR 2012), recorded from a car in
  Karlsruhe with stereo color cameras, a 64-beam LiDAR, and GPS/IMU.
  - Object subset: 7,481 training frames with 3D boxes for Car/Pedestrian/Cyclist; depth ground truth is
    sparse LiDAR projected into the camera — there is no depth camera.
  - Downloadable without registration from [AWS Open Data](https://registry.opendata.aws/kitti/)
    (`s3://avg-kitti`) or the [Kaggle mirror](https://www.kaggle.com/datasets/klemenko/kitti-dataset)
    (per-folder download); CC BY-NC-SA 3.0.
  - The same "KITTI Vision" suite also hosts tracking, stereo, odometry, and depth-prediction benchmarks.
- **SUN RGB-D** — 10,335 indoor RGB-D frames captured with four real depth sensors (Kinect v1/v2, Xtion,
  RealSense) and annotated with 64,595 oriented 3D boxes (CVPR 2015).
  - The standard 3D-detection protocol is 10 furniture classes scored at mAP @ 0.25 3D IoU.
  - Official label tooling is MATLAB; the
    [pre-extracted HF mirror](https://huggingface.co/datasets/youdaoyzbx/processed_sunrgbd) provides the
    same files without it. Direct HTTP download, no registration.
- **Our own recordings** — not a public dataset: `.bag` captures from a RealSense D435-class camera
  (~1 GB/min) — tape-measured static scenes, handheld walkthroughs, and staged approach clips
  ([pyrealsense2](https://pypi.org/project/pyrealsense2/) wheels work on Ubuntu 24.04).
- **nuScenes** — 1,000 twenty-second driving scenes from Boston and Singapore (CVPR 2020): 6 surround
  cameras (1600×900), LiDAR, and radar; 1.4M 3D boxes over 23 classes; includes night and rain.
  - The mini split (10 scenes, 3.9 GB) is a strict subset with a
    [direct download URL](https://www.nuscenes.org/data/v1.0-mini.tgz), no account; full trainval is
    ~314 GB (~60 GB keyframes-only).
  - Its official detection metric matches predictions by center distance (0.5–4 m thresholds), not box
    IoU.
- **nuImages** — 93k driving images with 2D boxes and instance segmentation masks; no depth, no LiDAR, no
  3D annotations.
- **COCO** — 330k images of everyday scenes with 80 object classes in 2D boxes and masks; the dataset
  YOLO detectors are pretrained on.
- **Amazon AOT** — Prime Air's airborne detect-and-avoid dataset (2021 AIcrowd challenge): 4,943 flight
  sequences, 5.9M grayscale air-to-air images, 2D boxes only, 13.4 TB on AWS Open Data.
- **KITTI tracking / MOT17** — driving and pedestrian video benchmarks with ground-truth track
  identities; MOT17 ships public detections with its 7 train + 7 test pedestrian sequences.
- **ScanNet** — 1,500+ RGB-D room scans with per-frame camera poses and 3D instance labels on the
  reconstructed meshes; access requires a signed terms-of-use agreement.
- **TUM RGB-D** — Kinect sequences with motion-capture ground-truth camera trajectories, built for
  SLAM/odometry evaluation; no object labels.
- **NYU Depth v2** — 464 indoor Kinect scenes with 1,449 densely labeled frames (2D semantic
  segmentation); the classic indoor depth-estimation benchmark.
- **SemanticKITTI** — per-point semantic labels over KITTI's LiDAR sequences; LiDAR point clouds only, no
  camera-aligned depth.

# Simulators

| Simulator | Status (mid-2026) | Install on our setup | GPU load | Verdict |
|---|---|---|---|---|
| [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) | Active | conda; ReplicaCAD instant, HM3D free request | Light | ✓ **chosen** (extension #2) |
| [Gazebo + ROS 2 + Nav2](https://docs.nav2.org/) | Active; official Jazzy support | apt | Light | runner-up — pairs with hardware ext |
| [AI2-THOR](https://ai2thor.allenai.org/) | Maintenance mode (2022) | pip | Light | fallback |
| [CARLA](https://carla.org/) | Active | Ubuntu 24.04 unsupported; ~20 GB | Heavy | ✗ skip |
| [Flightmare](https://github.com/uzh-rpg/flightmare) / [Colosseum](https://github.com/CodexLabsLLC/Colosseum) | Unmaintained / niche | — | — | ✗ skip |

- **Habitat-Sim** — chosen for the navigation extension.
  - Photorealistic real-building scans; the all-Python step API returns RGB, metric depth, and GT pose, so
    we inject our own mapper and plan with A* on our own Bird's Eye View (BEV) grid.
  - Built-in `GreedyGeodesicFollower` oracle = free baseline; Success Rate + SPL are citable metrics
    (Habitat, ICCV 2019).
  - ReplicaCAD scenes work day 1 with no approval; submit the free HM3D request immediately as backup.
- **Gazebo + ROS 2 Jazzy + Nav2** — the robotics-stack alternative.
  - Nav2's costmap accepts PointCloud2 from any topic: our perception node becomes a simulated
    TurtleBot 4's only obstacle sense (LiDAR disabled).
  - Chief appeal: the identical node drives the real-camera hardware demo. Chief risk: TF/QoS/costmap
    debugging concentrated on whoever knows ROS.
  - See [darshmenon/rosnav](https://github.com/darshmenon/rosnav) under References — an active Nav2 stack
    that could jump-start this path.
- **AI2-THOR** — pip-install fallback if Habitat's conda/EGL install fights back; synthetic Unity homes;
  in maintenance mode since 2022.
- **CARLA** — great ground-truth-depth data generator, expensive navigation platform; Ubuntu 24.04 is
  unofficial and it is by far the heaviest option here (~20 GB install, GPU-hungry).

# Challenges

| Challenge | Venue / years | Task | Status for us (2026-07-21) |
|---|---|---|---|
| [RoboWorld](https://roboworld2026.github.io/) | NeurIPS 2026 | RGB-D safe/social navigation | **Active**; deadlines ~Oct — post-course entry option |
| [CMU VLN](https://www.ai-meets-autonomy.com/cmu-vln-challenge) | IROS 2026 | Language-grounded navigation | Registration closes Jul 25; submission Aug 25 — post-course only |
| [Habitat Challenge](https://github.com/facebookresearch/habitat-challenge) | CVPR EAI 2019–2023 | PointNav / ObjectNav | Ended; **frozen benchmark reusable** — we adopt SR/SPL |
| [BARN](https://people.cs.gmu.edu/~xiao/Research/BARN_Challenge/BARN_Challenge26.html) | ICRA 2022–2026 | Cluttered ground-robot navigation | 2026 concluded; scoring formula reusable |
| [DodgeDrone](https://uzh-rpg.github.io/icra2022-dodgedrone/) | ICRA 2022 | Vision-based drone obstacle avoidance | One edition; starter kit public |
| [AI Driving Olympics](https://duckietown.com/research/ai-driving-olympics/) | NeurIPS 2018–2021 | Duckietown lane + obstacles | Dormant — the closest NeurIPS collision-avoidance challenge |
| [Waymo](https://waymo.com/open/challenges/) / [Argoverse](https://www.argoverse.org/tasks.html) / [nuScenes](https://www.nuscenes.org/nuscenes) / [CARLA](https://leaderboard.carla.org/) | CVPR WAD yearly | Driving perception suites | No 2026 cycles; leaderboards open year-round |
| [Earth Rover](https://earth-rover-challenge.github.io/) | IROS 2024–2026 | Real sidewalk-robot navigation | Active + free — event runs after the course |
| [3LC Multi-Vehicle Detection](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge) | Kaggle community, 2026 | 2D vehicle detection (frozen YOLOv8n, label cleaning) | **Finished June 2026**; 2D-only — not useful for our eval |
| iGibson / RoboTHOR / MultiON | CVPR EAI 2020–2023 | Interactive / object navigation | Dormant; benchmarks public |

- **RoboWorld (NeurIPS 2026)** — the one live challenge aligned with our topic (Tracks 2/3/5: RGB-D
  perception + collision-free navigation among moving humans); our idea-6 variant is deliberately shaped
  as a post-course on-ramp.
- **CMU VLN** — the sensor suite (3D LiDAR + 360° camera) and VLN task sit a step away from ours; only
  the free registration (deadline Jul 25) matters, and only if we want the post-course option.
- **3LC (Kaggle)** — verified: a data-cleaning competition on UA-DETRAC traffic-camera footage with a
  frozen YOLOv8n; finished, monocular 2D only. Considered and set aside.

# References — Repos, Tools & Papers

| Repo / tool | What we'd use it for |
|---|---|
| [ultralytics](https://github.com/ultralytics/ultralytics) | YOLO11 detection, ONNX export, optional fine-tuning; also serves [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) for idea 7 |
| [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | Metric monocular depth ([outdoor](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small) / [indoor](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small) checkpoints); run PyTorch — the metric ONNX export has a [known bug](https://github.com/DepthAnything/Depth-Anything-V2/issues/49) |
| [kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python) + [distance fork](https://github.com/xiazhiyi99/kitti_object_eval_python_by_distance) | KITTI 3D AP + distance-binned centroid error |
| [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) | nuScenes loading, transforms, official center-distance eval |
| [VoteNet](https://github.com/facebookresearch/votenet) | SUN RGB-D 10-class eval protocol + code; pretrained learned baseline |
| [habitat-sim](https://github.com/facebookresearch/habitat-sim) / [habitat-lab](https://github.com/facebookresearch/habitat-lab) | Navigation-extension platform + oracle baseline |
| [pyrealsense2](https://pypi.org/project/pyrealsense2/) | RealSense `.bag` recording + aligned-depth replay |
| [py-motmetrics](https://github.com/cheind/py-motmetrics) | MOTA/IDF1 scoring for the tracking variant |
| [processed_sunrgbd (HF)](https://huggingface.co/datasets/youdaoyzbx/processed_sunrgbd) | MATLAB-free SUN RGB-D labels |
| [darshmenon/rosnav](https://github.com/darshmenon/rosnav) | Active ROS 2 Nav2 navigation stack (Humble/Jazzy, last push Jul 2026) — jump-start for the Gazebo/hardware path |
| [Real-Time-Object-Detection-for-Collision-Avoidance](https://github.com/sakibsadmanshajib/Real-Time-Object-Detection-for-Collision-Avoidance) | One-off 2024 student project (YOLOv11 on KITTI 2D, inactive); only its KITTI→YOLO label conversion is a useful reference |

- **Core papers to cite in the report** (full venue-verified list with links in [papers.md](papers.md);
  the professor requires true peer-reviewed venues — note LMDepth and ZoeDepth are arXiv-only):
  - Pipeline lineage: Frustum PointNets (CVPR 2018) — 2D boxes + depth frustums → 3D, exactly our shape;
    Pseudo-LiDAR (CVPR 2019) — depth map → point cloud is what makes camera 3D detection work; RANSAC
    (Comm. ACM 1981); occupancy grids (Elfes, IEEE Computer 1989); Stixel World (DAGM 2009); ground
    segmentation + clustering archetype (Zermas et al., ICRA 2017).
  - Depth: Depth Anything V2 (NeurIPS 2024) — our depth model; Metric3D (ICCV 2023) — why zero-shot
    *metric* depth is possible; MiDaS (TPAMI 2022); Depth Pro (ICLR 2025) — alternative backbone;
    RealSense sensor accuracy study (IEEE Access 2024) — published error-vs-distance curves to compare
    against our tape measurements.
  - 3D detection + datasets: MonoDLE (CVPR 2021) — localization error dominates, motivates our metric;
    image-based 3D detection survey (TPAMI 2024); ImVoteNet (CVPR 2020); KITTI (CVPR 2012); SUN RGB-D
    (CVPR 2015); nuScenes (CVPR 2020) — source of the center-distance protocol.
  - Navigation + variants: Habitat (ICCV 2019) — SR/SPL protocol; VLFM (ICRA 2024) — zero-shot semantic
    navigation, the template for idea 7; SORT (ICIP 2016) + DeepSORT (ICIP 2017) for idea 6; YOLO-World
    (CVPR 2024); BARN benchmark (IEEE SSRR 2020).
  - Camera-to-BEV lineage: Lift-Splat-Shoot (ECCV 2020), BEVFormer (ECCV 2022), SparseOcc (ECCV 2024) —
    the learned counterparts our geometric BEV map is positioned against; plus two close system peers:
    indoor obstacle discovery on reflective ground (IJCV 2024) and real-time monocular occupancy-grid
    mapping (J. Field Robotics 2026).
  - Components: YOLO (CVPR 2016) + YOLOv10 (NeurIPS 2024) — the citable detector pair (Ultralytics
    v5/v8/v11 have no peer-reviewed papers; cite the repo as software); A* (IEEE TSSC 1968); Kalman
    (Trans. ASME 1960); Hungarian method (NRLQ 1955).
