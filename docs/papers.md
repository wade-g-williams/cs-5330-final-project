# Papers — RGB-D Object Detection for Collision Avoidance

Venue-verified reading list, compiled 2026-07-21. Every paper marked ✓ had its peer-reviewed venue confirmed against
the publisher's page (CVF open access, IEEE Xplore, ACM DL, Springer, NeurIPS/ICLR proceedings) — the professor requires
citing the true peer-reviewed venue, not arXiv. Papers marked **arXiv only** must NOT be cited as peer-reviewed in the report.

**Shortlist for the report's related-work section (pick ≥3):** Frustum PointNets (CVPR 2018), Pseudo-LiDAR (CVPR 2019),
Depth Anything V2 (NeurIPS 2024), plus the dataset papers (KITTI CVPR 2012, SUN RGB-D CVPR 2015) and RANSAC (CACM 1981) in methods.
**Detector citation note:** Ultralytics YOLOv5/v8/v11 have NO peer-reviewed paper — cite YOLO (CVPR 2016) + YOLOv10
(NeurIPS 2024) for the method and the Ultralytics GitHub repo as software.

## 1. Pipeline / geometry (ground removal, clustering, obstacle maps)

| ✓ | Paper | Venue | Why it matters to us |
|---|---|---|---|
| ✓ | Fischler & Bolles, *Random Sample Consensus* | Comm. ACM 24(6), 1981 | The original RANSAC — exactly our ground-plane removal step |
| ✓ | Labayrade et al., *Real Time Obstacle Detection in Stereovision... "V-disparity"* | IEEE IV 2002 | Seminal driving-domain alternative to RANSAC ground removal |
| ✓ | Elfes, *Using Occupancy Grids for Mobile Robot Perception and Navigation* | IEEE Computer 22(6), 1989 | Grounds our top-down Bird's Eye View (BEV) obstacle map as an occupancy grid |
| ✓ | Badino et al., *The Stixel World* | DAGM 2009 (LNCS 5748) | Classic compact obstacle representation from dense depth |
| ✓ | Zermas et al., *Fast Segmentation of 3D Point Clouds* | ICRA 2017 | The archetype of our pipeline: ground-plane fit → cluster the rest |
| ✓ | Bernini et al., *Real-Time Obstacle Detection Using Stereo Vision... A Survey* | IEEE ITSC 2014 | Survey organizing exactly our problem family — related-work taxonomy |
| ✓ | Wang et al., *Pseudo-LiDAR from Visual Depth Estimation* | CVPR 2019 | Depth map → 3D point cloud is what makes camera-based 3D detection work |
|   | Ester et al., *A Density-Based Algorithm...* (DBSCAN) | KDD 1996 | The clustering algorithm option for obstacle extraction |
|   | Pfeiffer & Franke, *Multi-Layer Stixel* | BMVC 2011 | Principled successor to stixels |
|   | Oniga & Nedevschi, *Elevation Maps* | IEEE TVT 2010 | Grid-based version of our BEV map from dense stereo |
|   | Rusu & Cousins, *3D is here: PCL* | ICRA 2011 | Standard citation for RANSAC-plane + Euclidean-cluster tooling |
|   | Lee et al., *Patchwork++* | IROS 2022 | Modern ground segmentation; documents where single-plane RANSAC fails (slopes, curbs) — good for discussion section |
|   | *Efficient Obstacle Detection and Tracking Using RGB-D...* | Sensors (MDPI) 2022 | Close published counterpart to our RealSense indoor pipeline |
| ✓ | *Indoor Obstacle Discovery on Reflective Ground via Monocular Camera* | IJCV, 2024 | Monocular obstacle discovery via ground-plane estimation — direct peer of our RANSAC stage; documents the reflective-floor failure mode for our limitations section |
| ✓ | Ren et al., *Real-Time Monocular 2D Occupancy Grid Mapping for Autonomous Navigation of Ground Robots* | J. Field Robotics, 2026 | Closest recent system paper to ours: camera → real-time 2D occupancy grid for ground-robot navigation |
|   | *OoDIS: Anomaly Instance Segmentation and Detection Benchmark* | ICRA 2025 | Unknown-road-obstacle benchmark — motivates our "unknown obstacle" clusters vs a closed-set detector |
|   | *Road Obstacle Video Segmentation* | GCPR 2025 | Temporal-consistency argument that supports our tracking/forecasting extension |
|   | *Vision Transformers for End-to-End Vision-Based Quadrotor Obstacle Avoidance* | ICRA 2025 | The end-to-end learned alternative our modular detect-map-plan pipeline is contrasted against |

## 2. Depth estimation

| ✓ | Paper | Venue | Why it matters to us |
|---|---|---|---|
| ✓ | Yang et al., *Depth Anything* | CVPR 2024 | Foundation monocular depth; lineage for V2 |
| ✓ | Yang et al., *Depth Anything V2* | **NeurIPS 2024** | Our monocular metric depth source (indoor + outdoor metric checkpoints) |
| ✓ | Yin et al., *Metric3D* | ICCV 2023 | Explains *why* zero-shot metric depth is possible (canonical camera space) — justifies our depth-to-meters step |
| ✓ | Piccinelli et al., *UniDepth* | CVPR 2024 | Alternative design point for metric depth (no test-time intrinsics) |
| ✓ | Ranftl et al., *MiDaS* | IEEE TPAMI 44(3), 2022 | Seminal zero-shot relative depth; explains why relative ≠ metric |
| ✓ | Bochkovskii et al., *Depth Pro* | ICLR 2025 | Strong alternative metric-depth backbone for our comparison |
| ✓ | Simon & Majumdar, *MonoNav* | ISER 2023 (Springer PAR 2024) | Closest system precedent: pretrained metric depth → 3D map → collision-free MAV navigation |
| ✓ | *Comparative Evaluation of Intel RealSense D415/D435i/D455, Azure Kinect* | IEEE Access 12, 2024 | Published error-vs-distance numbers for our exact sensors — compare to our tape-measure results |
| ✓ | *Video Depth Anything* | CVPR 2025 | Temporally consistent depth — cite for the video setting |
|   | Metric3D v2 | IEEE TPAMI 2024 | Journal version, metric depth + normals |
|   | ZoeDepth | **arXiv only** | Widely used but NOT peer-reviewed — do not cite as such |
|   | LMDepth | **arXiv only** | ⚠ In our original brainstorm doc — swap it for Depth Pro (ICLR 2025) or MiDaS in the report |
|   | Hirschmüller, *Semi-Global Matching* (SGM) | IEEE TPAMI 30(2), 2008 | Classical stereo-depth alternative to monocular models; underlies RealSense's own correlation engine |

## 3. 3D object detection & datasets

| ✓ | Paper | Venue | Why it matters to us |
|---|---|---|---|
| ✓ | Qi et al., *Frustum PointNets* | CVPR 2018 | Exactly our shape: 2D boxes + depth points → metric 3D localization; evaluated on KITTI **and** SUN RGB-D |
| ✓ | Ma et al., *Delving into Localization Errors for Monocular 3D Detection* (MonoDLE) | CVPR 2021 | Shows depth/localization error dominates — motivates our distance-binned metric |
| ✓ | Ma et al., *3D Object Detection from Images: A Survey* | IEEE TPAMI 46(5), 2024 | Ideal framing citation; explains KITTI 3D/BEV AP protocols |
| ✓ | Qi et al., *ImVoteNet* | CVPR 2020 | Indoor RGB-D analogue of our 2D-detection + depth design (SUN RGB-D) |
| ✓ | Geiger et al., *KITTI Vision Benchmark Suite* | CVPR 2012 | Our outdoor dataset (depth GT = projected LiDAR, not RGB-D) |
| ✓ | Song et al., *SUN RGB-D* | CVPR 2015 | Our indoor dataset — real RGB-D sensors + 3D boxes |
|   | Qi et al., *VoteNet* | ICCV 2019 | Source of the SUN RGB-D 10-class mAP@0.25 protocol + eval code |
|   | SMOKE | CVPR Workshops 2020 | Monocular 3D detection baseline family |
|   | Silberman et al., *NYU Depth v2* | ECCV 2012 | Optional depth-validation set |
|   | Sturm et al., *TUM RGB-D* | IROS 2012 | Only if ego-motion/SLAM enters the picture |
|   | Behley et al., *SemanticKITTI* | ICCV 2019 | Cite only to justify *not* using it (LiDAR-only) |

## 4. Navigation protocols (for the Habitat extension)

| ✓ | Paper | Venue | Why it matters to us |
|---|---|---|---|
| ✓ | Savva et al., *Habitat: A Platform for Embodied AI Research* | ICCV 2019 | The simulator + PointNav protocol we adopt |
| ✓ | Yokoyama et al., *VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation* | ICRA 2024 | Occupancy map + vision-language value map for zero-shot ObjectNav in Habitat — the direct template for our language-directed extension (idea 7) |
|   | Ramakrishnan et al., *PONI* | CVPR 2022 | Modular semantic-map-based ObjectNav — supports our map-then-plan design over end-to-end policies |
|   | Anderson et al., *On Evaluation of Embodied Navigation Agents* | **arXiv only** (1807.06757) | Defines SPL — standard to cite, but flag it's an arXiv report |
|   | Perille et al., *Benchmarking Metric Ground Navigation* | IEEE SSRR 2020 | BARN benchmark paper — the success×time scoring idea |

## 5. Camera-to-BEV lineage (learned counterparts of our geometric map)

| ✓ | Paper | Venue | Why it matters to us |
|---|---|---|---|
| ✓ | Philion & Fidler, *Lift, Splat, Shoot* | ECCV 2020 | Foundational learned camera→BEV: lifts image features via predicted depth onto a BEV grid — the learned analogue of our explicit unprojection |
| ✓ | Li et al., *BEVFormer* | ECCV 2022 | Canonical transformer camera-only BEV perception — the modern learned baseline our geometric pipeline is positioned against |
| ✓ | Liu et al., *Fully Sparse 3D Occupancy Prediction* (SparseOcc) | ECCV 2024 | Recent camera-only occupancy prediction — completes the lineage from LSS/BEVFormer to occupancy grids like ours |
|   | Wang et al., *PanoOcc* | CVPR 2024 | Camera-only occupancy with per-voxel semantics — learned counterpart of our class-labeled BEV cells |

## 6. Detection, tracking & planning components

| ✓ | Paper | Venue | Why it matters to us |
|---|---|---|---|
| ✓ | Redmon et al., *You Only Look Once* | CVPR 2016 | Foundational YOLO citation — the correct method reference since Ultralytics v5/v8/v11 have no peer-reviewed papers |
| ✓ | Wang et al., *YOLOv10: Real-Time End-to-End Object Detection* | NeurIPS 2024 | The most recent peer-reviewed mainline YOLO — the up-to-date detector citation |
| ✓ | Wojke et al., *DeepSORT* | IEEE ICIP 2017 | Appearance-augmented SORT — what our idea-6 training add-on implements (or contrasts against) |
| ✓ | Hart, Nilsson & Raphael, *A Formal Basis for the Heuristic Determination of Minimum Cost Paths* (A*) | IEEE Trans. Systems Science and Cybernetics 4(2), 1968 | The planner we run on our own BEV occupancy grid |
| ✓ | Kalman, *A New Approach to Linear Filtering and Prediction Problems* | Trans. ASME J. Basic Engineering 82, 1960 | The state estimator inside our tracker and constant-velocity forecaster |
| ✓ | Kuhn, *The Hungarian Method for the Assignment Problem* | Naval Research Logistics Quarterly 2, 1955 | The optimal assignment solver for detection-to-track association |
|   | Redmon & Farhadi, *YOLO9000* | CVPR 2017 | Anchor-box + multi-scale lineage between YOLOv1 and v7 |
|   | Wang et al., *YOLOv7* | CVPR 2023 | Alternative modern peer-reviewed YOLO citation |
|   | Dendorfer et al., *MOTChallenge* | IJCV 129(4), 2021 | Defines the MOT metrics (MOTA, IDF1, ID switches) we report |
|   | Schöller et al., *What the Constant Velocity Model Can Teach Us...* | IEEE RA-L 5(2), 2020 | Peer-reviewed justification that constant-velocity forecasting is a strong short-horizon baseline |
|   | Borenstein & Koren, *The Vector Field Histogram* | IEEE Trans. Robotics and Automation 7(3), 1991 | Classical reactive steering on exactly our kind of occupancy grid |
|   | Fox, Burgard & Thrun, *The Dynamic Window Approach to Collision Avoidance* | IEEE Robotics & Automation Magazine 4(1), 1997 | Velocity-space local planner — Nav2's ancestor, contrast to our global A* |
