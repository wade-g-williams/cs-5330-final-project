Thomas Kulch, Darshan Kedari, Wade Williams

CS 5330: Computer Vision & Pattern Recognition

Final Project Brainstorm

Summer 2026

CS 5330: Final Project Brainstorm

# **Project Ideas**

* **Constructing Interactive Simulation Environments from Images** → Turn a handful of ordinary photos into a navigable 3D scene using an RGB camera, matching features across the images, recovering camera poses and a sparse point cloud with [structure-from-motion](https://en.wikipedia.org/wiki/Structure_from_motion), and densifying it into a textured mesh with multi-view stereo to produce an interactive 3D environment you can fly through or drop virtual objects into, evaluated on a public multi-view stereo benchmark plus our own photo sets of a desk or room we capture from many angles. (Recent related work: [DUSt3R explainer](https://learnopencv.com/dust3r-geometric-3d-vision/), [MASt3R](https://github.com/naver/mast3r), [MASt3R-SLAM, CVPR 2025](https://edexheim.github.io/mast3r-slam/), [3D Gaussian Splatting for real scenes, 2025](https://www.mdpi.com/1424-8220/25/22/6999).)

* **Handling Object Occlusion** → Detect and recover partially hidden objects in cluttered scenes using an RGB camera, segmenting each visible object, reasoning about which objects block which, predicting the extent hidden behind the occluder, and re-identifying objects as they reappear to produce amodal masks (the complete object shape, visible plus hidden) with an occlusion ordering, evaluated on the public [KINS (KITTI amodal) dataset](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset) with amodal mask IoU plus our own recorded clips where objects pass behind one another. (Recent related work: [Sequential Amodal Segmentation, 2024](https://arxiv.org/abs/2405.05791), [amodal occlusion recovery, 2026](https://link.springer.com/article/10.1007/s00138-026-01853-6).)

* **Motion Tracking for Sports** → Track an athlete's body motion and the ball in sports video using an RGB camera, detecting and following each player with a detector plus a [Kalman-filter](https://en.wikipedia.org/wiki/Kalman_filter) tracker, estimating body keypoints with a pretrained pose model, and following the fast-moving ball to produce per-frame skeletons and ball trajectories plus motion analytics like joint angles, speed, and shot events, evaluated on public sports datasets plus our own recorded clips of a sport we play. (Recent related work: [TrackID3x3 basketball, 2025](https://arxiv.org/abs/2503.18282), [sports pose-tracking survey, 2025](https://link.springer.com/article/10.1007/s10462-025-11344-1), [PadelTracker100, 2026](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12926558/).)

* **RGB-D Object Detection for Collision Avoidance** → Map obstacles for collision avoidance using an [RGB-D camera](https://www.intelrealsense.com/depth-camera-d435i/), detecting and classifying them in each RGB frame, placing every obstacle in metric 3D from the aligned depth, removing the ground plane with [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus), and clustering the rest to produce a top-down obstacle map that labels each object's class and 3D position, evaluated on the public [KITTI](https://www.cvlibs.net/datasets/kitti/) and [SemanticKITTI](http://semantic-kitti.org/) datasets plus our own RGB-D recordings of indoor obstacles at tape-measured distances. (Recent related work: [LMDepth, 2025](https://arxiv.org/abs/2505.00980), [Depth Anything V2, 2024](https://github.com/DepthAnything/Depth-Anything-V2), [Video Depth Anything, CVPR 2025](https://github.com/DepthAnything/Video-Depth-Anything).)

* **Motion Tracking via Object Detection (SORT from scratch)** → Track objects across frames using any RGB camera, running a pretrained ONNX detector each frame, predicting motion with a [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter), and matching detections to tracks by IoU with the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) to produce labeled object tracks that hold a stable ID for each object as it moves, is occluded, and reappears, evaluated on the public [MOT16/17 benchmark](https://motchallenge.net/) plus our own webcam clips of people and vehicles that we annotate for identity. (Recent related work: [MOTIP, CVPR 2025](https://github.com/MCG-NJU/MOTIP), [TrackTrack, CVPR 2025](https://github.com/kamkyu94/TrackTrack).)

* **6-DoF Object Pose Estimation** → Estimate the full 6-DoF pose of known objects using an RGB camera, or an RGB-D camera for the depth-assisted variant, establishing 2D-to-3D correspondences from the object's features, solving with [PnP](https://docs.opencv.org/3.4/d5/d1f/calib3d_solvePnP.html) and RANSAC, and refining across frames to produce each object's 3D position and orientation (a rotation and translation) that a robot could use to grasp it, evaluated on the public [LINEMOD benchmark](https://bop.felk.cvut.cz/datasets/) with the ADD metric plus our own objects, ground-truthed with a hidden AprilTag. (Recent related work: [Any6D, CVPR 2025](https://github.com/taeyeopl/Any6D), [iG-6DoF, CVPR 2025](https://cvpr.thecvf.com/virtual/2025/poster/32800).)

* **Open-Vocabulary Object Finder (CLIP)** → Localize any object named by a typed text query using an ordinary RGB camera, proposing candidate regions with [Edge Boxes](https://www.microsoft.com/en-us/research/publication/edge-boxes-locating-object-proposals-from-edges/) or a sliding window, embedding each with [CLIP](https://github.com/openai/CLIP) on the CPU through ONNX Runtime, scoring them against the query, and keeping the best with non-max suppression to produce a labeled bounding box around the queried object, evaluated on public image frames plus our own webcam scenes that we hand-label for the queried objects. (Recent related work: [YOLOE, 2025](https://learnopencv.com/yoloe-tutorial-real-time-open-vocabulary-detection/), [YOLO-World, CVPR 2024](https://github.com/AILab-CVC/YOLO-World), [open-vocab on edge devices, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12583037/).)

* **Grasp-Point Detection** → Detect ranked, gripper-feasible [grasp points](https://arxiv.org/abs/1804.05172) using an RGB camera, or an RGB-D camera for depth-based segmentation, segmenting the object from the background, searching for opposing edge pairs within the gripper's width, and scoring candidates by grip stability and flatness to produce ranked grasp rectangles (a position, angle, and opening width for the gripper), evaluated on the public Cornell Grasping Dataset with the rectangle metric plus our own top-down photos of household objects that we annotate with good grasps. (Recent related work: [HFNet, 2025](https://www.sciencedirect.com/science/article/abs/pii/S0263224125011340), [Language-Guided Grasping, 2026](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202501276).)

## **Cool Papers**

* [SAM 2: Segment Anything in Images and Videos (2024)](https://arxiv.org/abs/2408.00714) — click/box/text to segment and track any object through a video  
* [VGGT: Visual Geometry Grounded Transformer (CVPR 2025, Best Paper)](https://arxiv.org/abs/2503.11651) — one network infers camera pose, depth, point maps, and 3D tracks from a few images in seconds  
* [Depth Anything V2 (NeurIPS 2024\)](https://arxiv.org/abs/2406.09414) — the go-to monocular depth model, metric-depth variants included  
* [LMDepth: Lightweight Mamba Monocular Depth (2025)](https://arxiv.org/abs/2505.00980) — edge/CPU-friendly depth estimation  
* [FoundationPose: 6D Pose & Tracking of Novel Objects (CVPR 2024\)](https://arxiv.org/abs/2312.08344) — pose of any object from a CAD model or a few reference images  
* [Any6D: Model-free 6D Pose of Novel Objects (CVPR 2025\)](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_Any6D_Model-free_6D_Pose_Estimation_of_Novel_Objects_CVPR_2025_paper.html) — pose from a single RGB-D reference image, no CAD model  
* [OneViewAll: One-View 6D Pose for Novel Objects (2026)](https://arxiv.org/abs/2605.07023) — semantic-prior-guided single-view pose  
* [PicoPose: Pixel-to-Pixel Correspondence for Novel-Object Pose (2025)](https://arxiv.org/abs/2504.02617) — progressive correspondence learning for pose  
* [UNOPose: Unseen-Object Pose from One Unposed RGB-D Image (2024/25)](https://arxiv.org/abs/2411.16106) — relative pose of an unseen object from a single reference  
* [DOPE: Deep Object Pose for Robotic Grasping (2018)](https://arxiv.org/abs/1809.10790) — the classic "pose so a robot can grasp it" paper  
* [YOLOE: Real-Time Seeing Anything (ICCV 2025\)](https://arxiv.org/abs/2503.07465) — real-time detection \+ segmentation from text or visual prompts  
* [YOLO-World: Real-Time Open-Vocabulary Detection (CVPR 2024\)](https://arxiv.org/abs/2401.17270) — detect arbitrary named objects in real time  
* [Grounding DINO 1.5 (2024)](https://arxiv.org/abs/2405.10300) — strong open-set / promptable detection  
* [Real-Time Open-Vocabulary Perception on Edge Devices (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12583037/) — accuracy-vs-latency study you could replicate directly  
* [CLIP: Learning Transferable Visual Models from NL Supervision (2021)](https://arxiv.org/abs/2103.00020) — image-text embeddings that power open-vocabulary work  
* [MOTIP: Multiple Object Tracking as ID Prediction (CVPR 2025\)](https://arxiv.org/abs/2403.16848) — reframes tracking as directly predicting object IDs  
* [SORT: Simple Online and Realtime Tracking (2016)](https://arxiv.org/abs/1602.00763) — the minimal, reimplementable tracker  
* [Sequential Amodal Segmentation via Cumulative Occlusion Learning (2024)](https://arxiv.org/abs/2405.05791) — segment the hidden parts of occluded objects  
* [Amodal Occlusion-Aware Instance Recovery (2026)](https://link.springer.com/article/10.1007/s00138-026-01853-6) — recent amodal segmentation under heavy occlusion  
* [HFNet: Hierarchical RGB-D Robotic Grasp Detection (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0263224125011340) — high-precision real-time grasping  
* [GG-CNN: Real-Time Generative Grasp Synthesis (2018)](https://arxiv.org/abs/1804.05172) — the grasp-rectangle approach, reimplementable  
* [Language-Guided Robot Grasping via Shape Fitting (2026)](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202501276) — grasp by fitting geometric primitives, driven by language  
* [MASt3R: Grounding Image Matching in 3D (ECCV 2024\)](https://arxiv.org/abs/2406.09756) — dense 3D reconstruction and matching from uncalibrated images  
* [MUSt3R: Multi-View Network for Stereo 3D Reconstruction (2025)](https://arxiv.org/abs/2503.01661) — multi-image 3D reconstruction  
* [Enhanced 3D Gaussian Splatting for Real Scenes (2025)](https://www.mdpi.com/1424-8220/25/22/6999) — interactive, navigable scene reconstruction  
* [OpenVLA: Open-Source Vision-Language-Action Model (2024)](https://arxiv.org/abs/2406.09246) — a robot that acts from images \+ language (your lab's direction)  
* [TrackID3x3: Multi-Player Tracking \+ Pose in Basketball (2025)](https://arxiv.org/abs/2503.18282) — players, IDs, and pose in sports video  
* [BlurBall: Ball \+ Motion-Blur Tracking for Table Tennis (2025)](https://arxiv.org/abs/2509.18387) — tracking fast, blurry objects  
* [Sports Pose Estimation & Tracking Survey (2025)](https://link.springer.com/article/10.1007/s10462-025-11344-1) — a map of the whole sports-motion field, good for scoping ideas

## **Notes**

* Interests:  
  * Thomas: VLMs, Object Detection, Robotics  
  * Darshan: Object Recognition  
  * Wade: VLMs / VLAs, Robotics

## **Research Topics**

* Object Detection, Segmentation & Tracking  
* Object Tracking  
* Image Segmentation  
* Visual SLAM  
* 3D Vision & Scene Understanding  
* Image Classification  
* Domain Adaptation & Generalization  
* 3D Gaussian Splatting  
* Trajectory Prediction

## **Domains**

* Mobile Robotics  
* Robotic Manipulation  
* Self Driving  
* Facial Recognition

