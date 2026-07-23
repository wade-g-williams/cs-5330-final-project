# CS 5330 Final Project

**Thomas Kulch, Darshan Kedari, Wade Williams**
CS 5330: Computer Vision & Pattern Recognition — Summer 2026

## Project Description

RGB-D Object Detection for Collision Avoidance. We are building a perception pipeline that converts an
RGB-D video stream into a top-down Bird's Eye View (BEV) occupancy map with real-world distances for
collision avoidance. Each frame, a COCO-pretrained YOLO11 detector finds and classifies objects in the RGB
image, the aligned depth map is back-projected into a metric 3D point cloud, RANSAC removes the ground
plane, and clustering groups the remaining points into obstacles, so that objects the detector was never
trained on are still caught and labeled "unknown." Detections and clusters are fused so every obstacle
gets a class, a 3D centroid, and a bounding box on the map. We evaluate outdoors on KITTI (using Depth
Anything V2 for metric depth, since KITTI has no depth camera) and indoors on SUN RGB-D and our own
RealSense recordings (using both sensor depth and Depth Anything V2 on the same frames) — measuring how
well estimated depth substitutes for a real depth sensor. Planned extensions include a live demo with
collision alerts, closed-loop navigation in Habitat-Sim using our BEV map, detector and depth fine-tuning
ablations, and possibly 3D scene reconstruction.

Key dates: progress check-in email **July 31, 2026** · final report, code, and presentation due
**August 13, 2026**.

## Links

- Demo video: *TBD (will be hosted on Google Drive or YouTube — not submitted to Gradescope)*
- Our recorded RGB-D dataset: *TBD (will be hosted on Google Drive)*
- Datasets: [KITTI 3D object detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) ·
  [SUN RGB-D](https://rgbd.cs.princeton.edu/) · [nuScenes](https://www.nuscenes.org/nuscenes) (possible
  KITTI substitute)

## Building

- **Proposal (LaTeX, Project-5 format)**: `cd report && pdflatex proposal.tex && pdflatex proposal.tex`
  (two passes resolve citations). The `fphw.cls` (proposal) and `IEEEtran.cls` (final report, IEEE
  2-column) class files are vendored in `report/`, so no TeX package installs are needed.
- **Proposal (.docx)**: `cd docs && pandoc proposal.md -f gfm -t docx -o proposal.docx`
- **Docs PDFs** (project-ideas, research): generated from the corresponding `.md` via
  `pandoc <name>.md -f gfm -t html5 -s --embed-resources -o <name>.html` followed by
  `google-chrome --headless --no-pdf-header-footer --print-to-pdf=<name>.pdf <name>.html`. Regenerate
  after editing the markdown; do not edit PDFs directly.

## Repository Layout

- `docs/` — planning and research documents
  - `proposal.md` / `proposal.docx` — the final project proposal (submitted form)
  - `project-ideas.md` / `.pdf` — seven detailed project variants with pipelines, data, evaluation,
    training add-ons, and 3D-reconstruction add-ons
  - `research.md` / `.pdf` — hub document: all datasets, simulators, challenges, and references
    considered, with verdicts
  - `datasets-and-simulators.md` — deep dive with download logistics and verified gotchas
  - `papers.md` — venue-verified reading list (the professor requires true peer-reviewed venues)
  - `idea-variants.md` — alternative project shapes kept as pivot options
  - `assignment.md` — the course's final project assignment text
- `report/` — LaTeX: `proposal.tex` / `proposal.pdf` (fphw class, matching our Project 5 report format),
  plus vendored `fphw.cls` and `IEEEtran.cls` (the final report must use IEEE 2-column format)
- `archive/` — superseded drafts
