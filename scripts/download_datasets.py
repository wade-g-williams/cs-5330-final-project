"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Download the datasets into data/. The loaders index frames randomly and we
spot-check them by hand, so the data has to live as real files on a stable, inspectable path.

    python scripts/download_datasets.py sunrgbd     # ~35 GB
    python scripts/download_datasets.py kitti       # ~12 GB (images + calib + labels)
    python scripts/download_datasets.py all

Downloaded archives are deleted once extraction is verified..
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from huggingface_hub import scan_cache_dir, snapshot_download

from collision_avoidance import config

# KITTI's official site wants an approved institutional account, but AWS Open Data serves
# the identical files with no login.
# Each entry is (archive, description, bytes, the training/ subdirectory it fills). Carrying the
# subdirectory lets us skip an archive whose data is already on disk instead of re-pulling 12 GB.
KITTI_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
KITTI_ARCHIVES = [
    ("data_object_image_2.zip", "left colour images", 12_569_945_557, "image_2"),
    ("data_object_calib.zip", "camera calibration", 26_854_811, "calib"),
    ("data_object_label_2.zip", "3D box labels", 5_601_213, "label_2"),
]

# The pipeline's outdoor depth comes from Depth Anything V2, so the laser is not needed to run
# it. It is needed to *bound* it: feeding real LiDAR through the identical Stage 2-6 head
# separates "our geometry is weak" from "the estimated depth is weak", which is the strongest
# ablation available to us. Opt in with --with-lidar; it is 27 GB on its own.
KITTI_LIDAR_ARCHIVE = ("data_object_velodyne.zip", "Velodyne LiDAR scans", 28_750_710_812, "velodyne")

SUNRGBD_REPO = "youdaoyzbx/processed_sunrgbd"

# After the data is downloaded, check the number of frames to verify they downloaded completely
SUNRGBD_SUBDIRS = {"image": 10_335, "depth": 10_335, "calib": 10_335}
KITTI_SUBDIRS = {"image_2": 7_481, "calib": 7_481, "label_2": 7_481}
KITTI_LIDAR_SUBDIRS = {"velodyne": 7_481}


def _human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024


def _check_counts(root: Path, expected: dict[str, int]) -> list[str]:
    """Return a human-readable problem per subdirectory that is missing or short-counted."""
    problems = []
    for subdir, want in expected.items():
        path = root / subdir
        if not path.is_dir():
            problems.append(f"{path} is missing")
            continue
        got = sum(1 for _ in path.iterdir())
        if got != want:
            problems.append(f"{path} has {got} files, expected {want}")
    return problems


def _sunrgbd_complete() -> bool:
    return not _check_counts(config.SUNRGBD_ROOT, SUNRGBD_SUBDIRS)


def _kitti_complete(with_lidar: bool = False) -> bool:
    expected = KITTI_SUBDIRS | (KITTI_LIDAR_SUBDIRS if with_lidar else {})
    return not _check_counts(config.KITTI_ROOT / "training", expected)


def _reclaim_hf_cache(repo_id: str) -> None:
    """Drop the cached archive once it has been extracted, so we don't store the data twice."""
    cache = scan_cache_dir()
    revisions = [rev.commit_hash
                 for repo in cache.repos if repo.repo_id == repo_id
                 for rev in repo.revisions]
    if not revisions:
        return
    strategy = cache.delete_revisions(*revisions)
    freed = strategy.expected_freed_size_str
    strategy.execute()
    print(f"    reclaimed {freed} from the Hugging Face cache")


def _stream_download(url: str, dest: Path, expected_bytes: int) -> Path:
    """Download with a resumable byte-range request, so a dropped 12 GB transfer isn't fatal."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have >= expected_bytes:
        print(f"    already downloaded ({_human(have)})")
        return dest

    headers = {"Range": f"bytes={have}-"} if have else {}
    if have:
        print(f"    resuming at {_human(have)} / {_human(expected_bytes)}")

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()

        # A server is free to ignore Range and answer 200 with the whole file. Appending that
        # to a partial download would corrupt it silently, so start over instead.
        if have and r.status_code != 206:
            print("    server ignored the resume request -- restarting from zero")
            have = 0

        # Content-Length covers the remaining bytes on a 206 and the whole file on a 200, so
        # either way it should complete us to expected_bytes. A mismatch means the mirror
        # changed or we were handed an error page -- catch it before writing 12 GB of it.
        remaining = r.headers.get("Content-Length")
        offered = have + int(remaining) if remaining is not None else expected_bytes
        if offered != expected_bytes:
            # Exact bytes, not _human: this number gets pasted straight into KITTI_ARCHIVES.
            raise RuntimeError(
                f"{url}\n  server offers {offered:_} bytes, expected {expected_bytes:_} "
                f"-- update KITTI_ARCHIVES if the mirror moved")

        with open(dest, "ab" if have else "wb") as f:
            done = have
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r    {_human(done)} / {_human(expected_bytes)} "
                      f"({done / expected_bytes:.0%})", end="", flush=True)
    print()

    # A connection that dies mid-transfer ends the loop without raising, so the size is the
    # only thing that says whether we got it all. Leave the partial file for the next resume.
    got = dest.stat().st_size
    if got != expected_bytes:
        raise RuntimeError(f"{dest.name} stopped at {got:_} of {expected_bytes:_} bytes "
                           f"({_human(got)} of {_human(expected_bytes)}); re-run to resume")
    return dest


def download_sunrgbd(force: bool = False, keep_archives: bool = False) -> None:
    """Pre-extracted mirror in mmdetection3d layout -- sidesteps SUN RGB-D's MATLAB tooling."""
    if _sunrgbd_complete() and not force:
        print(f"SUN RGB-D already present at {config.SUNRGBD_ROOT} "
              f"({SUNRGBD_SUBDIRS['image']} frames). Use --force to redo.")
        return

    print("SUN RGB-D: fetching mirror from Hugging Face (~35 GB, this takes a while)...")
    snapshot_dir = Path(snapshot_download(SUNRGBD_REPO, repo_type="dataset"))
    zip_path = next(snapshot_dir.glob("*.zip"))

    extract_root = config.SUNRGBD_ROOT.parent      # the zip carries its own sunrgbd_trainval/ folder
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"SUN RGB-D: extracting into {extract_root} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)

    problems = _check_counts(config.SUNRGBD_ROOT, SUNRGBD_SUBDIRS)
    if problems:
        raise RuntimeError("SUN RGB-D extraction is incomplete:\n  " + "\n  ".join(problems))

    # Only now that the extraction is verified is the archive safe to drop.
    if not keep_archives:
        _reclaim_hf_cache(SUNRGBD_REPO)
    print(f"SUN RGB-D: done, {SUNRGBD_SUBDIRS['image']} frames at {config.SUNRGBD_ROOT}")


def download_kitti(force: bool = False, keep_archives: bool = False,
                   with_lidar: bool = False) -> None:
    """KITTI 3D object detection: left images, calibration, labels, optionally the LiDAR."""
    archives = KITTI_ARCHIVES + ([KITTI_LIDAR_ARCHIVE] if with_lidar else [])
    expected = KITTI_SUBDIRS | (KITTI_LIDAR_SUBDIRS if with_lidar else {})

    if _kitti_complete(with_lidar) and not force:
        print(f"KITTI already present at {config.KITTI_ROOT} "
              f"({KITTI_SUBDIRS['image_2']} frames). Use --force to redo.")
        return

    archive_dir = config.KITTI_ROOT / "_archives"
    training = config.KITTI_ROOT / "training"
    config.KITTI_ROOT.mkdir(parents=True, exist_ok=True)

    for name, description, size, subdir in archives:
        # Adding --with-lidar to an existing download should fetch only the LiDAR, so each
        # archive is skipped independently once the directory it fills is complete.
        if not force and not _check_counts(training, {subdir: expected[subdir]}):
            print(f"KITTI: training/{subdir}/ already complete, skipping {name}")
            continue
        print(f"KITTI: {name} -- {description} ({_human(size)})")
        archive = _stream_download(f"{KITTI_BASE}/{name}", archive_dir / name, size)
        print(f"    extracting into {config.KITTI_ROOT} ...")
        with zipfile.ZipFile(archive) as zf:            # each zip holds training/ and testing/
            zf.extractall(config.KITTI_ROOT)

    problems = _check_counts(training, expected)
    if problems:
        raise RuntimeError("KITTI extraction is incomplete:\n  " + "\n  ".join(problems))

    if not keep_archives:
        shutil.rmtree(archive_dir, ignore_errors=True)
    print(f"KITTI: done, {KITTI_SUBDIRS['image_2']} training frames at {config.KITTI_ROOT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Purpose: ")[1].split("\n\n")[0])
    ap.add_argument("dataset", choices=["sunrgbd", "kitti", "all"], nargs="?", default="all")
    ap.add_argument("--force", action="store_true", help="re-download even if the data looks present")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep the downloaded zips instead of reclaiming the space after extracting")
    ap.add_argument("--with-lidar", action="store_true",
                    help="also fetch KITTI's Velodyne scans (27 GB) for the LiDAR-oracle ablation")
    args = ap.parse_args()

    if args.dataset in ("sunrgbd", "all"):
        download_sunrgbd(args.force, args.keep_archives)
    if args.dataset in ("kitti", "all"):
        download_kitti(args.force, args.keep_archives, args.with_lidar)


if __name__ == "__main__":
    main()
