"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Fetch the SUN RGB-D mirror from Hugging Face and extract it into
data/sunrgbd/, so config.SUNRGBD_ROOT points at real files on disk instead of
an unopened zip sitting in the HF cache.
"""

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

from collision_avoidance.config import SUNRGBD_ROOT

REQUIRED_SUBDIRS = ("image", "depth", "calib")


def _looks_complete(root: Path) -> bool:
    return all((root / sub).is_dir() and any((root / sub).glob("*")) for sub in REQUIRED_SUBDIRS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-extract even if data already looks present")
    args = ap.parse_args()

    if _looks_complete(SUNRGBD_ROOT) and not args.force:
        n_images = len(list((SUNRGBD_ROOT / "image").glob("*.jpg")))
        print(f"SUN RGB-D already extracted at {SUNRGBD_ROOT} ({n_images} frames). Use --force to re-extract.")
        return

    print("fetching SUN RGB-D mirror from Hugging Face (~35 GB, this can take a while)...")
    snapshot_dir = Path(snapshot_download("youdaoyzbx/processed_sunrgbd", repo_type="dataset"))
    zip_path = next(snapshot_dir.glob("*.zip"))
    print(f"downloaded to cache: {zip_path}")

    extract_root = SUNRGBD_ROOT.parent  # data/sunrgbd/ ; the zip contains its own sunrgbd_trainval/ folder
    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"extracting into {extract_root} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)

    if not _looks_complete(SUNRGBD_ROOT):
        raise RuntimeError(f"extraction finished but {SUNRGBD_ROOT} is still missing expected subdirs")

    n_images = len(list((SUNRGBD_ROOT / "image").glob("*.jpg")))
    print(f"done: {n_images} frames available at {SUNRGBD_ROOT}")


if __name__ == "__main__":
    main()
