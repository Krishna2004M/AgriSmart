#!/usr/bin/env python3
"""
Split a class-structured image dataset into train/val.

Source tree (input):
  PlantVillage/
    ├── Tomato___healthy/
    ├── Tomato___Early_blight/
    └── ...

Destination tree (created automatically):
  PlantVillage_split/
    ├── train/
    │   ├── Tomato___healthy/
    │   └── ...
    └── val/
        ├── Tomato___healthy/
        └── ...

Usage (WSL/Ubuntu example):
  python Make_Split.py \
    --src "/mnt/e/capsteon P/Agri/PlantVillage" \
    --dst "/mnt/e/capsteon P/Agri/PlantVillage_split" \
    --val-ratio 0.2
"""

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def gather_images(cls_dir: Path) -> List[Path]:
    # Only take files in the class directory (not nested subdirs)
    return [p for p in cls_dir.iterdir() if is_image(p)]


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def split_counts(n: int, val_ratio: float) -> int:
    if n <= 0:
        return 0
    # Keep at least 1 in val when possible, and at least 1 in train when n>1
    raw = int(round(n * val_ratio))
    raw = max(1 if n > 1 else 1, raw)  # ensure >=1 val if n>=1
    raw = min(n - 1 if n > 1 else 1, raw)  # leave at least 1 for train if n>1
    return raw


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val.")
    parser.add_argument("--src", type=Path, required=True,
                        help="Path to original dataset with class subfolders.")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Path to output folder (will be created).")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction for validation split (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling.")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying (default copies).")
    args = parser.parse_args()

    SRC: Path = args.src
    DST: Path = args.dst
    VAL_RATIO: float = args.val_ratio
    rng_seed: int = args.seed
    do_move: bool = args.move

    # --- validations ---
    if not SRC.exists():
        raise FileNotFoundError(f"Source path not found: {SRC}\n"
                                f"Tip (WSL): Windows E:\\... -> /mnt/e/...")

    if not SRC.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {SRC}")

    # Create destination/train/val
    TRAIN = DST / "train"
    VAL = DST / "val"
    safe_mkdir(TRAIN)
    safe_mkdir(VAL)

    random.seed(rng_seed)

    total_train = 0
    total_val = 0
    classes_processed = 0

    print(f"\nSplitting dataset")
    print(f"  Source      : {SRC}")
    print(f"  Destination : {DST}")
    print(f"  Val ratio   : {VAL_RATIO:.2f}")
    print(f"  Mode        : {'MOVE' if do_move else 'COPY'}")
    print("-" * 60)

    class_dirs = [d for d in SRC.iterdir() if d.is_dir()]
    if not class_dirs:
        print("No class subfolders found in source. Nothing to do.")
        return

    for cls_dir in sorted(class_dirs):
        files = gather_images(cls_dir)
        if not files:
            print(f"[WARN] No images in: {cls_dir.name} (skipped)")
            continue

        random.shuffle(files)
        n = len(files)
        n_val = split_counts(n, VAL_RATIO)
        n_train = n - n_val

        # Create output class dirs
        train_cls = TRAIN / cls_dir.name
        val_cls = VAL / cls_dir.name
        safe_mkdir(train_cls)
        safe_mkdir(val_cls)

        # Split
        val_files = files[:n_val]
        train_files = files[n_val:]

        # Copy/Move
        op = shutil.move if do_move else shutil.copy2
        for f in train_files:
            op(str(f), str(train_cls / f.name))
        for f in val_files:
            op(str(f), str(val_cls / f.name))

        total_train += n_train
        total_val += n_val
        classes_processed += 1
        print(f"[OK] {cls_dir.name:35s}  train: {n_train:4d}   val: {n_val:4d}   total: {n}")

    print("-" * 60)
    print(f"Classes processed : {classes_processed}")
    print(f"Total -> train: {total_train}   val: {total_val}   all: {total_train + total_val}")
    print("Done ✅")


if __name__ == "__main__":
    main()
