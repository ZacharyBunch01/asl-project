#!/usr/bin/env python3
"""
Split ASL video dataset into train/val/test folders.

Assumes input structure:

Data/Visual/
    bathroom/
        vid1.mp4
        vid2.mp4
        ...
    bed/
    hello/
    ...

Creates output structure:

Data/Visual_split/
    train/
        bathroom/
        bed/
        hello/
        ...
    val/
        ...
    test/
        ...

By default, it COPIES files (safer). You can enable --move to move instead.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def split_list(items, train_ratio, val_ratio, test_ratio):
    """Split list into train/val/test according to given ratios."""
    n = len(items)
    if n == 0:
        return [], [], []

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Put the remainder into test
    n_test = n - n_train - n_val

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]

    assert len(train_items) + len(val_items) + len(test_items) == n
    return train_items, val_items, test_items


def split_dataset(
    src_root: Path,
    dst_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    move_files: bool = False,
):
    random.seed(seed)

    # Basic sanity check on ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    # Create root of split dataset
    for split in ["train", "val", "test"]:
        (dst_root / split).mkdir(parents=True, exist_ok=True)

    # Iterate over each class folder
    class_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {src_root}")

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")

        # Collect video files
        video_files = [p for p in class_dir.iterdir()
                       if p.is_file() and is_video_file(p)]

        if not video_files:
            print(f"  WARNING: no video files found in {class_dir}, skipping.")
            continue

        # Shuffle videos
        random.shuffle(video_files)

        # Split into train/val/test
        train_files, val_files, test_files = split_list(
            video_files, train_ratio, val_ratio, test_ratio
        )

        print(f"  Total: {len(video_files)}, "
              f"train: {len(train_files)}, "
              f"val: {len(val_files)}, "
              f"test: {len(test_files)}")

        # Make class subdirs in each split
        for split in ["train", "val", "test"]:
            split_class_dir = dst_root / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

        # Helper to copy/move
        def transfer(files, split_name):
            split_class_dir = dst_root / split_name / class_name
            for src_path in files:
                dst_path = split_class_dir / src_path.name
                if move_files:
                    shutil.move(str(src_path), str(dst_path))
                else:
                    shutil.copy2(str(src_path), str(dst_path))

        transfer(train_files, "train")
        transfer(val_files, "val")
        transfer(test_files, "test")

    print("\nDone! Split dataset created at:", dst_root.resolve())
    if move_files:
        print("Files were MOVED from the original location.")
    else:
        print("Files were COPIED from the original location.")


def main():
    parser = argparse.ArgumentParser(
        description="Split ASL video dataset into train/val/test."
    )
    parser.add_argument(
        "--src-root",
        type=str,
        default="../../Data/Visual",
        help="Path to source dataset root (class folders inside). "
             "Default: Data/Visual",
    )
    parser.add_argument(
        "--dst-root",
        type=str,
        default="../../Data/Visual_split",
        help="Path to destination root for split dataset. "
             "Default: Data/Visual_split",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (DANGEROUS, original videos will be relocated).",
    )

    args = parser.parse_args()

    split_dataset(
        src_root=Path(args.src_root),
        dst_root=Path(args.dst_root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        move_files=args.move,
    )


if __name__ == "__main__":
    main()

