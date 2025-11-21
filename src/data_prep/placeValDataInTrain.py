#!/usr/bin/env python3
"""
Merge val split into train for tiny ASL dataset.

Assumes structure:

Data/Visual_split/
    train/
        bathroom/
        bed/
        ...
    val/
        bathroom/
        bed/
        ...
    test/
        ...

After running, all files from val/<class>/ will be MOVED into train/<class>/.
The test split is untouched.
"""

import os
import shutil
from pathlib import Path


def merge_val_into_train(root: Path):
    train_dir = root / "train"
    val_dir = root / "val"

    if not val_dir.exists():
        print(f"No val directory found at {val_dir}, nothing to do.")
        return

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory does not exist: {train_dir}")

    # Iterate over each class folder in val
    class_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"No class folders found in {val_dir}, nothing to merge.")
        return

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        print(f"\nMerging class: {class_name}")

        src_class_dir = class_dir
        dst_class_dir = train_dir / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        files = [p for p in src_class_dir.iterdir() if p.is_file()]
        if not files:
            print(f"  No files in {src_class_dir}, skipping.")
            continue

        for src_path in files:
            dst_path = dst_class_dir / src_path.name
            print(f"  Moving {src_path} → {dst_path}")
            shutil.move(str(src_path), str(dst_path))

        # Optionally remove empty class dir in val
        try:
            src_class_dir.rmdir()
            print(f"  Removed empty directory {src_class_dir}")
        except OSError:
            # Not empty for some reason
            print(f"  Could not remove {src_class_dir} (not empty?)")

    # Try to remove val dir if it's empty
    try:
        val_dir.rmdir()
        print(f"\nRemoved empty val directory: {val_dir}")
    except OSError:
        print(f"\nval directory {val_dir} not removed (not empty?).")

    print("\nDone! All val samples are now in train.")


def main():
    # Default root is Data/Visual_split relative to this script
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / ".." / ".." / "Data" / "Visual_split"

    root = default_root
    print(f"Using dataset root: {root}")

    merge_val_into_train(root)


if __name__ == "__main__":
    main()

