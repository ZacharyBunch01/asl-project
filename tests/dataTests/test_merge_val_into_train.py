import os
from pathlib import Path

import pytest

from data_prep.placeValDataInTrain import merge_val_into_train

def test_merge(tmp_path: Path):
    # Create root directory structure
    root = tmp_path

    # Create train/ and val/
    train_dir = root / "train"
    val_dir = root / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    # Create matching class directories inside train/ and val/
    train_bath = train_dir / "bathroom"
    train_bed = train_dir / "bed"
    train_bath.mkdir()
    train_bed.mkdir()

    val_bath = val_dir / "bathroom"
    val_bed = val_dir / "bed"
    val_bath.mkdir()
    val_bed.mkdir()

    # Put fake image files into the validation directories
    (val_bath / "bath1.jpg").write_text("x")
    (val_bath / "bath2.jpg").write_text("x")
    (val_bed / "bed1.jpg").write_text("x")

    # Run the merge logic
    merge_val_into_train(root)

    # Verify validation images were moved into the correct train folders
    assert (train_bath / "bath1.jpg").exists()
    assert (train_bath / "bath2.jpg").exists()
    assert (train_bed / "bed1.jpg").exists()

    # After merge, val/* class directories should be deleted
    assert not val_bath.exists()
    assert not val_bed.exists()

    # And val/ itself should be removed entirely
    assert not val_dir.exists()

def test_no_val_directory_prints_message(tmp_path: Path, capsys):
    # Create only train/, no val/ directory
    root = tmp_path
    (root / "train").mkdir()

    # Run function — expected: print message and exit early
    merge_val_into_train(root)

    # Capture printed output
    captured = capsys.readouterr()
    assert "No val directory found" in captured.out

def test_missing_train_directory_raises(tmp_path: Path):
    # Create val/ but NOT train/
    root = tmp_path
    (root / "val").mkdir()

    # Expected: raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        merge_val_into_train(root)

def test_val_has_no_class_dirs(tmp_path: Path, capsys):
    # Create train/ and empty val/ directory
    root = tmp_path
    (root / "train").mkdir()
    (root / "val").mkdir()

    # Run function — expected: prints message about missing class folders
    merge_val_into_train(root)
    captured = capsys.readouterr()

    assert "No class folders found" in captured.out