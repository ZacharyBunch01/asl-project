import os
from pathlib import Path

import pytest

from data_prep.placeValDataInTrain import merge_val_into_train

def test_merge(tmp_path: Path):
    """
    End-to-end test;
    Create fake train/ and val/ directories
    Add Class folders and files to val/
    Run merge_val_into_train
    Assert files moved and val is cleaned up
    """
    root = tmp_path

    # Create train/ and val/
    train_dir = root / "train"
    val_dir = root / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    # Create matching class directories in both
    train_bath = train_dir / "bathroom"
    train_bed = train_dir / "bed"
    train_bath.mkdir()
    train_bed.mkdir()

    val_bath = val_dir / "bathroom"
    val_bed = val_dir / "bed"
    val_bath.mkdir()
    val_bed.mkdir()

    # Add files into val classes
    (val_bath / "bath1.jpg").write_text("x")
    (val_bath / "bath2.jpg").write_text("x")
    (val_bed / "bed1.jpg").write_text("x")

    # Run merge
    merge_val_into_train(root)

    # Files should now exist in train dirs
    assert (train_bath / "bath1.jpg").exists()
    assert (train_bath / "bath2.jpg").exists()
    assert (train_bed / "bed1.jpg").exists()

    # val class dirs should be removed
    assert not val_bath.exists()
    assert not val_bed.exists()

    # val root dir should also be removed
    assert not val_dir.exists()

def test_no_val_directory_prints_message(tmp_path: Path, capsys):
    """
    If val/ does not exist, function should print a message and return.
    """
    root = tmp_path
    (root / "train").mkdir()

    merge_val_into_train(root)

    captured = capsys.readouterr()
    assert "No val directory found" in captured.out

def test_missing_train_directory_raises(tmp_path: Path):
    """
    Train dir must exist or a FileNotFoundError is raised.
    """
    root = tmp_path
    (root / "val").mkdir()

    with pytest.raises(FileNotFoundError):
        merge_val_into_train(root)

def test_val_has_no_class_dirs(tmp_path: Path, capsys):
    """
    If val/ exists but has no subdirectories, it should print a message and return.
    """
    root = tmp_path
    (root / "train").mkdir()
    (root / "val").mkdir()

    merge_val_into_train(root)
    captured = capsys.readouterr()

    assert "No class folders found" in captured.out