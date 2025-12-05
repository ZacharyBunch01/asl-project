'''
	test_dataset.py

	PURPOSE : Test dataset.py functions
'''

from pathlib import Path

import torch
import pytest

from ml_scripts.visualmodel import dataset

# Returns fake video tensor (20 frames) 
def fake_read_video(path, pts_unit="sec"):
    # Return (T, H, W, C)
    video = torch.randint(0, 255, (20, 64, 64, 3), dtype=torch.uint8)
    audio = torch.empty(0)
    info = {}
    return video, audio, info

# Test video classifier
def test_video_dataset_discovers_classes_and_len(tmp_path, monkeypatch):
    root = tmp_path / "train"
    (root / "class_a").mkdir(parents=True)
    (root / "class_b").mkdir()

    # Create empty .mp4 files (we'll fake read_video)
    for i in range(2):
        (root / "class_a" / f"a_{i}.mp4").touch()
    (root / "class_b" / "b_0.mp4").touch()

    # Patch dataset.read_video (imported at module level in dataset.py)
    monkeypatch.setattr(dataset, "read_video", fake_read_video)

    ds = dataset.VideoDataset(root, num_frames=16, transform=None)

    # Classes are lexicographically sorted
    assert ds.classes == ["class_a", "class_b"]
    assert len(ds) == 3

# Ensure __getitem__ returns the shape and label of the video
def test_video_dataset_getitem_shape_and_label(tmp_path, monkeypatch):
    root = tmp_path / "train"
    (root / "cls").mkdir(parents=True)
    (root / "cls" / "vid_0.mp4").touch()

    monkeypatch.setattr(dataset, "read_video", fake_read_video)

    ds = dataset.VideoDataset(root, num_frames=16, transform=None)

    video, label = ds[0]
    assert isinstance(video, torch.Tensor)
    # (T, C, H, W)
    assert video.shape == (16, 3, 64, 64)
    assert isinstance(label, int)

# Ensure each vieo with less than the desired amount of frames are padded 
def test_video_dataset_frame_selection_padding(monkeypatch, tmp_path):
    # Test _select_frame_indices when total_frames < num_frames
    root = tmp_path / "train"
    (root / "cls").mkdir(parents=True)
    (root / "cls" / "vid_0.mp4").touch()

    def short_read_video(path, pts_unit="sec"):
        video = torch.randint(0, 255, (5, 32, 32, 3), dtype=torch.uint8)
        return video, torch.empty(0), {}

    monkeypatch.setattr(dataset, "read_video", short_read_video)

    ds = dataset.VideoDataset(root, num_frames=8, transform=None)
    video, _ = ds[0]

    assert video.shape == (8, 3, 32, 32)

# Ensure each frame is transformed
def test_video_dataset_applies_transform_per_frame(tmp_path, monkeypatch):
    root = tmp_path / "train"
    (root / "cls").mkdir(parents=True)
    (root / "cls" / "vid_0.mp4").touch()

    monkeypatch.setattr(dataset, "read_video", fake_read_video)

    # Simple transform that doubles values
    def fake_transform(frame):
        return frame * 2

    ds = dataset.VideoDataset(root, num_frames=4, transform=fake_transform)
    video, _ = ds[0]

    assert video.shape == (4, 3, 64, 64)
    # Check that values are in [0, 2] since original were [0,1]
    assert (video >= 0).all()
    assert (video <= 2).all()

