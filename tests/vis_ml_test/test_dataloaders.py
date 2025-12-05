'''
	test_dataloaders.py

	PURPOSE : Test test_dataloaders.py functions.
'''

import torch
from torch.utils.data import Dataset

from ml_scripts.visualmodel import dataloaders

class DummyVideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.classes = ["cls_a", "cls_b"]

    def __len__(self):
        return 6

    def __getitem__(self, idx):
        # Return (T, C, H, W) and label
        video = torch.zeros(self.num_frames, 3, 112, 112)
        label = 0
        return video, label


def test_get_dataloaders_uses_dataset_and_returns_classes(monkeypatch):
    # Patch VideoDataset and get_video_transform inside dataloaders module
    monkeypatch.setattr(dataloaders, "VideoDataset", DummyVideoDataset)
    monkeypatch.setattr(dataloaders, "get_video_transform", lambda: (lambda x: x))

    train_loader, test_loader, classes = dataloaders.get_dataloaders(
        root="dummy_root",
        batch_size=2,
        num_frames=8,
    )

    assert classes == ["cls_a", "cls_b"]
    assert len(train_loader.dataset) == 6
    assert len(test_loader.dataset) == 6

    batch_videos, batch_labels = next(iter(train_loader))
    # (B, T, C, H, W)
    assert batch_videos.shape == (2, 8, 3, 112, 112)
    assert batch_labels.shape == (2,)

