import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_video

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        """
        root_dir: path to 'train' or 'test' dir
        structure: root_dir/class_name/*.mp4
        """
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform

        self.classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []  # list of (video_path, label_idx)
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for vid_path in cls_dir.glob("*.mp4"):
                self.samples.append((vid_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def _select_frame_indices(self, total_frames):
        """Uniformly sample num_frames indices from [0, total_frames)."""
        if total_frames <= self.num_frames:
            # pad by repeating frames if too short
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices += indices
            return indices[:self.num_frames]
        else:
            step = total_frames / self.num_frames
            return [int(step * i) for i in range(self.num_frames)]

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        # video: (T, H, W, C) in [0, 255], audio ignored
        video, _, _ = read_video(str(video_path), pts_unit='sec')
        # video shape: (T, H, W, C)
        total_frames = video.shape[0]

        frame_indices = self._select_frame_indices(total_frames)
        video = video[frame_indices]  # (num_frames, H, W, C)

        # To tensor & normalize to [0,1]
        # Then we apply image transforms frame-wise.
        video = video.float() / 255.0  # (T, H, W, C)
        # Rearrange to (T, C, H, W) for transforms
        video = video.permute(0, 3, 1, 2)

        if self.transform:
            # apply transform to each frame
            video = torch.stack([self.transform(frame) for frame in video], dim=0)

        # final shape: (T, C, H, W)
        return video, label

