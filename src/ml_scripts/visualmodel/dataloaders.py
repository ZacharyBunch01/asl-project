'''
	DataLoaders.py

	Splits the dataset into train and test folders.
'''

from torch.utils.data import DataLoader
from .dataset import VideoDataset
from .transforms import get_video_transform

def get_dataloaders(root="../../../Data/Visual_split/", batch_size=4, num_frames=16):
    train_dir = f"{root}/train"
    test_dir = f"{root}/test"

    transform = get_video_transform()

    train_ds = VideoDataset(train_dir, num_frames=num_frames, transform=transform)
    test_ds  = VideoDataset(test_dir,  num_frames=num_frames, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, train_ds.classes

