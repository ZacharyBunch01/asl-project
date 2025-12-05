'''
	test_train_video_classifier.py

	PURPOSE : Test test_train_video_classifier.py
'''

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from ml_scripts.visualmodel import train_video_classifier as tvc

# Fake Video Dataset class for tests
class DummyVideoDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        # (T, C, H, W)
        video = torch.zeros(4, 3, 8, 8)
        label = 0
        return video, label

# Test nn
class DummyNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Just a dummy parameter so loss.backward() has something
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        # Create logits that depend only on a dummy parameter
        # but ignore the actual input structure.
        param = self.fc.weight.mean()
        logits = torch.zeros(batch_size, self.fc.out_features, device=x.device)
        logits = logits + param  # keep gradient connected
        return logits

# Test training epochs
def test_train_one_epoch_runs():
    dataset = DummyVideoDataset()
    loader = DataLoader(dataset, batch_size=2)
    model = DummyNet(num_classes=2).to(tvc.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    loss, acc = tvc.train_one_epoch(model, loader, criterion, optimizer, epoch=1)

    assert isinstance(loss, float)
    assert 0.0 <= acc <= 100.0

# Test model evals
def test_eval_model_runs():
    dataset = DummyVideoDataset()
    loader = DataLoader(dataset, batch_size=2)
    model = DummyNet(num_classes=2).to(tvc.device)
    criterion = nn.CrossEntropyLoss()

    loss, acc = tvc.eval_model(model, loader, criterion)

    assert isinstance(loss, float)
    assert 0.0 <= acc <= 100.0

