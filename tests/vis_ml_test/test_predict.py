'''
	test_predict.py

	PURPOSE : Test test_predict.py functions.
'''

import torch
import pytest

from ml_scripts.visualmodel import predict as predict_module

# Input fake video data to test tensor
def fake_read_video(path, pts_unit="sec"):
    # (T, H, W, C)
    video = torch.randint(0, 255, (10, 64, 64, 3), dtype=torch.uint8)
    audio = torch.empty(0)
    info = {}
    return video, audio, info

# Test transform getter on fake video data
def fake_get_video_transform():
    # Identity "transform"
    return lambda frame: frame

# Test frame-padding on short videos below the desired length
def test_sample_frames_short_video():
    video = torch.zeros(5, 1, 1, 1)
    idxs = predict_module.sample_frames(video, num_frames=8)

    assert len(idxs) == 8
    assert all(0 <= i < 5 for i in idxs)

# Test sample frames on videos longer than the desired length
def test_sample_frames_long_video():
    video = torch.zeros(20, 1, 1, 1)
    idxs = predict_module.sample_frames(video, num_frames=8)

    assert len(idxs) == 8
    # Indices must be valid
    assert all(0 <= i < 20 for i in idxs)

# Tests video tensor shape loader function on fake data
def test_load_video_tensor_shapes(monkeypatch):
    monkeypatch.setattr(predict_module, "read_video", fake_read_video)
    monkeypatch.setattr(predict_module, "get_video_transform", fake_get_video_transform)

    clip = predict_module.load_video_tensor("dummy.mp4", num_frames=4)

    # (B, C, T, H, W)
    assert isinstance(clip, torch.Tensor)
    assert clip.shape == (1, 3, 4, 64, 64)

# Dummy Model test class
class DummyModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Dummy parameter so optimizer/state_dict make sense
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size = x.size(0)
        logits = torch.zeros(batch_size, self.num_classes, device=x.device)
        # Make class index 1 the winner
        logits[:, 1] = 10.0
        return logits

# Tests class prediction
def test_predict_uses_checkpoint_and_returns_class(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_module, "read_video", fake_read_video)
    monkeypatch.setattr(predict_module, "get_video_transform", fake_get_video_transform)

    # Fake build_model used inside predict.py
    def fake_build_model(num_classes, pretrained):
        return DummyModel(num_classes)

    monkeypatch.setattr(predict_module, "build_model", fake_build_model)

    # Create a fake checkpoint
    class_names = ["class0", "class1", "class2"]
    dummy_model = DummyModel(len(class_names))
    ckpt = {
        "model_state_dict": dummy_model.state_dict(),
        "class_names": class_names,
    }
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(ckpt, ckpt_path)

    pred, conf = predict_module.predict("dummy.mp4", checkpoint=str(ckpt_path))

    assert pred == "class1"
    assert 0.0 <= conf <= 1.0









