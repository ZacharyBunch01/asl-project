'''
	test_transforms.py

	PURPOSE : Test functions within the transforms.py script
'''

import torch
import torchvision.transforms as T

from ml_scripts.visualmodel.transforms import get_video_transform

# Test frame normalization and resizing
def test_get_video_transform_structure():
    transform = get_video_transform()

    assert isinstance(transform, T.Compose)
    # Should contain a Resize and a Normalize transform
    types = {type(t) for t in transform.transforms}
    assert any(issubclass(t, T.Resize) for t in types)
    assert any(issubclass(t, T.Normalize) for t in types)

# Test video image resolution resizing
def test_get_video_transform_applies_to_frame():
    transform = get_video_transform()

    # Single frame: (C, H, W)
    frame = torch.rand(3, 50, 50)

    out = transform(frame)
    assert isinstance(out, torch.Tensor)
    # Expect resized to (112, 112) by default
    assert out.shape[0] == 3
    assert out.shape[1] == 112
    assert out.shape[2] == 112

