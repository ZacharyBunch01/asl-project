# model.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

def build_model(num_classes, pretrained=True):
    if pretrained:
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
    else:
        model = r3d_18(weights=None)

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

