'''
	test_model.py

	PURPOSE : Test test_model.py functions.
'''

import torch.nn as nn

from ml_scripts.visualmodel.model import build_model


def test_build_model_output_layer_num_classes():
    num_classes = 5
    model = build_model(num_classes=num_classes, pretrained=False)

    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == num_classes


def test_build_model_pretrained_flag():
    # Just ensure both branches construct a model with the same head size
    m_pretrained = build_model(num_classes=3, pretrained=True)
    m_scratch = build_model(num_classes=3, pretrained=False)

    assert m_pretrained.fc.in_features == m_scratch.fc.in_features
    assert m_pretrained.fc.out_features == m_scratch.fc.out_features

