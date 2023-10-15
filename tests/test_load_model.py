"""
test_load_model.py

Unit tests for the load_model function in load_model.py.
"""

import pytest
import timm

from aiml.load_data.load_model import load_model


@pytest.fixture
def pre_trained_model():
    model = timm.create_model("resnet34", pretrained=True)
    return model

def test_load_model_with_pretrained_model(pre_trained_model):
    model = load_model(pre_trained_model, device="cpu")
    assert isinstance(model, timm.models.resnet.ResNet)  # Adjust the type as per your model

if __name__ == '__main__':
    pytest.main([__file__])
