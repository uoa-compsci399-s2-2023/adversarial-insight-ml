"""
load_model.py

This script is responsible for loading the model.
"""


import detectors


def load_model(model, device):
    if type(model) == type("a"):
        model = detectors.create_model(model, pretrained=True)
        model = model.to(device)
        
    return model
