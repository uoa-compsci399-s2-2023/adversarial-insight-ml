"""
load_model.py

This script is responsible for loading the model.
"""


#import detectors
from robustbench.utils import load_model


def load_model(model, device):
    """
    Load a machine learning model.

    Parameters:
        model (model or string): If a string is provided, it will search for 
            the target model by detectors.
        device (string): The device to use, either 'cpu' or 'gpu'.

    Returns:
        model: The loaded machine learning model.
    """
    
    if type(model) == type("a"):
        try:
            #model = detectors.create_model(model, pretrained=True)
            model = model.to(device)
        except:
            try:
                model= load_model(model)
                model = model.to(device)
            except:
                print("We can't find your model.")
                return None

    return model
