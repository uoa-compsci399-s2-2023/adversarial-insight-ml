"""
load_model.py

This script is responsible for loading the model.
"""


import detectors

"""
The function is for loading model
args:
model(model/string):if input is string, it will find the target model by detectors
device:cpu or gpu
return 
model
"""
def load_model(model):
    if type(model) == type("a"):
        try:
            model = detectors.create_model(model, pretrained=True)
            model = model.to(device)
        except:
            print("we can't find your model")
            return None

    return model
