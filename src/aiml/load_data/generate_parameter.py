"""
generate_parameter.py

This module provides a function for generating parameters based on input dataset.
"""


import torch
import numpy as np
"""
generate the images shape, clip values and the number of classes based on input dataset.
args:
 input_shape(nparray):images shape, if it is given(not None) it will not change
 clip_values(tuple):the range of the data, if it is given(not None) it will not change
 nb_classes(int):the number of classes, if it is given(not None) it will not change
 dataset_test:given dataset
 dataloader_test:the loader of the given dataset
return:
input_shape(nparray):images shape
 clip_values(tuple):the range of the data
 nb_classes(int):the number of classes
 

"""

def generate_parameter(
    input_shape, clip_values, nb_classes, dataset_test, dataloader_test
):
    if input_shape == None:
        (x, y) = next(iter(dataset_test))
        input_shape = np.array(x.size())

    if clip_values == None:
        global_min = np.inf
        global_max = (-1) * np.inf
        s = 0
        n = 0
        b = True
        for batch in dataloader_test:
            x, _ = batch
            if b:
                b = False
            global_min = min(torch.min(x).item(), global_min)
            global_max = max(torch.max(x).item(), global_max)
        
        clip_values = (global_min, global_max)

    if nb_classes == None:
        list1 = set([y for _, y in dataset_test])  # use set to get unique classes
        nb_classes = len(list1)
    
    return (input_shape, clip_values, nb_classes)
