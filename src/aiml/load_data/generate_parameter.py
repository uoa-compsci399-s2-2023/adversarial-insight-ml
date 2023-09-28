"""
generate_parameter.py

This module provides a function for generating parameters based on input data and settings.
"""


import torch
import numpy as np


def find_clip_range(dataloader_test):
    """Return the range of a dataset."""
    global_min = float('inf')
    global_max = float('-inf')
    s = 0
    n = 0
    b = True
    for batch in dataloader_test:
        x, _ = batch
        if b:
            print(batch)
            b = False
        global_min = min(torch.min(x).item(), global_min)
        global_max = max(torch.max(x).item(), global_max)
    return global_min, global_max


def generate_parameter(input_shape, clip_values, nb_classes, dataset_test, dataloader_test):
    # Get input_shape from dataset_test if not given as parameter
    if input_shape is None:
        (x, y) = next(iter(dataset_test))
        input_shape = tuple(np.array(x.size()))
        print(f'input_shape: {input_shape}')

    # Define clip_values from dataloader_test if not given as parameter
    if clip_values is None:
        clip_values = global_min, global_max
        print(f'Min: {global_min}, Max: {global_max}')

    # Calculate nb_classes from dataset_test if not given as parameter
    if nb_classes is None:
        # Use set to get unique classes
        unique_classes = set(y for _, y in dataset_test)
        nb_classes = len(unique_classes)

    return input_shape, clip_values, nb_classes
