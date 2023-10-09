"""
generate_parameter.py

This module provides a function for generating parameters based on input dataset.
"""


import torch
import numpy as np


def generate_parameter(
    input_shape, clip_values, nb_classes, dataset_test, dataloader_test
):
    """
    Generate the image shape, clip values, and number of classes based on the input dataset.

    Parameters:
        input_shape (numpy array): Image shape; if given (not None), it will remain unchanged.
        clip_values (tuple): Data range; if given (not None), it will remain unchanged.
        nb_classes (int): Number of classes; if given (not None), it will remain unchanged.
        dataset_test: The input dataset for reference.
        dataloader_test: DataLoader for the input dataset.

    Returns:
        tuple: A tuple containing (input_shape, clip_values, nb_classes).
    """
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
        # use set to get unique classes
        list1 = set([y for _, y in dataset_test])
        nb_classes = len(list1)

    return (input_shape, clip_values, nb_classes)
