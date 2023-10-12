"""
check_accuracy.py

This module defines functions for calculating the accuracy of a given 
dataset on a model.
"""


import torch


def check_accuracy(model, dataloader, device):
    """
    Calculate the accuracy of a dataset using a pre-trained machine learning model.

    Parameters:
        model: The pre-trained machine learning model.
        dataloader: The dataloader for the dataset to be tested.
        device (str): The device to use, either 'cpu' or 'gpu'.

    Returns:
        float: The accuracy of the dataset when tested on the provided model.
    """
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)

            outputs = model(x)

            _, predictions = torch.max(outputs, 1)
            predictions = predictions.to("cpu")

            total += y.size(0)
            correct += (predictions == y).sum().item()

    accuracy = correct / total

    return accuracy


def check_accuracy_with_flags(model, dataloader, device):
    """
    Calculate the accuracy of a dataset using a machine learning model.

    Parameters:
        model: The machine learning model for evaluation.
        dataloader: The dataloader for the dataset to be tested.
        device (str): The device to use, either 'cpu' or 'gpu'.

    Returns:
        float: The accuracy of the dataset when tested on the provided model.
        list: A list showing which images were correctly recognized (True) or not (False).
    """
    correct = 0
    total = 0

    model.eval()
    correct_image_bool = []

    with torch.no_grad():
        for batch in dataloader:
            # obtain image and label
            x, y = batch

            # pass image to device
            x = x.to(device)

            # obtain model output
            outputs = model(x)

            # obtain model predicted label
            confidence, predictions = torch.max(outputs, 1)
            predictions = predictions.to("cpu")

            # generates a tensor giving True for correct classification
            # and False otherwise
            correct_bool = predictions == y
            total += y.size(0)
            correct += correct_bool.sum().item()
            correct_image_bool.append(correct_bool.tolist())

    accuracy = correct / total
    return accuracy, correct_image_bool
