"""
test_accuracy.py

This module defines functions for calculating the accuracy of a given 
dataset on a model.
"""

import pytest
import torch


class TestEvolution:
    def test_accuracy(model, dataloader, device):
        """This function returns the accuracy of a given dataset on a pre-trained model."""
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

    """
    calculating the accuracy of a given dataset on a model.
    args:
    model:ML model
    dataloader: the dataloader of the tested dataset
    device: cpu or gpu
    return:
    accuracy(float):the accuracy of the dataset tested on the given model
    correct_image_bool: a list that figure out which images are correctly recognised and which images are misrecognised.

    """
    def test_accuracy_with_flags(model, dataloader, device):
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
