"""
test_accuracy.py

This module defines a function for calculating the accuracy of a given dataset on a model.
"""


import torch

def test_accuracy(model, dataloader, device):
    """
    Calculate the accuracy of a given dataset on a pre-trained model.

    Args:
        model (torch.nn.Module): The pre-trained neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): Device ('cpu' or 'cuda') on which to perform inference.

    Returns:
        float: Accuracy of the model on the dataset.
    """
    # Initialize variables to keep track of correct predictions and total samples
    correct = 0
    total = 0

    # Set the model to evaluation mode (no gradient computation)
    model.eval()

    # Disable gradient calculation for inference
    with torch.no_grad():
        model = model.to(device)

        # Iterate through batches in the DataLoader
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            
            # Forward pass through the model
            outputs = model(x)
            
            # Get predicted class labels by selecting the class with the highest score
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.to('cpu')

            # Update total and correct counts
            total += y.size(0)
            correct += (predictions == y).sum().item()

    # Calculate accuracy and return
    accuracy = correct / total
    return accuracy
