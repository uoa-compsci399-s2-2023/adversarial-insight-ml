"""
test_white_box.py

This module contains functions for testing white-box attack methods on a
PyTorch classifier model, and returns respecitve accuracies in a list.
"""


import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from aiml.evaluation.test_accuracy import test_accuracy
from aiml.attack.adversarial_attacks import * 


def test_white_box_attack(
    attack_method,
    model,
    pytorch_classifier,
    dataloader_test,
    batch_size_attack,
    num_threads_attack,
    device,
):
    """
    Test the performance of a single white-box attack method against a PyTorch classifier.
    To be used by test_all_white_box_attack() to test multiple white-box attack methods.

    Args:
        attack_method: The white-box attack method to test.
        model: The PyTorch classifier being attacked.
        pytorch_classifier: The classifier class for the PyTorch model.
        dataloader_test: DataLoader for the test dataset.
        batch_size_attack: Batch size for generating adversarial examples.
        num_threads_attack: Number of worker threads for data loading during the attack.
        device: The device (e.g., 'cpu' or 'cuda') to use for computation.

    Returns:
        acc_advx: Accuracy of the classifier on the adversarial examples as a percentage,
        where 1 = 100% accuracy.
    """
    # Create the attack instance
    attack = attack_method(classifier=pytorch_classifier)

    # Get a batch of data from the test dataloader
    batch = next(iter(dataloader_test))
    X, y = batch

    # Generate adversarial examples
    X_advx = attack.generate(x=X.numpy())

    # Create a TensorDataset for the adversarial examples
    dataset_advx = TensorDataset(torch.Tensor(X_advx), y)

    # Create a DataLoader for the adversarial examples
    dataloader_advx = DataLoader(
        dataset_advx,
        batch_size=batch_size_attack,
        shuffle=False,
        num_workers=num_threads_attack,
    )

    # Test the model's accuracy on the adversarial examples
    acc_advx = test_accuracy(model, dataloader_advx, device)

    return acc_advx * 100  # 1.0 represents 100% accuracy


# Define a function for testing multiple white-box attacks on a given model
def test_all_white_box_attack(
    model,
    pytorch_classifier,
    dataloader_test,
    batch_size_attack,
    num_threads_attack,
    device,
):
    # List of attack methods to test
    attack_method_list = [
        pixel_attack, 
        zoo_attack,
        carlini_L0_attack,
        carlini_L2_attack,
        carlini_Linf_attack,
        deep_fool
    ]
    # Add more attack methods here as implemented

    # List to store the accuracy results for each attack method
    accuracy_list = []
    i_tmp = 0

    # Loop through each attack method
    for attack_method in attack_method_list:
        # Test the current attack method and append the accuracy result to the list
        try:
            accuracy_list.append(
                test_white_box_attack(
                    attack_method,
                    model,
                    pytorch_classifier,
                    dataloader_test,
                    batch_size_attack,
                    num_threads_attack,
                    device,
                )
            )
        except:
            print(f"Test {i_tmp} Error!")
        finally:
            i_tmp += 1

    # Return the list of accuracy results for each attack method
    return accuracy_list
