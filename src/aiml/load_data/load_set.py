"""
load_set.py

This script is responsible for loading the test dataset for model evaluation.
"""


from aiml.surrogate_model.utils import load_cifar10


def load_test_set(test_set):
    """
    Load a test dataset.

    Parameters:
        dataset (dataset or string): Given a string, it will search for the target dataset.

    Returns:
        dataset: The loaded test dataset.
    """
    if type(test_set) == type("a"):
        if test_set == "cifar10":
            dataset_train = load_cifar10(train=True, require_normalize=True)
        else:
            print("Currently we cannot find the dataset you input.")
            return None

    
    return test_set


def load_train_set(test_set):
    """
    Load a training dataset.

    Parameters:
        dataset (dataset or string): Given a string, it will search for the target dataset.

    Returns:
        dataset: The loaded training dataset.
    """
    if type(test_set) == type("a"):
        if test_set == "cifar10":
            dataset_train = load_cifar10(train=False, require_normalize=True)
        else:
            print("Currently we cannot find the dataset you input.")
            return None

    return test_set
