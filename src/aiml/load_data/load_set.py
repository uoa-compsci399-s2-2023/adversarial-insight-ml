"""
load_set.py

This script is responsible for loading the test dataset for model evaluation.
"""


from aiml.surrogate_model.utils import load_cifar10
from datasets import list_datasets
from datasets import load_dataset
import torchvision as tv

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
            test_set = load_cifar10(train=False, require_normalize=True)
        elif train_set =="mnist":
            train_set=tv.datasets.MNIST('./data', download=True, train=False)
        elif train_set == "cifar100":
            train_set = tv.datasets.CIFAR100('./data', download=True, train=False)
        else:
            try:
                test_set = load_dataset(path=test_set, split='test')
                print("we find the test set on huggingface")
            except:
                print("Currently we cannot find the test dataset you input. you can call huggingface_hub.list_datasets to see the whole valiable set")
                return None

    
    return test_set


def load_train_set(train_set):
    """
    Load a training dataset.

    Parameters:
        dataset (dataset or string): Given a string, it will search for the target dataset.

    Returns:
        dataset: The loaded training dataset.
    """
    if type(train_set) == type("a"):
        if train_set == "cifar10":
            train_set = tv.datasets.CIFAR10('./data', download=True, train=True)
        elif train_set =="mnist":
            train_set=tv.datasets.MNIST('./data', download=True, train=True)
        elif train_set == "cifar100":
            train_set = tv.datasets.CIFAR100('./data', download=True, train=True)
        else:
            try:
                train_set = load_dataset(path=train_set, split='train')
                print("we find the train set on huggingface")
            except:
                print("Currently we cannot find the train dataset you input. you can call huggingface_hub.list_datasets to see the whole valiable set")
                return None

    
    return train_set
