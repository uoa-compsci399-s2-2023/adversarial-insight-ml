"""
load_test_set.py

This script is responsible for loading the test dataset for model evaluation.
"""


import torch

from aiml.surrogate_model.utils import load_cifar10


def load_test_set(test_set, batch_size_test):
    if type(test_set) == type("a"):
        if test_set == "cifar10":
            dataset_train = load_cifar10(train=True, require_normalize=True)
        else:
            print("currently we cannot find the dataset you input")
            return None, None

    loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_test, shuffle=False
    )

    data = next(iter(loader))
    mean = data[0].mean()
    std = data[0].std()

    if mean > 0.1 or mean < -0.1 or std > 1.1 or std < 0.9:
        print("the dataset is unnormalized dataset")
    else:
        print("the dataset is normalized dataset")
    return test_set, loader


def load_train_set(test_set, batch_size_test):
    if type(test_set) == type("a"):
        if test_set == "cifar10":
            dataset_train = load_cifar10(train=False, require_normalize=True)
        else:
            print("currently we cannot find the dataset you input")
            return None, None

    loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_test, shuffle=False
    )

    data = next(iter(loader))
    mean = data[0].mean()
    std = data[0].std()

    if mean > 0.1 or mean < -0.1 or std > 1.1 or std < 0.9:
        print("the dataset is unnormalized dataset")
    else:
        print("the dataset is normalized dataset")

    return test_set, loader
