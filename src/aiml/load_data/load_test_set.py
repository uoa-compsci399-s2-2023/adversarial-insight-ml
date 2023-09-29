"""
load_test_set.py

This script is responsible for loading the test dataset for model evaluation.
"""


import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


def load_test_set(test_set, batch_size_test):
    # need normlization
    if type(test_set) == type("a"):
        data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        test_set = ImageFolder(test_set, transform=data_transform)

    loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_test, shuffle=False
    )
    
    data = next(iter(loader))
    mean = data[0].mean()
    std = data[0].std()

    if mean > 0.1 or mean < -0.1 or std > 1.1 or std < 0.9:
        print("need normalization")

    return test_set, loader
