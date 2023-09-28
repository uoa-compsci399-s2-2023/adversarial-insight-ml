"""
utils.py

This module contains various utility functions and configurations for 
working with the CIFAR-10 dataset and PyTorch Lightning-based training 
for creating and training a surrogate model. This file supports the 
"create_surrogate_model.py" file. 
"""


import torchvision as tv
import torchvision.transforms as T


cifar10_normalize_values = {
    "mean": [0.4914, 0.4822, 0.4465],
    "std": [0.2470, 0.2435, 0.2616],
}


def get_transforms(train=True, require_normalize=False):
    """Get data transformations for CIFAR-10 dataset."""
    state = "train" if train else "val"
    data_transforms = {
        "train": [
            T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ],
        "val": [
            T.ToTensor(),
        ],
    }
    transform_list = data_transforms[state]
    if require_normalize:
        transform_list = data_transforms[state] + [
            T.Normalize(
                mean=cifar10_normalize_values["mean"],
                std=cifar10_normalize_values["std"],
            )
        ]

    return T.Compose(transform_list)


def load_cifar10(train=True, require_normalize=False):
    """Return CIFAR10 dataset."""
    dataset = tv.datasets.CIFAR10(
        "./data",
        download=True,
        train=train,
        transform=get_transforms(train, require_normalize),
    )
    return dataset
