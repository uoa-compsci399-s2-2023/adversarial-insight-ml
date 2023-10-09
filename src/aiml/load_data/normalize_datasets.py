"""
normalize_datasets.py

This module provides functions for calculating the mean and standard deviation of image channels 
in a dataset and normalizing the training and testing datasets using the calculated values.
"""

import torch
import torchvision.transforms as T

normalize_values = {}


def get_mean_std(dataset):
    """
    Get the mean and standard deviation of the dataset's image channels.

    Parameters:
        dataset (dataset): The dataset containing images.

    Returns:
        None.
    """
    imgs = [item[0] for item in dataset]
    imgs = torch.stack(imgs, dim=0).numpy()
    num_channels = imgs[0].shape[0]

    if num_channels == 3:
        mean_r = imgs[:, 0, :, :].mean()
        mean_g = imgs[:, 1, :, :].mean()
        mean_b = imgs[:, 2, :, :].mean()

        mean = [mean_r, mean_g, mean_b]

        std_r = imgs[:, 0, :, :].std()
        std_g = imgs[:, 1, :, :].std()
        std_b = imgs[:, 2, :, :].std()

        std = [std_r, std_g, std_b]
    else:
        mean = [imgs[:, 0, :, :].mean()]
        std = [imgs[:, 0, :, :].std()]

    normalize_values['mean'] = mean
    normalize_values['std'] = std


def get_transforms():
    """
    Get a list of transformations for datasets, including normalization.

    Returns:
        torchvision.transforms.Compose: A composition of transformations.
    """
    transform_list = [T.ToTensor(), T.Normalize(
        mean=normalize_values['mean'], std=normalize_values['std'])]

    return T.Compose(transform_list)


def transform_dataset_to_tensor(dataset):
    """
    Transform the given dataset to a tensor format.

    Parameters:
        dataset (dataset): The dataset to be transformed.

    Returns:
        dataset: The transformed dataset with tensors.
    """
    transform_tensor = T.Compose([
        T.ToTensor(),
    ])

    dataset.transform = transform_tensor

    return dataset


def normalize_datasets(dataset_test, dataset_train=None):
    """
    Normalize the training and testing datasets.

    Parameters:
        dataset_test (dataset): The testing dataset.
        dataset_train (dataset, optional): The training dataset (Default is None).

    Returns:
        tuple: A tuple containing the normalized testing dataset and, if provided, the normalized training dataset.
    """
    if dataset_train:
        dataset_find_mean_std = transform_dataset_to_tensor(dataset_train)
    else:
        dataset_find_mean_std = transform_dataset_to_tensor(dataset_test)

    get_mean_std(dataset_find_mean_std)

    if dataset_train:
        dataset_train.transform = get_transforms()

    dataset_test.transform = get_transforms()

    return dataset_test, dataset_train
