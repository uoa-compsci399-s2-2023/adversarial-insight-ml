"""
normalize_dataset.py

This module provides functions for calculating the mean and standard deviation of image channels 
in a dataset and normalizing the training and testing datasets using the calculated values.
"""

import torch
import torchvision.transforms as T

normalize_values = {}  # contains the mean and std for a dataset


def get_mean_std(dataset):
    """Get mean and std from a dataset"""
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


def normalize_datasets(dataset_train, dataset_test):
    """Normalize the training and testing datasets"""
    transform_tensor = T.Compose([
        T.ToTensor(),
    ])

    dataset_train.transform = transform_tensor

    get_mean_std(dataset_train)

    transform_list = [T.ToTensor(), T.Normalize(
        mean=normalize_values['mean'], std=normalize_values['std'])]

    dataset_train.transform = transform_list
    dataset_test.transform = transform_list

    return dataset_train, dataset_test
