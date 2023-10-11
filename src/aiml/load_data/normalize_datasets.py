"""
normalize_datasets.py

This module contains functions for normalizing and denormalizing a dataset.
"""

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

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

    normalize_values["mean"] = mean
    normalize_values["std"] = std


def get_transforms():
    """
    Get a list of transformations for datasets, including normalization.

    Returns:
        torchvision.transforms.Compose: A composition of transformations.
    """
    transform_list = [
        T.ToTensor(),
        T.Normalize(mean=normalize_values["mean"],
                    std=normalize_values["std"]),
    ]

    return T.Compose(transform_list)


def check_normalize(dataloader):
    """
    Check if the data in a dataloader is normalized.

    Parameters:
        dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader containing the dataset.

    Returns:
        bool: True if the data is normalized (mean close to 0, std close to 1), False otherwise.
    """
    data = next(iter(dataloader))
    mean = data[0].mean()
    std = data[0].std()

    if mean > 0.1 or mean < -0.1 or std > 1.1 or std < 0.9:
        return False

    return True


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
        dataset_train.transform = get_transforms()

    dataset_test.transform = get_transforms()

    return dataset_test, dataset_train


def normalize_and_check_datasets(num_workers, batch_size_test, batch_size_train, dataset_test, dataset_train):
    """
    Normalize and check the given test and optionally, training datasets for normalization.

    Parameters:
        num_workers (int): Number of workers for data loading.
        batch_size_test (int): Batch size for the test dataset.
        batch_size_train (int): Batch size for the training dataset (if provided).
        test_dataset: The test dataset.
        train_dataset (optional): The training dataset (Default is None).

    Returns:
        Tuple: If normalization is required, returns a tuple containing the normalized test
        and training datasets along with their data loaders. If no normalization is needed,
        returns the test dataset as-is.
    """
    dataloader_test = None
    dataloader_train = None

    if dataset_train:
        get_mean_std(dataset_train)

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
        )

        dataloader_test = DataLoader(
            dataset_test,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
        )

        if not check_normalize(dataloader_test) or not check_normalize(dataloader_train):
            dataset_test_norm, dataset_train_norm = normalize_datasets(
                dataset_test, dataset_train)

            dataloader_train = DataLoader(
                dataset_test_norm,
                batch_size=batch_size_train,
                shuffle=True,
                num_workers=num_workers,
            )

            dataloader_test = DataLoader(
                dataset_train_norm,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=num_workers,
            )
    else:
        get_mean_std(dataset_test)

        dataloader_test = DataLoader(
            dataset_test,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
        )

        if not check_normalize(dataloader_test):
            dataset_test_norm, _ = normalize_datasets(dataset_test)

            dataloader_test = DataLoader(
                dataset_test_norm,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=num_workers,
            )

    return dataset_test, dataset_train, dataloader_test, dataloader_train


def denormalize(batch, device):
    """
    Denormalize a batch of normalized data using mean and standard deviation values.

    Args:
        batch (torch.Tensor): A batch of normalized data, typically in the shape
            (batch_size, channels, height, width).
        device (torch.device): The device on which to perform the denormalization.

    Returns:
        torch.Tensor: The denormalized batch of data with the same shape as the input.
    """
    if isinstance(normalize_values["mean"], list):
        mean = torch.tensor(normalize_values["mean"]).to(device)
    if isinstance(normalize_values["std"], list):
        std = torch.tensor(normalize_values["std"]).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
