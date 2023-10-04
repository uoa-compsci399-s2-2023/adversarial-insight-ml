import torch
import torchvision.transforms as T

normalize_values = {}


def get_mean_std(dataset):
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
    transform_list = [T.ToTensor(), T.Normalize(
        mean=normalize_values['mean'], std=normalize_values['std'])]

    return T.Compose(transform_list)


def normalize_dataset(dataset_train, dataset_test):
    transform_tensor = T.Compose([
        T.ToTensor(),
    ])

    dataset_train.transform = transform_tensor

    get_mean_std(dataset_train)

    dataset_train.transform = get_transforms()
    dataset_test.transform = get_transforms()

    return dataset_test, dataset_train
