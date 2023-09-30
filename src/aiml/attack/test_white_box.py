"""
test_white_box.py

This module contains functions for testing white-box attack methods on a
PyTorch classifier model, and returns respecitve accuracies in a list.
"""

import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision.transforms as T

from aiml.evaluation.test_accuracy import test_accuracy2
from aiml.attack.adversarial_attacks import *


def test_white_box_attack(
    attack_method,
    model,
    dataset,
    batch_size_attack,
    num_threads_attack,
    device,
    nb_classes,
    require_n=3,
):
    """
    Test the performance of a single white-box attack method against a PyTorch classifier.
    To be used by test_all_white_box_attack() to test multiple white-box attack methods.

    Returns:
        acc_advx: Accuracy of the classifier on the adversarial examples as a percentage,
        where 1 = 100% accuracy.
    """

    # Create the attack instance
    attack = attack_method
    X = []
    y = []

    require_y = [require_n] * nb_classes
    enough = False
    i = 0
    while not enough:
        a, b = dataset[i]
        i += 1
        if i >= len(dataset) - 1:
            enough = True
        if require_y[b] <= 0:
            continue
        outputs = model(a)
        _, predictions = torch.max(outputs, 1)

        if b != predictions.numpy()[0]:
            continue

        X += [a.numpy()]
        y += [b]
        require_y[b] = require_y[b] - 1

        all_zero = True
        for requ_n in require_y:
            if requ_n > 0:
                all_zero = False
                break
        if all_zero:
            enough = True

        

    X = np.array(X)
    y = np.array(y)

    y = torch.from_numpy(y)
    # Generate adversarial examples
    X_advx = attack.generate(X)
    print("ori", len(X), type(X))
    print("adv", len(X_advx), type(X))
    X_tensor = torch.Tensor(X)
    X_advx_tensor = torch.Tensor(X_advx)
    # Create a TensorDataset for the adversarial examples
    dataset_advx = TensorDataset(torch.Tensor(X_advx), y)

    # Create a DataLoader for the adversarial examples
    dataloader_advx = DataLoader(
        dataset_advx,
        batch_size=batch_size_attack,
        shuffle=False,
        num_workers=num_threads_attack,
    )  # high num_workers may cause err

    # Test the model's accuracy on the adversarial examples
    # acc_advx = test_accuracy(model, dataloader_advx, device)
    acc_advx, correct_advx = test_accuracy2(model, dataloader_advx, device)
    print(correct_advx, len(correct_advx))
    # generate image
    print(len(correct_advx))
    for i in range(len(correct_advx[0])):
        if correct_advx[0][i] == False:
            transform = T.ToPILImage()
            orig_img = transform(X_tensor[i])
            advx_img = transform(X_advx_tensor[i])
            img_path = f"./img"
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            orig_img.save(f"{img_path}/fail_label{y[i]}_original_{i}.png", "PNG")
            advx_img.save(f"{img_path}/fail_label{y[i]}_advers_{i}.png", "PNG")
        else:
            transform = T.ToPILImage()
            orig_img = transform(X_tensor[i])
            advx_img = transform(X_advx_tensor[i])
            img_path = f"./img"
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            orig_img.save(f"{img_path}/succeed_label{y[i]}_original_{i}.png", "PNG")
            advx_img.save(f"{img_path}/succeed_label{y[i]}_advers_{i}.png", "PNG")

    return acc_advx * 100  # 1.0 represents 100% accuracy
