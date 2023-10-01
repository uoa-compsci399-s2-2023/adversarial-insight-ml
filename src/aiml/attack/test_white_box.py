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

from aiml.evaluation.test_accuracy import test_accuracy
from aiml.attack.adversarial_attacks import *


def test_white_box_attack(
    attack_n,
    para_n,
    model,
    classifer,
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
    attack_method_list = [
        [
            0,
            auto_projected_cross_entropy,
            [[16], [20], [32]],
            "auto_projected_cross_entropy",
            ["batch"],
        ],
        [
            1,
            auto_projected_difference_logits_ratio,
            [[1], [16], [32]],
            "auto_projected_difference_logits_ratio",
            ["batch"],
        ],
        [2, carlini_L0_attack, [[1], [16], [32]], "carlini_L0_attack", ["batch"]],
        [3, carlini_L2_attack, [[1], [16], [32]], "carlini_L2_attack", ["batch"]],
        [4, carlini_Linf_attack, [[1], [16], [32]], "carlini_Linf_attack", ["batch"]],
        [5, deep_fool_attack, [[1], [16], [32]], "deep_fool_attack", ["batch"]],
        [6, pixel_attack, [[None]], "pixel_attack", ["th"]],
        [7, square_attack, [[1], [16], [32]], "square_attack", ["batch"]],
        [8, zoo_attack, [[1], [16], [32]], "zoo_attack", ["batch"]],
    ]
    para = attack_method_list[attack_n][2][para_n]
    if len(para) == 1:
        attack = attack_method_list[attack_n][1](classifer, para[0])
    elif len(para) == 2:
        attack = attack_method_list[attack_n][1](classifer, para[0], para[1])
    elif len(para) == 3:
        attack = attack_method_list[attack_n][1](classifer, para[0], para[1], para[2])
    else:
        attack = attack_method_list[attack_n][1](
            classifer, para[0], para[1], para[2], para[3]
        )
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
    acc_advx, correct_advx = test_accuracy(model, dataloader_advx, device)
    print(correct_advx, len(correct_advx))

    # generate image
    print(len(correct_advx))
    for i in range(len(correct_advx[0])):
        if correct_advx[0][i] == False:
            transform = T.ToPILImage()
            orig_img = transform(X_tensor[i])
            advx_img = transform(X_advx_tensor[i])
            img_path = f"./img/{attack_method_list[attack_n][3]}/{para_n}/{y[i]}/fail"
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            orig_img.save(f"{img_path}/{i}orignial.png", "PNG")
            advx_img.save(f"{img_path}/{i}advers.png", "PNG")
        else:
            transform = T.ToPILImage()
            orig_img = transform(X_tensor[i])
            advx_img = transform(X_advx_tensor[i])
            img_path = (
                f"./img/{attack_method_list[attack_n][3]}/{para_n}/{y[i]}/succeed"
            )
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            orig_img.save(f"{img_path}/{i}orignial.png", "PNG")
            advx_img.save(f"{img_path}/{i}advers.png", "PNG")

    return acc_advx * 100  # 1.0 represents 100% accuracy
