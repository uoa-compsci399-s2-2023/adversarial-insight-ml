"""
attack_evaluation.py

This module contains a function attack_evaluation that use the inputted 
attack method and parameter, generate adversarial images by changing the
given images a little using adversarial attack. Then output the images 
into the "img" folder and return accuracy.
"""


import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision.transforms as T
import numpy as np

from aiml.load_data.normalize_datasets import denormalize
from aiml.evaluation.check_accuracy import check_accuracy_with_flags
from aiml.attack.adversarial_attacks import (
    auto_projected_cross_entropy,
    auto_projected_difference_logits_ratio,
    carlini_L0_attack,
    carlini_L2_attack,
    carlini_Linf_attack,
    deep_fool_attack,
    pixel_attack,
    square_attack,
    zoo_attack,
)


def attack_evaluation(
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
    dry=False,
    attack_para_list=[
        [[1], [16], [32]],
        [[1], [16], [32]],
        [[1], [16], [32]],
        [[1], [16], [32]],
        [[1], [16], [32]],
        [[1], [16], [32]],
        [[50], [100], [150]],
        [[1], [16], [32]],
        [[1], [16], [32]],
    ],
):
    """
    Check the performance of adversarial attack methods against the ML model.

    Parameters:
        attack_n (int): Attack number (0 to 7).
        para_n (int): Parameter number for selecting a combination of attack parameters.
        model (MLModel): The machine learning model.
        classifier (PytorchClassifier): The PyTorch classifier defined using the ART library.
        dataset: The dataset to be modified with adversarial attacks.
        batch_size_attack (int): Parameter for adversarial images data loader.
        num_threads_attack (int): Parameter for adversarial images data loader.
        device (str): "cpu" or "gpu".
        nb_classes (int): The number of possible labels.
        require_n (int): For every label, how many images marked as this label will be modified to 
            get adversarial images.
        dry (bool): When True, the code only tests one example.
        attack_para_list (list): List of parameter combinations for the attack.

    Returns:
        float: Accuracy of the classifier on the adversarial examples as a percentage (1 = 100%).
    """

    attack_method_list = [
        [
            0,
            auto_projected_cross_entropy,
            attack_para_list[0],
            "auto_projected_cross_entropy",
            ["batch", "eps", "eps_step"],
        ],
        [
            1,
            auto_projected_difference_logits_ratio,
            attack_para_list[1],
            "auto_projected_difference_logits_ratio",
            ["batch", "eps", "eps_step"],
        ],
        [
            2,
            carlini_L0_attack,
            attack_para_list[2],
            "carlini_L0_attack",
            [
                "batch",
                "learning_rate",
                "binary_search_steps",
                "max_iter",
            ],
        ],
        [
            3,
            carlini_L2_attack,
            attack_para_list[3],
            "carlini_L2_attack",
            [
                "batch",
                "learning_rate",
                "binary_search_steps",
                "max_iter",
            ],
        ],
        [
            4,
            carlini_Linf_attack,
            attack_para_list[4],
            "carlini_Linf_attack",
            [
                "batch",
                "learning_rate",
                "max_iter",
            ],
        ],
        [
            5,
            deep_fool_attack,
            attack_para_list[5],
            "deep_fool_attack",
            ["batch", "max_iter"],
        ],
        [
            6,
            pixel_attack,
            attack_para_list[6],
            "pixel_attack",
            ["max_iter"],
        ],
        [
            7,
            square_attack,
            attack_para_list[7],
            "square_attack",
            ["batch", "max_iter"],
        ],
        [
            8,
            zoo_attack,
            attack_para_list[8],
            "zoo_attack",
            [
                "batch",
                "learning_rate",
                "max_iter",
                "binary_search_steps",
            ],
        ],
    ]

    """
    attack_method_list contains all eight adversarial attack methods used.

    Each entry in the list is a sublist that represents an attack method:
    - The first element is the attack number.
    - The second element is the attack function.
    - The third element is a list of parameter combinations.
    - The fourth element is the name of the attack.
    - The fifth element is the parameter name for every combination of parameters.

    For example, for the 'auto_projected_cross_entropy' attack method:
    - The attack number is 0.
    - The attack function is 'auto_projected_cross_entropy'.
    - There are three possible parameter choices: batch=16, batch=20, or batch=32.
    """

    para = attack_method_list[attack_n][2][para_n]  # Get parameter

    """
    Generate attack object with attack function. The parameter of the attack 
    function is pytorch classifer and given parameter.
    """

    if len(para) == 1:
        attack = attack_method_list[attack_n][1](classifer, para[0])
    elif len(para) == 2:
        attack = attack_method_list[attack_n][1](classifer, para[0], para[1])
    elif len(para) == 3:
        attack = attack_method_list[attack_n][1](
            classifer, para[0], para[1], para[2])
    else:
        attack = attack_method_list[attack_n][1](
            classifer, para[0], para[1], para[2], para[3]
        )
    X = []  # Store the tensors of images
    y = []  # Store the corresponding labels of images

    require_y = [
        require_n
    ] * nb_classes  # Create a list to record how many images are needed for each label

    i = 0
    enough = False
    while not enough:
        # Stop the loop when all elements in require_y are zero or all dataset images are processed
        a, b = dataset[i]
        i += 1
        if i >= len(dataset) - 1:
            enough = (
                True  # All images in dataset are looked through then loop will stop
            )
        if require_y[b] <= 0:
            continue
        try:
            outputs = model(
                a
            )  # Test whether the original image can be correctly recognized by the ML model
            _, predictions = torch.max(outputs, 1)

            if b != predictions.numpy()[0]:
                continue  # Skip unrecognizable original images for adversarial generation
        except:
            print("processing")

        X += [a.numpy()]
        y += [b]
        require_y[b] = require_y[b] - 1
        # add the image for generating adversarial image further
        all_zero = True
        for requ_n in require_y:  # check whether the required images are enough
            if requ_n > 0:
                all_zero = False
                break
        if all_zero:
            enough = True
        if dry:
            enough = True

    X = np.array(X)
    y = np.array(y)

    y = torch.from_numpy(y)

    # Generate adversarial examples
    X_advx = attack.generate(X)

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
    acc_advx, correct_advx = check_accuracy_with_flags(
        model, dataloader_advx, device)

    for i in range(len(correct_advx[0])):  # put images in the folder
        if correct_advx[0][i] == False:
            transform = T.ToPILImage()
            X_tensor[i] = denormalize(X_tensor[i])
            X_advx_tensor[i] = denormalize(X_advx_tensor[i])
            orig_img = transform(X_tensor[i])
            advx_img = transform(X_advx_tensor[i])
            img_path = (
                f"./img/{attack_method_list[attack_n][3]}/{para_n}/{y[i]}/fail"
            )

            if not os.path.exists(img_path):
                os.makedirs(img_path)

            orig_img.save(f"{img_path}/{i}orignial.png", "PNG")
            advx_img.save(f"{img_path}/{i}advers.png", "PNG")
        else:
            transform = T.ToPILImage()
            X_tensor[i] = denormalize(X_tensor[i])
            X_advx_tensor[i] = denormalize(X_advx_tensor[i])
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
