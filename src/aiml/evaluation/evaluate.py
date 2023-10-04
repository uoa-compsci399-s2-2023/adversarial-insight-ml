"""
evaluate.py

This module provides the evaluate function which will evaluate the model
with the given data and attack methods.
"""


import torch
import os
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier

from aiml.load_data.generate_parameter import generate_parameter
from aiml.load_data.normalize_dataset import normalize_dataset
from aiml.attack.test_attack import test_attack
from aiml.evaluation.test_accuracy import test_accuracy
from aiml.evaluation.dynamic import decide_attack
from aiml.surrogate_model.create_surrogate_model import create_surrogate_model


def evaluate(
    input_model,
    input_test_data,
    input_train_data=None,
    input_shape=None,
    clip_values=None,
    nb_classes=None,
    batch_size_attack=64,
    num_threads_attack=0,
    batch_size_train=64,
    batch_size_test=64,
    num_workers=int(os.cpu_count() / 2)
):
    """
    Evaluate the model's performance using the provided data and attack methods.

    Parameters:
        input_model (str): Path to the trained machine learning model.
        input_train_data (str, optional): Path to the training data (default is None).
        input_test_data (str): Path to the test data.
        input_shape (tuple, optional): Shape of input data (default is None).
        clip_values (tuple, optional): Range of input data values (default is None).
        nb_classes (int, optional): Number of classes in the dataset (default is None).
        batch_size_attack (int, optional): Batch size for attack testing (default is 64).
        num_threads_attack (int, optional): Number of threads for attack testing (default is 0).
        batch_size_train (int, optional): Batch size for training data (default is 64).
        batch_size_test (int, optional): Batch size for test data (default is 64).

    Returns:
        None.
    """
    # Load model and data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = input_model.to(device)

    if input_train_data:
        print("Including a training dataset will create a surrogate model. This may take a long time.")
        user_response = input(
            "Do you want to proceed? (Yes/No): ").strip().lower()

        responded = False
        while not responded:
            if user_response in ["y", "yes", ""]:
                responded = True
                print("Creating the surrogate model...")

                try:
                    dataset_train, dataset_test = normalize_dataset(
                        input_train_data, input_test_data)

                    dataloader_train = DataLoader(
                        dataset_train, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)

                    dataloader_test = DataLoader(
                        dataset_train, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)

                    model = create_surrogate_model(
                        model, dataloader_train, dataloader_test)
                    print("Surrogate model created successfully.")

                except:
                    raise Exception("Failed to create surrogate model.")

            elif user_response in ["n", "no"]:
                raise Exception("Evaluation aborted.")

            else:
                user_response = input(
                    "Invalid Input. Please enter Yes or No: ").strip().lower()

    input_shape, clip_values, nb_classes = generate_parameter(
        input_shape, clip_values, nb_classes, dataset_test, dataloader_test
    )

    if input_train_data:
        acc_train = test_accuracy(model, dataloader_train, device)
        print(f"Train accuracy: {acc_train * 100:.2f}")

    acc_test = test_accuracy(model, dataloader_test, device)
    print(f"Test accuracy:  {acc_test * 100:.2f}")

    classifier = PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        loss=None,
        optimizer=None,
        input_shape=input_shape,
        nb_classes=nb_classes,
    )

    result_list = [0]
    b = True
    current_attack_n, para_n, b, overall_mark = decide_attack(
        result_list,
    )

    while b:
        result_list[0] = overall_mark
        result_list += [
            [
                current_attack_n,
                para_n,
                test_attack(
                    current_attack_n,
                    para_n,
                    model,
                    classifier,
                    dataset_test,
                    batch_size_attack,
                    num_threads_attack,
                    device,
                    nb_classes,
                ),
            ]
        ]
        print(result_list)

        current_attack_n, para_n, b, overall_mark = decide_attack(
            result_list,
        )
    print(result_list)
