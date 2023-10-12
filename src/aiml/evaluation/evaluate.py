"""
evaluate.py

This module provides the evaluate function which will evaluate the model
with the given data and attack methods.
"""


import torch
import os
from art.estimators.classification import PyTorchClassifier

from aiml.load_data.generate_parameter import generate_parameter
from aiml.load_data.normalize_datasets import normalize_and_check_datasets, check_normalize
from aiml.attack.attack_evaluation import attack_evaluation
from aiml.evaluation.check_accuracy import check_accuracy
from aiml.evaluation.dynamic import decide_attack
from aiml.surrogate_model.create_surrogate_model import create_surrogate_model
from aiml.load_data.load_model import load_model
from aiml.load_data.load_set import load_test_set
from aiml.load_data.load_set import load_train_set


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
    num_workers=int(os.cpu_count() / 2),
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
    Evaluate the model's performance using the provided data and attack methods.

    Parameters:
        input_model (str or model): A string of the name of the machine learning model or the machine learning model itself.
        input_test_data (str or dataset): A string of the name of the testing dataset or the testing dataset itself.
        input_train_data (str or dataset, optional): A string of the name of the training dataset or the training dataset itself (default is None).
        input_shape (tuple, optional): Shape of input data (default is None).
        clip_values (tuple, optional): Range of input data values (default is None).
        nb_classes (int, optional): Number of classes in the dataset (default is None).
        batch_size_attack (int, optional): Batch size for attack testing (default is 64).
        num_threads_attack (int, optional): Number of threads for attack testing (default is 0).
        batch_size_train (int, optional): Batch size for training data (default is 64).
        batch_size_test (int, optional): Batch size for test data (default is 64).
        num_workers (int, optional): Number of workers to use for data loading 
            (default is half of the available CPU cores).
        dry (bool, optional): When True, the code should only test one example.
        attack_para_list (list, optional): List of parameter combinations for the attack.

    Returns:
        None.
    """
    # Load model and data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_model = load_model(input_model)
    input_test_data = load_test_set(input_test_data)
    model = input_model.to(device)
    input_train_data = load_train_set(input_train_data)

    dataset_test, dataset_train, dataloader_test, dataloader_train = normalize_and_check_datasets(
        num_workers, batch_size_test, batch_size_train, input_test_data, input_train_data)

    surrogate_model = None

    # Check if the user wants to create surrogate model
    if dataset_train:
        print("Creating the surrogate model. This may take a long time.")

        # Check if the testing dataset is normalized
        if not check_normalize(dataloader_train):
            raise Exception(
                "Failed to normalized training dataset. Please normalize it manually."
            )

        surrogate_model = create_surrogate_model(
            model, dataloader_train, dataloader_test
        )
        print("Surrogate model created successfully.")

        acc_train = check_accuracy(model, dataloader_train, device)
        print(f"Train accuracy: {acc_train * 100:.2f}%")

    # Check if the testing dataset is normalized
    if not check_normalize(dataloader_test):
        raise Exception(
            "Failed to normalized testing dataset. Please normalize it manually."
        )

    acc_test = check_accuracy(model, dataloader_test, device)
    print(f"Test accuracy: {acc_test * 100:.2f}%")

    input_shape, clip_values, nb_classes = generate_parameter(
        input_shape, clip_values, nb_classes, dataset_test, dataloader_test
    )

    if surrogate_model:
        model_to_use = surrogate_model
    else:
        model_to_use = model

    classifier = PyTorchClassifier(
        model=model_to_use,
        clip_values=clip_values,
        loss=None,
        optimizer=None,
        input_shape=input_shape,
        nb_classes=nb_classes,
    )

    result_list = [0]
    b = True
    current_attack_n, para_n, b, overall_mark = decide_attack(
        result_list, attack_para_list=attack_para_list
    )

    while b:
        result_list[0] = overall_mark
        result_list += [
            [
                current_attack_n,
                para_n,
                attack_evaluation(
                    current_attack_n,
                    para_n,
                    model,
                    classifier,
                    dataset_test,
                    batch_size_attack,
                    num_threads_attack,
                    device,
                    nb_classes,
                    dry=dry,
                    attack_para_list=attack_para_list,
                ),
            ]
        ]
        print(result_list)

        current_attack_n, para_n, b, overall_mark = decide_attack(
            result_list, attack_para_list=attack_para_list
        )
    print(result_list)
