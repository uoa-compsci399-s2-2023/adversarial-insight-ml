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
from aiml.load_data.normalize_datasets import normalize_datasets
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
        input_model (model): The machine learning model to be evaluated.
        input_test_data (dataset): A dataset containing testing data.
        input_train_data (dataset, optional): A dataset containing training data (default is None).
        input_shape (tuple, optional): Shape of input data (default is None).
        clip_values (tuple, optional): Range of input data values (default is None).
        nb_classes (int, optional): Number of classes in the dataset (default is None).
        batch_size_attack (int, optional): Batch size for attack testing (default is 64).
        num_threads_attack (int, optional): Number of threads for attack testing (default is 0).
        batch_size_train (int, optional): Batch size for training data (default is 64).
        batch_size_test (int, optional): Batch size for test data (default is 64).
        num_workers (int, optional): Number of workers to use for data loading 
            (default is half of the available CPU cores).

    Returns:
        None.
    """
    # Load model and data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_model = load_model(input_model)
    input_test_data = load_test_set(input_test_data)
    model = input_model.to(device)
    input_train_data = load_train_set(input_train_data)
    dataset_test, dataset_train = normalize_datasets(
        input_test_data, input_train_data)

    dataloader_test = DataLoader(
        dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=num_workers
    )

    # Check if the user wants to create surrogate model
    if input_train_data:
        print("Creating the surrogate model. This may take a long time.")

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
        )

        surrogate_model = create_surrogate_model(
            model, dataloader_train, dataloader_test
        )
        print("Surrogate model created successfully.")

        acc_train = check_accuracy(model, dataloader_train, device)
        print(f"Train accuracy: {acc_train * 100:.2f}%")

    # Check if the testing dataset is normalized
    data = next(iter(dataloader_test))
    mean = data[0].mean()
    std = data[0].std()

    if mean > 0.1 or mean < -0.1 or std > 1.1 or std < 0.9:
        raise Exception(
            "Failed to normalized testing dataset. Please manually normalize it."
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
