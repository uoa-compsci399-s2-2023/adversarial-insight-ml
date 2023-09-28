"""
get_accuracy_results.py

This module provides the get_accuracy_results function which will call 
functions to apply all appropriate attacks, return their accuracy scores
in a list format.
"""


import torch
from art.estimators.classification import PyTorchClassifier
from aiml.load_data.load_model import load_model
from aiml.load_data.load_test_set import load_test_set
from aiml.load_data.generate_parameter import generate_parameter
from aiml.attack.test_white_box import *
from aiml.evaluation.test_accuracy import test_accuracy
from aiml.surrogate_model.create_surrogate_model import create_surrogate_model


def get_accuracy_results(
    input_model,
    input_train_data=None,
    input_test_data=None,
    input_shape=None,
    clip_values=None,
    nb_classes=None,
    batch_size_attack=64,
    num_threads_attack=8,
    batch_size_train=64,
    batch_size_test=64,
):
    # Load model and data
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model = load_model(input_model)

    # Create and load surrogate model
    model = create_surrogate_model(model, input_test_data)

    if input_train_data != None:
        dataset_train = load_test_set(input_train_data)
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size_train, shuffle=False
        )
    if input_test_data == None:
        print("please input test_data")
        return None

    dataset_test = load_test_set(input_test_data)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size_test, shuffle=False
    )
    input_shape, clip_values, nb_classes = generate_parameter(
        input_shape, clip_values, nb_classes, dataset_test, dataloader_test
    )

    if input_train_data != None:
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

    result_list = test_all_white_box_attack(
        model,
        classifier,
        dataloader_test,
        batch_size_attack,
        num_threads_attack,
        device,
    )
    return result_list
