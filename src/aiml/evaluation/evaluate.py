"""
evaluate.py

This module provides the evaluate function which will evaluate the model
with the given data and attack methods.
"""


import torch
from art.estimators.classification import PyTorchClassifier
from aiml.load_data.load_model import load_model
from aiml.load_data.load_test_set import load_test_set
from aiml.load_data.generate_parameter import generate_parameter
from aiml.attack.test_white_box import test_white_box_attack
from aiml.evaluation.test_accuracy import test_accuracy
from aiml.evaluation.dynamic import decide_attack
from aiml.surrogate_model.create_surrogate_model import create_surrogate_model


def evaluate(
    input_model,
    input_train_data=None,
    input_test_data=None,
    input_shape=None,
    clip_values=None,
    nb_classes=None,
    batch_size_attack=64,
    num_threads_attack=0,
    batch_size_train=64,
    batch_size_test=64,
):
    # Load model and data
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"  # Enable GPU if possible
    )
    model = load_model(input_model, device)
    if model==None:
        return None

    

    if input_test_data == None:
        print("please input test_data and try again")
        return None

    dataset_test, dataloader_test = load_test_set(input_test_data, batch_size_test)
    if dataset_test==None:
        return None
    input_shape, clip_values, nb_classes = generate_parameter(
        input_shape, clip_values, nb_classes, dataset_test, dataloader_test
    )
    if input_train_data != None:
        print("you inputted the training data, so we will try making a surrogate model to test attack")
        dataset_train, dataloader_train = load_train_set(
            input_train_data, batch_size_train
        )
        if dataset_train==None:
            return None
        
        try:
            model = create_surrogate_model(model, dataloader_train, dataloader_test)
            print("succeed in making surrogate model")

        except:
            print("sorry, we failed in making surrogate model. We will assume the attacker knows all the detail of your model to test")
            
    else:
        print("We will assume the attacker knows all the detail of your model to test")
    
    

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

    result_list = [0]
    b = True
    current_attack_n, para_n, current_attack, b, overall_mark = decide_attack(
        result_list, classifier
    )

    while b:
        result_list[0] = overall_mark
        result_list += [
            [
                current_attack_n,
                para_n,
                test_white_box_attack(
                    current_attack,
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

        current_attack_n, para_n, current_attack, b, overall_mark = decide_attack(
            result_list, classifier
        )
    print(result_list)
