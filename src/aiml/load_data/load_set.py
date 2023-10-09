"""
load_test_set.py

This script is responsible for loading the test dataset for model evaluation.
"""


import torch

from aiml.surrogate_model.utils import load_cifar10
"""
The function is for loading test data set
args:
dataset(dataset/string):if input is string, it will find the target dataset

return 
dataset,

"""

def load_test_set(test_set):
    if type(test_set) == type("a"):
        if test_set == "cifar10":
            dataset_train = load_cifar10(train=True, require_normalize=True)
        else:
            print("currently we cannot find the dataset you input")
            return None

    
    return test_set
"""
The function is for loading train data set
args:
dataset(dataset/string):if input is string, it will find the target dataset

return 
dataset,

"""

def load_train_set(test_set):
    if type(test_set) == type("a"):
        if test_set == "cifar10":
            dataset_train = load_cifar10(train=False, require_normalize=True)
        else:
            print("currently we cannot find the dataset you input")
            return None

  
    

    

    return test_set
