"""
test_white_box.py
"""
from test_accuracy.test_accuracy import *
import torch
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader 
from torch.utils.data import TensorDataset
from standard_white_box_attack.standard_white_box_test import*
def test_white_box_attack(attack_method,model,PyTorchClassifier,dataloader_test,batch_size_attack,num_threads_attack,device):
    attack=attack_method(classifier=PyTorchClassifier)
    batch = next(iter(dataloader_test))
    X, y = batch
    X_advx = attack.generate(x=X.numpy())
    dataset_advx = TensorDataset(torch.Tensor(X_advx), y)
    dataloader_advx = DataLoader(dataset_advx, batch_size=batch_size_attack, shuffle=False, num_workers=num_threads_attack)
    acc_advx = test_accuracy(model, dataloader_advx, device)
    return (acc_advx * 100)
    
def test_all_white_box_attack(model,PyTorchClassifier,dataloader_test,batch_size_attack,num_threads_attack,device):
    attack_method_list = [carlini_L0_method]
    accuracy_list=[]
    for attack_method in attack_method_list:
        accuracy_list+=[test_white_box_attack(attack_method,model,PyTorchClassifier,dataloader_test,batch_size_attack,num_threads_attack,device)]
    return accuracy_list
