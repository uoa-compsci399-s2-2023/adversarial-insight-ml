import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from art.estimators.classification import PyTorchClassifier

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def user_called_function(input_model,input_train_data=None,input_test_data=None,input_shape=None,
                         clip_values=None,nb_classes=None,batch_size_attack = 64,num_threads_attack= 8,batch_size_train = 64,batch_size_test = 64):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(input_model)            
    if input_train_data != None:
      dataset_train = load_set(input_train_data)
      dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_train, shuffle=False)
    if input_test_data == None:
      print("please input test_data")
    
    dataset_test = load_set(input_test_data)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size_test, shuffle=False)
    
    if input_shape == None:
        (x, y) = next(iter(dataset_test))
        input_shape = np.array(x.size())
        print(f'input_shape: {input_shape}')

    if clip_values == None:
        global_min = 9999.
        global_max = 0.
        s = 0
        n = 0
        b = True
        for batch in dataloader_train:
            x, _ = batch
            if b == True:
                print(batch)
                b = False
            global_min = min(torch.min(x).item(), global_min)
            global_max = max(torch.max(x).item(), global_max)
        clip_values = (global_min,global_max)

        print(f'Min: {global_min}, Max: {global_max}')
    if nb_classes == None:
        list1 = []
        for i in range(len(dataset_test)):
            x,y = dataset_test[i]
            if y not in list1:
                list1+=[y]
        nb_classes=len(dataset_train)
    if input_train_data!=None:
      acc_train = test_accuracy(model, dataloader_train, device)
      print(f'Train accuracy: {acc_train * 100:.2f}')
    acc_test = test_accuracy(model, dataloader_test, device)
    print(f'Test accuracy:  {acc_test * 100:.2f}')
    classifier = PyTorchClassifier(
        model=model,
        clip_values=clip_values, 
        loss=None,
        optimizer=None,
        input_shape=input_shape,
        nb_classes=nb_classes,
    )
    result_list = test_all_attack(classifier,dataloader_test,batch_size_attack,num_threads_attack,device)
    evaluate(result_list)







