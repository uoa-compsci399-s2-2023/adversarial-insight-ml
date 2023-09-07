import torch
import numpy as np
import test_accuracy
from art.estimators.classification import PyTorchClassifier
from load_data.load_model import load_model
from load_data.load_test_set import load_test_set
from load_data.generate_parameter import generate_parameter
from test_all_white_box_attack.test_all_white_box_attack import test_all_white_box_attack
from evaluate import evaluate

def get_accuracy_results(input_model,input_train_data=None,input_test_data=None,input_shape=None,
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
    input_shape,clip_values,nb_classes=generate_parameter(input_shape,clip_values,nb_classes)
    
        
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

    result_list = test_all_white_box_attack(classifier,dataloader_test,batch_size_attack,num_threads_attack,device)
    evaluate(result_list)







