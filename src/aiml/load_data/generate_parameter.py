import torch
import numpy as np
import time

def generate_parameter(input_shape,clip_values,nb_classes,dataset_test,dataloader_test):
    if input_shape == None:
        (x, y) = next(iter(dataset_test))
        input_shape = np.array(x.size()).tolist()
        print(f'input_shape: {input_shape}')

    if clip_values == None:
        global_min = np.inf
        global_max = (-1)*np.inf
        s = 0
        n = 0
        b = True
        for batch in dataloader_test:
            x, _ = batch
            if b == True:
                print(batch)
                b = False
            global_min = min(torch.min(x).item(), global_min)
            global_max = max(torch.max(x).item(), global_max)
        clip_values = (global_min,global_max)

        print(f'Min: {global_min}, Max: {global_max}')

    if nb_classes == None:
        list1 = set([y for _,y in dataset_test]) # use set to get unique classes
        nb_classes=len(list1)
    return (input_shape,clip_values,nb_classes)
