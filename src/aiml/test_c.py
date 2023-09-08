import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from art.estimators.classification import PyTorchClassifier

from torch.utils.data import DataLoader
import torchvision 
from torch.utils.data import TensorDataset
from evaluate.evaluate import evaluate

if __name__ == '__main__':
    print("start")
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
     
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)


    model = Net()
    model.load_state_dict(torch.load("model.pth"))
    print(model)
    dataset_train = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))
    dataset_test = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))
    print(dataset_train)
    loader = DataLoader(dataset_test, batch_size=len(dataset_test), num_workers=1)
    data = next(iter(loader))
    print("the mean",data[0].mean(),"the std", data[0].std())
    
    list1=[]
    sum_x=0
    for i in range(len(dataset_test)):
        x,y=dataset_test[i]
        sum_x+=x
    print("mean",sum_x/len(dataset_test))
    evaluate(model,input_train_data=dataset_train,input_test_data=dataset_test)
