import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from art.estimators.classification import PyTorchClassifier

from torch.utils.data import DataLoader
import torchvision 
from torch.utils.data import TensorDataset

def user_called_function(input_model,input_train_data=None,input_test_data=None,input_shape=None,
                         clip_values=None,nb_classes=None,batch_size_attack = 64,num_threads_attack= 8,batch_size_train = 64,batch_size_test = 64):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=load_model(input_model)            
    if input_train_data!=None:
      dataset_train=load_set(input_train_data)
      dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_train, shuffle=False)
    if input_test_data==None:
      print("please input test_data")
    
    dataset_test=load_set(input_test_data)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size_test, shuffle=False)
    
    if input_shape==None:
        (x, y) = next(iter(dataset_test))
        input_shape = np.array(x.size())
        print(f'input_shape: {input_shape}')

    if clip_values==None:
        global_min = 9999.
        global_max = 0.
        s=0
        n=0
        b=True
        for batch in dataloader_train:
            x, _ = batch
            if b==True:
                print(batch)
                b=False
            global_min = min(torch.min(x).item(), global_min)
            global_max = max(torch.max(x).item(), global_max)
        clip_values=(global_min,global_max)

        print(f'Min: {global_min}, Max: {global_max}')
    if nb_classes==None:
        list1=[]
        for i in range(len(dataset_test)):
            x,y=dataset_test[i]
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
    attack=test_attack(classifier,dataloader_test,batch_size_attack,num_threads_attack,device)
def evaluation(model, dataloader, device):
    """This function returns the accuracy of a given dataset on a pre-trained model."""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            outputs = model(x)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.to('cpu')
            total += y.size(0)
            correct += (predictions == y).sum().item()
    accuracy = correct / total
    return accuracy


def test_attack(PyTorchClassifier,dataloader_test,batch_size_attack,num_threads_attack,device):
    import importlib
    attack_list_estimator=["FastGradientMethod"]
    model_name="art.attacks.evasion"
    m=importlib.import_module(model_name)
    for attack_name in attack_list_estimator:
        c = getattr(m, attack_name)
        attack=c(estimator=PyTorchClassifier)
        batch = next(iter(dataloader_test))

        X, y = batch

        X_advx = attack.generate(x=X.numpy())
        
        dataset_advx = TensorDataset(torch.Tensor(X_advx), y)
        
        dataloader_advx = DataLoader(dataset_advx, batch_size=batch_size_attack, shuffle=False, num_workers=num_threads_attack)
        acc_advx = evaluation(model, dataloader_advx, device)

        print('{} tested, Adversarial examples accuracy: {:.2f}'.format(attack_name,acc_advx * 100))
    attack_list_classifier=["DeepFool"]
    for attack_name in attack_list_classifier:
        c = getattr(m, attack_name)
        attack=c(classifier=PyTorchClassifier)
        batch = next(iter(dataloader_test))

        X, y = batch

        X_advx = attack.generate(x=X.numpy())
        
        dataset_advx = TensorDataset(torch.Tensor(X_advx), y)
        
        dataloader_advx = DataLoader(dataset_advx, batch_size=batch_size_attack, shuffle=False, num_workers=num_threads_attack)
        acc_advx = evaluation(model, dataloader_advx, device)

        print('{} tested, Adversarial examples accuracy: {:.2f}'.format(attack_name,acc_advx * 100))

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
    dataset_train = torchvision.datasets.MNIST('./data/', train=True, download=False,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))
    dataset_test = torchvision.datasets.MNIST('./data/', train=False, download=False,
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
    test_attack_function(model,train_data=dataset_train,test_data=dataset_test)


