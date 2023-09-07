
import torch
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T
from torchvision import utils
import detectors
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader

from art.estimators.classification import PyTorchClassifier


class surrogateDataset(Dataset):
    def __init__(self,data,result):
        self.x = [item for item in data]
        self.y = [item for item in result]
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return len(self.x)

    

device = torch.device('cpu')
model = detectors.create_model("resnet18_cifar10", pretrained=True)
surrogate_model = detectors.create_model("resnet34_cifar10", pretrained=True)
model = model.to(device)
surrogate_model = surrogate_model.to(device)
transform_train = detectors.create_transform(model, is_training=True)
transform_test = detectors.create_transform(model)


BATCH_SIZE = 10  # Based on GPU's VRAM
NUM_THREADS = 4  # Based on # of CPU cores

# NOTE: We use `transform_test` for training set, because the model is pre-trained, we only interested in its accuracy.
dataset_train = tv.datasets.CIFAR10('./data', download=True, train=True, transform=transform_test)
dataset_test = tv.datasets.CIFAR10('./data', download=True, train=False, transform=transform_test)

# NOTE: Evaluation only. Turn shuffle off.
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREADS)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREADS)

data=[]
results=[]

with torch.no_grad():
        for batch in dataloader_train:
            x, y = batch
            x = x.to(device)
            outputs = model(x)
            data.append(x)
            results.append(outputs)

surrogate_dataset = surrogateDataset(data,results)

surrogate_train = DataLoader(surrogate_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREADS)
surrogate_test = DataLoader(surrogate_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREADS)












correct = 0
total = 0
surrogate_model.eval()
with torch.no_grad():
        for batch in surrogate_train:
            x, y = batch
            x = x.to(device)
            outputs = surrogate_model(x)
            print(outputs)
            _, predictions = torch.max(outputs, 1)
            print(predictions)
            predictions = predictions.to('cpu')
            total += y.size(0)
            correct += (predictions == y).sum().item()
accuracy = correct / total
print (accuracy)