import torch
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T
import detectors
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

def create_surrogate_model(black_box_model):
    surrogate_model = None
    return surrogate_model

class surrogateDataset(Dataset):
    def __init__(self,data,result):
        self.x = [item for item in data]
        self.y = [item for item in result]
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return len(self.x)

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

# transform_train = T.Compose([
#      T.Pad(4),
#      T.RandomCrop(32, fill=128),
#      T.RandomHorizontalFlip(),
#      T.ToTensor(),
#      T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#  ])
#  # Only normalize the range (From 0-255 to 0-1)
# transform_test = T.Compose([
#      T.ToTensor(),
#      T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#  ])    
class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
       loss, scores, y = self._common_step(batch, batch_idx)
       self.log('train_loss', loss)
       return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y
    
    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.loss_fn(out, gt)

        self.log("val/loss", loss)

        acc = self.accuracy(out, gt)
        self.log("val/acc", acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer
    
def main():
    device = torch.device('cpu')
    model = detectors.create_model("resnet18_cifar10", pretrained=True)
    surrogate_model = detectors.create_model("resnet34_cifar10", pretrained=True)
    model = model.to(device)
    surrogate_model = surrogate_model.to(device)
    transform_train = detectors.create_transform(model, is_training=True)
    transform_test = detectors.create_transform(model)
    surrogate_transform = detectors.create_transform(surrogate_model, is_training=True)
    surrogate_transform_test = detectors.create_transform(surrogate_model)
    
    #use vgg
    BATCH_SIZE = 10  # Based on GPU's VRAM
    NUM_THREADS = 8  # Based on # of CPU cores

    # NOTE: We use `transform_test` for training set, because the model is pre-trained, we only interested in its accuracy.
    dataset_train = tv.datasets.CIFAR10('./data', download=True, train=True, transform=transform_test)
    dataset_test = tv.datasets.CIFAR10('./data', download=True, train=False, transform=transform_test)
    
    # NOTE: Evaluation only. Turn shuffle off.
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREADS)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREADS)
    
    #google collab for training
    # original_accuracy = evaluation(model,dataloader_test,device)
    # pretrain_accuracy = evaluation(surrogate_model,dataloader_test,device)
    # print("original accuracy", original_accuracy)
    # print("pretrain accuracy", pretrain_accuracy)

    #tensorboard, save loss or save performance matrix for testing how close the models outputs are 

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
    
    surrogate_model = LitModel(surrogate_model)
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(surrogate_model, surrogate_train, surrogate_test) 


    final_accuracy = evaluation(surrogate_model,dataloader_test,device)
    print("surrogate accuracy", final_accuracy)
    
if __name__ == "__main__":
    main()
