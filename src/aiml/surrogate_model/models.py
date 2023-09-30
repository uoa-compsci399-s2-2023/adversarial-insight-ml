"""
models.py

This module contains utility functions and PyTorch Lightning modules for
working with the CIFAR-10 dataset. The VGG16 BN model is used as a
substitute for the black box model. This functions and classes in this
file are used in the "create_surrogate_model.py" file.
"""


from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from torchmetrics import Accuracy


def create_cifar10_model(num_classes=10):
    """VGG16 model for CIFAR10"""
    model = tv.models.vgg16(weights=None, num_classes=num_classes)
    model.features[0] = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features[4] = nn.Identity()
    return model


def create_cifar100_model(num_classes=100):
    """VGG16 model for CIFAR-100"""
    model = tv.models.vgg16(weights=None, num_classes=num_classes)
    model.features[0] = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features[4] = nn.Identity()
    return model


class Surrogate(pl.LightningModule):
    """A PyTorch Lightning module representing a surrogate model.

    This surrogate model is designed to mimic the behavior of an oracle model.

    Attributes:
        oracle (nn.Module): The oracle model for reference.
        substitute (nn.Module): The surrogate model to be trained.
        loss_fn (Callable): The loss function for surrogate training.
        accuracy (Accuracy): A metric for computing accuracy during training/validation.
    """

    def __init__(self, lr, num_training_batches, oracle, substitute, loss_fn, num_classes, softmax=True):
        super().__init__()
        self.save_hyperparameters(ignore=["oracle", "substitute", "loss_fn"])

        self.oracle = oracle
        # Oracle's parameters should not alter.
        for param in self.oracle.parameters():
            param.requires_grad = False
        self.substitute = substitute
        self.loss_fn = loss_fn
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.substitute(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch  # Ignore true labels
        out = self(x)
        if self.hparams.softmax:
            out = F.log_softmax(out, 1)
        self.oracle.eval()
        with torch.no_grad():
            out_oracle = self.oracle(x)
        loss = self.loss_fn(out, out_oracle)
        self.log(f"train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        if self.hparams.softmax:
            out = F.log_softmax(out, 1)
        out_oracle = self.oracle(x)
        loss = self.loss_fn(out, out_oracle)
        pred_oracle = out_oracle.argmax(1)
        acc = self.accuracy(out, y)
        match_oracle = self.accuracy(out, pred_oracle)
        self.log(f"val_loss", loss)
        self.log(f"val_acc", acc)
        self.log(f"val_match", match_oracle)

    def predict_step(
        self, batch, batch_idx, dataloader_idx=0
    ):
        x, _ = batch
        out = self(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.substitute.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler_dict = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.hparams.num_training_batches,
                max_lr=0.1,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
