"""
create_surrogate_model.py

This module creates surrogate models for black-box attacks.
"""

import os

import pytorch_lightning as pl
import detectors
import timm
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import LogSoftmaxModule, Surrogate, create_vgg16_bn_cifar10
from utils import (choose_dataset, cifar10_normalize_values,
                   evaluate_dataloader, inverse_normalize, load_cifar10)


def evaluate_model(model: nn.Module):
    """Evaluate a PyTorch model using various methods including Softmax and LogSoftmax."""

    # Testing the model. This model does not run no Softmax function.
    x = torch.rand([10, 3, 32, 32])
    out = model(x)
    print(out[0])
    # Without SoftMax
    print(out.sum(1))

    # With SoftMax
    out = nn.functional.softmax(out, 1)
    print(out.sum(1))

    loss_fn = nn.KLDivLoss(reduction='batchmean')
    print('[softmax] Random:', loss_fn(out[:5], out[5:]))
    print('[softmax] Match:', loss_fn(out[:5], out[:5]))

    # With LogSoftMax
    out = nn.functional.log_softmax(out, 1)
    print(out.sum(1))

    loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=True)
    print('[log softmax] Random:', loss_fn(out[:5], out[5:]))
    print('[log softmax] Match:', loss_fn(out[:5], out[:5]))


def create_surrogate_model(model: nn.Module) -> Surrogate:
    """Create and train a surrogate model for CIFAR-10 dataset using PyTorch Lightning."""
    NUM_WORKERS = int(os.cpu_count() / 2)
    BATCH_SIZE = 256
    MAX_EPOCHS = 50
    LEARNING_RATE = 0.0005

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    if str(device) == 'cuda:0':
        torch.set_float32_matmul_precision('high')
        print('Set precision to High.')

    evaluate_model(model)

    dataset_test = load_cifar10(train=False, require_normalize=True)
    dataloader_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    acc_test = evaluate_dataloader(model, dataloader_test)
    print(f'[Input Model] Test accuracy: {acc_test*100:.2f}')

    # The model does not output normalized outputs.
    oracle = LogSoftmaxModule(model)
    substitute = create_vgg16_bn_cifar10()  # Using the PyTorch implementation
    # Check the cell above. Note that without log function. The loss doesn't seem correct.
    loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=True)

    sub_train = load_cifar10(train=True, require_normalize=True)
    dataloader_train = DataLoader(
        sub_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    num_training_batches = len(dataloader_train)
    print("num_training_batches", num_training_batches)

    # NOTE: `num_training_batches` is used by LRSchedular. Cannot be loaded dynamically due to a bug in PyTorch Lightning
    surrogate_module = Surrogate(
        lr=LEARNING_RATE,
        num_training_batches=num_training_batches,
        oracle=oracle,
        substitute=substitute,
        loss_fn=loss_fn,
        softmax=True,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        enable_progress_bar=True,
        logger=TensorBoardLogger(
            "logs", name="surrogate_cifar10", default_hp_metric=False),
        callbacks=[LearningRateMonitor(logging_interval='step')],
        # fast_dev_run=True,
    )
    trainer.fit(
        surrogate_module,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_test,
    )

    # Trained on original training data
    acc_test = evaluate_dataloader(surrogate_module, dataloader_test)
    print(f'[Surrogate Model] Test accuracy: {acc_test*100:.2f}')

    return surrogate_module


# For testing purposes
if __name__ == '__main__':
    model_to_test = timm.create_model("resnet18_cifar10", pretrained=True)
    create_surrogate_model(model_to_test)
