"""
create_surrogate_model.py

This module creates surrogate models for black-box attacks.
"""


import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from aiml.surrogate_model.models import LogSoftmaxModule, Surrogate, create_vgg16_bn_cifar10
from aiml.surrogate_model.utils import load_cifar10


def create_surrogate_model(model, dataset):
    """Create and train a surrogate model for CIFAR-10 dataset using PyTorch Lightning."""
    NUM_WORKERS = int(os.cpu_count() / 2)
    BATCH_SIZE = 256
    MAX_EPOCHS = 50
    LEARNING_RATE = 0.0005

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if str(device) == "cuda:0":
        torch.set_float32_matmul_precision("high")

    dataset_test = load_cifar10(train=False, require_normalize=True)
    dataloader_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # The model does not output normalized outputs.
    oracle = LogSoftmaxModule(model)
    # Using the PyTorch implementation
    substitute = create_vgg16_bn_cifar10(len(dataset.classes))
    # Check the cell above. Note that without log function. The loss doesn't seem correct.
    loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)

    sub_train = load_cifar10(train=True, require_normalize=True)
    dataloader_train = DataLoader(
        sub_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    num_training_batches = len(dataloader_train)

    # NOTE: `num_training_batches` is used by LRSchedular.
    # It Cannot be loaded dynamically due to a bug in PyTorch Lightning
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
            "logs", name="surrogate_cifar10", default_hp_metric=False
        ),
        callbacks=[LearningRateMonitor(logging_interval="step")],
        # fast_dev_run=True,
    )

    trainer.fit(
        surrogate_module,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_test,
    )

    return surrogate_module
