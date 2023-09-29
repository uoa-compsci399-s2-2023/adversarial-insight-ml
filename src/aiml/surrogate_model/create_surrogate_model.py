"""
create_surrogate_model.py

This module creates surrogate models for black-box attacks.
"""


import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from aiml.surrogate_model.models import (
    LogSoftmaxModule,
    Surrogate,
    create_vgg16_bn_cifar10,
)


def create_surrogate_model(
    model: nn.Module, dataloader_train, dataloader_test
) -> Surrogate:
    """Create and train a surrogate model for CIFAR-10 dataset using PyTorch Lightning."""
    MAX_EPOCHS = 50
    LEARNING_RATE = 0.0005

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if str(device) == "cuda:0":
        torch.set_float32_matmul_precision("high")

    # The model does not output normalized outputs.
    oracle = LogSoftmaxModule(model)
    substitute = create_vgg16_bn_cifar10()  # Using the PyTorch implementation
    
    # Check the cell above. Note that without log function. The loss doesn't seem correct.
    loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)

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
    )
    trainer.fit(
        surrogate_module,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_test,
    )
    return surrogate_module
