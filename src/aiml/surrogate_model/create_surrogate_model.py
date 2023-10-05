"""
create_surrogate_model.py

This module creates surrogate models for black-box attacks.
"""


import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from aiml.surrogate_model.models import LogSoftmaxModule, Surrogate, create_substitute_model


def get_num_classes(dataloader):
    """Get number of classes from a dataloader"""
    try:
        return len(dataloader.dataset.classes)
    except:
        unique_labels = set()
        for batch in dataloader:
            _, labels = batch
            unique_labels.update(labels.numpy().tolist())
        return len(unique_labels)


def create_substitute(dataloader_train, num_classes):
    """Create a substitute model based on training dataloader"""
    num_channels = dataloader_train.dataset[0][0].shape[0]

    surrogate = create_substitute_model(num_classes, num_channels)

    return surrogate


def create_surrogate_model(model, dataloader_train, dataloader_test):
    """Create and train a surrogate model using PyTorch Lightning."""
    MAX_EPOCHS = 50
    LEARNING_RATE = 0.0005

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if str(device) == "cuda:0":
        torch.set_float32_matmul_precision("high")

    oracle = LogSoftmaxModule(model)

    num_classes = get_num_classes(dataloader_train)

    substitute = create_substitute(dataloader_train, num_classes)

    loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)

    num_training_batches = len(dataloader_train)

    surrogate_module = Surrogate(
        lr=LEARNING_RATE,
        num_training_batches=num_training_batches,
        oracle=oracle,
        substitute=substitute,
        loss_fn=loss_fn,
        num_classes=num_classes,
        softmax=True
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        enable_progress_bar=True,
        logger=TensorBoardLogger(
            "logs", name="surrogate", default_hp_metric=False
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
