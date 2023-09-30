"""
create_surrogate_model.py

This module creates surrogate models for black-box attacks.
"""


import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from aiml.surrogate_model.models import LogSoftmaxModule, Surrogate, create_cifar10_model, create_cifar100_model


def create_substitute(dataloader_train):
    """Create a substitute model based on training dataloader"""
    num_classes = len(dataloader_train.dataset.classes)
    dataset_size = len(dataloader_train.dataset)
    num_channels = dataloader_train.dataset[0][0].shape[0]

    # Retrieve the image size from the first sample in the dataset
    sample_image, _ = next(iter(dataloader_train))
    image_size = sample_image.shape[-2:]

    print(num_classes, dataset_size, num_channels, image_size)

    if num_classes == 10 and dataset_size == 50000 and num_channels == 3 and image_size == torch.Size([32, 32]):
        surrogate = create_cifar10_model()
    elif num_classes == 100 and dataset_size == 50000 and num_channels == 3 and image_size == torch.Size([32, 32]):
        surrogate = create_cifar100_model()

    return surrogate


def create_surrogate_model(model, dataloader_train, dataloader_test):
    """Create and train a surrogate model for CIFAR-10 dataset using PyTorch Lightning."""
    MAX_EPOCHS = 25
    LEARNING_RATE = 0.0005

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if str(device) == "cuda:0":
        torch.set_float32_matmul_precision("high")

    oracle = LogSoftmaxModule(model)

    substitute = create_substitute(dataloader_train)

    loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)

    num_training_batches = len(dataloader_train)

    surrogate_module = Surrogate(
        lr=LEARNING_RATE,
        num_training_batches=num_training_batches,
        oracle=oracle,
        substitute=substitute,
        loss_fn=loss_fn,
        num_classes=len(dataloader_train.dataset.classes),
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
