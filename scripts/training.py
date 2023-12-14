"""training.py: helper functions for convenient training."""
import os
import random

import numpy as np
import segmentation_models_pytorch as smp
import torch
from ray import tune
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.evaluation import MetricMonitor


def train_epoch(
    model, dataloader, criterion, optimizer, scheduler, epoch, **kwargs
) -> (float, float):
    """
    Train the model and return epoch loss and average f1 score.

    :param model: to be trained (with pretrained encoder)
    :param dataloader: with images
    :param criterion: loss function
    :param optimizer: some SGD implementation
    :param scheduler: for optimizing learning rate
    :param epoch: current epoch
    :param kwargs: used for saving the predictions for ensembling
    :return: average loss, average f1 score
    """
    ensembler = kwargs.get("ensembler")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()

    metric_monitor = MetricMonitor()
    stream = tqdm(dataloader)

    for i, (inputs, labels) in enumerate(stream, 1):

        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits = model(inputs.float())
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()
        scheduler.step()

        ensembler.update(logits.sigmoid(), labels, "training") if ensembler else None

        tp, fp, fn, tn = smp.metrics.get_stats(
            logits.sigmoid(), labels, mode="binary", threshold=0.5
        )
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("f1", f1_score.item())

        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=(3 - len(str(epoch))) * " " + str(epoch),  # for better alignment,
                metric_monitor=metric_monitor,
            )
        )

    return metric_monitor.averages()


@torch.no_grad()
def valid_epoch(model, dataloader, criterion, epoch, **kwargs) -> (float, float):
    """
    Validate the model performance by calculating epoch loss and average f1 score.

    :param model: used for inference
    :param dataloader: with validation fold of images
    :param criterion: loss function
    :param epoch: current epoch
    :param kwargs: used for saving the predictions for ensembling
    :return: average loss, average f1 score
    """
    ensembler = kwargs.get("ensembler")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    metric_monitor = MetricMonitor()
    stream = tqdm(dataloader)

    for i, (inputs, labels) in enumerate(stream, 1):

        # use gpu whenever possible
        inputs, labels = inputs.to(device), labels.to(device)

        # predict
        logits = model(inputs.float())

        # save predictions and labels down to monitor (JSON eventually)
        ensembler.update(logits.sigmoid(), labels, "validation") if ensembler else None

        # calculate metrics
        loss = criterion(logits, labels.float())

        tp, fp, fn, tn = smp.metrics.get_stats(
            logits.sigmoid(), labels, mode="binary", threshold=0.35
        )
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("f1", f1_score.item())

        stream.set_description(
            "Epoch: {epoch}. Validation. {metric_monitor}".format(
                epoch=(3 - len(str(epoch))) * " " + str(epoch),  # for better alignment,
                metric_monitor=metric_monitor,
            )
        )

    loss, f1 = metric_monitor.averages()
    tune.report(loss=loss, f1=f1) if kwargs.get("tune") else None

    return loss, f1


def train_model(
    model, dataloaders, criterion, optimizer, scheduler, num_epochs, **kwargs
) -> tuple:
    """
    Train model for number of epochs and calculate loss and f1.

    :param model: to be trained (with pretrained encoder)
    :param dataloaders: tuple of dataloaders with images (train and validation)
    :param criterion: loss function
    :param optimizer: some SGD implementation
    :param scheduler: for optimizing learning rate
    :param num_epochs:
    :return: lists of train_losses, valid_losses, train_f1s, valid_f1s
    """
    train_loader, valid_loader = dataloaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_losses, valid_losses, train_f1s, valid_f1s = [], [], [], []

    for i in range(num_epochs):

        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, i + 1
        )
        train_losses.append(train_loss)
        train_f1s.append(train_f1)

        if valid_loader:
            valid_loss, val_f1 = valid_epoch(
                model, valid_loader, criterion, i + 1, **kwargs
            )
            valid_losses.append(valid_loss)
            valid_f1s.append(val_f1)

    return train_losses, valid_losses, train_f1s, valid_f1s


def tune_hyperparams(
    config, encoder: str, decoder: str, datasets: tuple, checkpoint_dir=None
):

    train_dataset, val_dataset = datasets

    # Create training and validation loaders by providing current K-Fold train/validation indices to Sampler
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"])
    valid_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # Initialize model
    model = smp.create_model(decoder, encoder_name=encoder, encoder_weights="imagenet")
    optimizer = torch.optim.Adam(model.parameters(), config["lr"])
    scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_loader.dataset) * config["num_epochs"])
        // train_loader.batch_size,
    )

    criteria_dict = {
        'dice_loss': smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True),
        'focal_loss': smp.losses.FocalLoss(smp.losses.BINARY_MODE)
    }
    criterion = criteria_dict[config["criterion"]]

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    _ = train_model(
        model,
        (train_loader, valid_loader),
        criterion,
        optimizer,
        scheduler_,
        int(config["num_epochs"]),
        tune=True,
    )


def setup_seed(seed: int, cuda: bool = False):
    """
    Create global seed for torch, numpy and cuda.

    :param seed:
    :param cuda: boolean whether to use gpu
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
