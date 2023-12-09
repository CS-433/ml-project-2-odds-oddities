"""training.py: helper functions for convenient training."""
import random
from collections import defaultdict

import numpy as np
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm


class MetricMonitor:
    """
    Inspired from examples of Albumentation:
        https://albumentations.ai/docs/examples/pytorch_classification/
    """
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.metrics = {}
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def averages(self):
        """Return the average per metric (loss, f1)"""
        return tuple([metric['avg'] for (metric_name, metric) in self.metrics.items()])

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, **kwargs) -> (float, float):
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
    ensembler = kwargs.get('ensembler')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        ensembler.update(logits.sigmoid(), labels, 'training') if ensembler else None

        tp, fp, fn, tn = smp.metrics.get_stats(logits.sigmoid(), labels, mode='binary', threshold=0.5)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro-imagewise')

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("f1", f1_score.item())

        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=(3 - len(str(epoch))) * " " + str(epoch),  # for better alignment,
                metric_monitor=metric_monitor
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
    ensembler = kwargs.get('ensembler')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    metric_monitor = MetricMonitor()
    stream = tqdm(dataloader)

    for i, (inputs, labels) in enumerate(stream, 1):

        # use gpu whenever possible
        inputs, labels = inputs.to(device), labels.to(device)

        # predict
        logits = model(inputs.float())

        # save predictions and labels down to monitor (JSON eventually)
        ensembler.update(logits.sigmoid(), labels, 'validation') if ensembler else None

        # calculate metrics
        loss = criterion(logits, labels.float())

        tp, fp, fn, tn = smp.metrics.get_stats(logits.sigmoid(), labels, mode='binary', threshold=0.5)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro-imagewise')

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("f1", f1_score.item())

        stream.set_description(
            "Epoch: {epoch}. Validation. {metric_monitor}".format(
                epoch=(3 - len(str(epoch))) * " " + str(epoch),  # for better alignment,
                metric_monitor=metric_monitor
            )
        )

    return metric_monitor.averages()


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


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs) -> tuple:
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_losses, valid_losses, train_f1s, valid_f1s = [], [], [], []

    for i in range(num_epochs):

        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, i + 1
        )
        valid_loss, val_f1 = valid_epoch(
            model, valid_loader, criterion, i + 1
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_f1s.append(train_f1)
        valid_f1s.append(val_f1)

    return train_losses, valid_losses, train_f1s, valid_f1s
