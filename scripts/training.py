"""training.py: helper functions for convenient training."""
import random

import numpy as np
import segmentation_models_pytorch as smp
import torch


def _train_epoch(model, device, dataloader, criterion, optimizer) -> (float, float):
    """
    Train the model and return epoch loss and average f1 score.

    :param model: to be trained (with pretrained encoder)
    :param device: either "cuda" or "cpu"
    :param dataloader: with images
    :param criterion: loss function
    :param optimizer: some SGD implementation
    :return: average loss, average f1 score
    """
    running_loss, running_f1 = 0., 0.
    model.train()

    for batch_id, (inputs, labels) in enumerate(dataloader, 0):

        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.float())
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        # calculate metrics
        running_loss += loss.item()

        tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels, mode='binary', threshold=0.5)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro-imagewise')
        running_f1 += f1_score.item()

    average_loss = running_loss / len(dataloader)
    average_f1 = running_f1 / len(dataloader)

    return average_loss, average_f1


def _valid_epoch(model, device, dataloader, criterion) -> (float, float):
    """
    Validate the model performance by calculating epoch loss and average f1 score.

    :param model: used for inference
    :param device: either "cuda" or "cpu"
    :param dataloader: with validation fold of images
    :param criterion: loss function
    :return: average loss, average f1 score
    """
    running_loss, running_f1 = 0., 0.
    model.eval()

    for inputs, labels in dataloader:

        # use gpu whenever possible
        inputs, labels = inputs.to(device), labels.to(device)

        # predict
        outputs = model(inputs.float())

        # calculate metrics
        loss = criterion(outputs, labels.float())
        running_loss += loss.item()

        tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels, mode='binary', threshold=0.5)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro-imagewise')
        running_f1 += f1_score.item()

    average_loss = running_loss / len(dataloader)
    average_f1 = running_f1 / len(dataloader)

    return average_loss, average_f1


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


def train_model(model, criterion, optimizer, dataloaders, num_epochs):
    train_loader, valid_loader = dataloaders

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_losses, valid_losses, train_f1s, valid_f1s = [], [], [], []

    for epoch in range(num_epochs):

        train_loss, train_f1 = _train_epoch(model, device, train_loader, criterion, optimizer)
        valid_loss, val_f1 = _valid_epoch(model, device, valid_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_f1s.append(train_f1)
        valid_f1s.append(val_f1)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, valid_loss, val_f1))

    return train_losses, valid_losses, train_f1s, valid_f1s
