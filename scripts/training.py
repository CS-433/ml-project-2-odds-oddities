"""training.py: helper functions for convenient training."""
import segmentation_models_pytorch as smp


def train_epoch(model, device, dataloader, criterion, optimizer):
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

        print(f'[{batch_id + 1:5d}] loss: {loss.item():.3f}')

    average_loss = running_loss / len(dataloader)
    average_f1 = running_f1 / len(dataloader)

    return average_loss, average_f1


def valid_epoch(model, device, dataloader, criterion):
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
