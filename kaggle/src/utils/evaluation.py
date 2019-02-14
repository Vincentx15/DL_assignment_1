import numpy as np
import torch

import torch.nn.functional as functional

from tqdm import tqdm


def predict(model, dataloader, device):
    """
    Evaluates a model on a given set. Returns the prediction

    Args:
        model (torch.nn.Module):
        dataloader (torch.utils.data.Dataloader): dataloader of the set on which to make a prediction
        device (torch.device): device used to make the prediction

    Returns:
        np.array: array of the predictions.
    """

    results = []

    model.eval()

    with torch.no_grad():

        iterator = tqdm(
            dataloader,
            ascii=True,
            ncols=100,
            total=len(dataloader),
            desc='Predicting'
        )

        for sample in iterator:
            data = sample['image']

            # Move to device
            data = data.to(device)

            outputs = model(data)
            _, predictions = outputs.max(1)

            results.append(np.array(predictions.cpu()))

    return np.concatenate(results)


def compare(model, dataloader, device):
    """
    Evaluates a model on a given set. Returns the prediction

    Args:
        model (torch.nn.Module):
        dataloader (torch.utils.data.Dataloader): dataloader of the set on which to make a prediction
        device (torch.device): device used to make the prediction

    Returns:
        np.array: array of the predictions.
    """

    results = []
    labels = []

    model.eval()

    with torch.no_grad():

        iterator = tqdm(
            dataloader,
            ascii=True,
            ncols=100,
            total=len(dataloader),
            desc='Predicting'
        )

        for sample in iterator:
            data = sample['image']
            label = sample['label']

            # Move to device
            data = data.to(device)

            outputs = functional.softmax(model(data), dim=1)

            results.append(np.array(outputs.cpu()))
            labels.append(np.array(label))

    return np.concatenate(results), np.concatenate(labels)


def validation(model, dataloader, device):
    model.eval()

    with torch.no_grad():
        iterator = tqdm(
            dataloader,
            ascii=True,
            ncols=100,
            total=len(dataloader),
            desc='Predicting'
        )

        running_nb = 0
        running_corrects = 0

        for sample in iterator:
            data = sample['image']
            label = sample['label']

            # Move to device
            data = data.to(device)

            outputs = model(data)

            _, predictions = outputs.max(1)

            running_nb += data.size(0)
            running_corrects += int(torch.sum(predictions == label))

    return running_corrects / running_nb
