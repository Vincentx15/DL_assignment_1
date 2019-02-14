import torch
import torch.optim as optim

import time
import copy

import json


class Writer:
    """
    Class for writing training logs.
    """

    def __init__(self, file):
        self.file = file

    def __call__(self, data_dict):
        with open(self.file, 'a') as f:
            f.write(json.dumps(data_dict) + '\n')


def train_model(model, criterion, optimizer, scheduler, device, dataloaders, writer=None,
                num_epochs=25, log_interval=30, wall_time=.25, save=None, patience=5):
    """
    Performs the entire training routine.

    Args:
        model (torch.nn.Module): the model to train.
        criterion: the criterion to use (eg CrossEntropy)
        optimizer: the optimizer to use (eg SGD or Adam)
        scheduler: the scheduler for the learning rate
        device: the device on which to run
        dataloaders: the dataloaders for training and validation
        writer (Writer): the writer for gathering training data (loss and accuracy)
        num_epochs (int): the number of epochs for training
        log_interval (int): controls the rate of logs
        wall_time (float): the wall time (in hours) given to the compute node, to make sure everything is saved
        save (str): where to save the model mid-training
        patience (int): patience during training

    Returns:
        model: the best model among all epochs.
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    j = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()   # Set model to training mode
            else:
                model.eval()    # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_nb = 0

            # Iterate over data.
            for i, sample in enumerate(dataloaders[phase]):

                data, labels = sample['image'], sample['label']

                # Move to device
                data = data.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # Forward
                    outputs = model(data)
                    _, predictions = outputs.max(1)

                    # Computing the loss, according to the criterion
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                corrects = int(torch.sum(predictions == labels))

                # Statistics
                running_nb += data.size(0)
                running_loss += loss.item() * data.size(0)
                running_corrects += corrects

                if writer is not None and i % log_interval == 0 and i != 0 and phase == 'train':

                    time_elapsed = time.time() - since
                    writer({
                        'epoch': epoch + round(i / len(dataloaders['train']), 3),
                        'phase': phase,
                        'loss': round(running_loss / running_nb, 4),
                        'accuracy': round(running_corrects / running_nb, 4),
                        'time': round(time_elapsed / 60, 2)
                    })

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            if phase == 'valid':
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_acc)
                else:
                    scheduler.step()

            time_elapsed = time.time() - since
            print('{} Loss: {:.4f} Acc: {:.4f} ({:.0f}m {:.0f}s)'.format(
                phase, epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60))

            if writer is not None:
                writer({
                    'epoch': epoch + 1,
                    'phase': phase,
                    'loss': round(epoch_loss, 4),
                    'accuracy': round(epoch_acc, 4),
                    'time': round(time_elapsed / 60, 2)
                })

            # Deep copy the model if it yields better results
            if phase == 'valid' and epoch_acc > best_acc:

                j = 0

                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                if isinstance(save, str):
                    torch.save(model, save)

            else:
                j += 1

        # If validation objective has not improved in a while
        if j > patience:
            break

        # Break out of the loop if we might go beyond the wall time
        if time_elapsed * (1 + 1/(epoch + 1)) > .95 * wall_time * 3600:
            break

        print()

    time_elapsed = time.time() - since
    print()
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60)
    )
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model
