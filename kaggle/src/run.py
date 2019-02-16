from utils.dataset import KaggleDataset
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
import torch
from torch import nn

from models import baseline, cnn, resnet

from utils import training

import numpy as np

import json

import os


def run_experiment(name, model, root_dir='', split=.9, size=60, random=True, seed=None,
                   torch_seed=None, n_epochs=100, wall_time=4, batch_size=10,
                   learning_rate=1e-2, log_interval=10, save=None):
    """
    Runs an experiment (training a model on a pre-processed dataset).

    Args:
        name (str): name of the experiment and folder within the results directory where the results will be stored.
        model (torch.nn.Module): model to train.
        root_dir (str): the root directory of the project. Useful for running on a remote cluster.
        split (float, between 0 and 1): the proportion of examples to use as training set
        size (int, at most 64): the size of the training examples. A random crop is applied.
        random (bool): whether to randomize the training/validation split.
        seed (int): the seed for numpy.
        torch_seed (int): the random seed for torch.
        n_epochs (int): the maximum number of epochs to train on.
        wall_time (float): the maximum number of hours to train for.
        batch_size (int): the batch size.
        learning_rate (float): the learning rate
        log_interval (int): log the performance every log_interval batches
        save (str): where to save the best performing model mid-training
            (especially useful with an unstable compute environment)
    """

    configuration = {
        'name': name,
        'model': str(model),
        'split': split,
        'size': size,
        'random': random,
        'seed': seed,
        'torch_seed': torch_seed,
        'n_epochs': n_epochs,
        'wall_time': wall_time,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'log_interval': log_interval
    }

    result_folder = root_dir + 'results/' + name + '/'
    result_log = root_dir + 'logs/' + name + '.json'

    try:
        os.mkdir(result_folder)
    except FileExistsError:
        pass

    with open(result_folder + 'config.json', 'w') as f:
        json.dump(configuration, f, indent=2, separators=(',', ': '))

    ############################################
    #           Network architecture           #
    ############################################

    torch.manual_seed(torch_seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    ############################################
    #            Defining the writer           #
    ############################################

    writer = training.Writer(result_log)

    ############################################
    #            Defining the dataset          #
    ############################################

    transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    dataset = KaggleDataset(
        data_dir='../data/trainset/',
        transform=transform
    )

    ############################################
    #    Splitting into training/validation    #
    ############################################

    n = len(dataset)
    indices = list(range(n))

    if random:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[:int(split * n)]
    valid_indices = indices[int(split * n):]

    train_set = Subset(dataset, train_indices)
    valid_set = Subset(dataset, valid_indices)

    ############################################
    #  Creating the corresponding dataloaders  #
    ############################################

    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=batch_size
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        shuffle=True,
        batch_size=batch_size
    )

    print('>> Total batch number for training: {}'.format(len(train_loader)))
    print('>> Total batch number for validation: {}'.format(len(valid_loader)))
    print()

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader
    }

    ############################################
    #            Training parameters           #
    ############################################

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

    ############################################
    #              Actual Training             #
    ############################################

    best_model = training.train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        dataloaders=dataloaders,
        writer=writer,
        num_epochs=n_epochs,
        log_interval=log_interval,
        wall_time=wall_time,
        save=save
    )

    torch.save(best_model, result_folder + 'best_model.pth')


if __name__ == '__main__':

    with open('config.json', 'r') as f:
        config = json.load(f)

    m = config.get('model', 'baseline')
    size = config['size']

    networks = {
        'baseline': baseline.Baseline,
        'cnn': cnn.Network,
        'resnet': resnet.ResNet,
        'huge-cnn': cnn.HugeNetwork
    }

    net = networks[m](size)
    del config['model']

    run_experiment(model=net, root_dir='../', **config)
