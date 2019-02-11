from utils.dataset import KaggleDataset
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
import torch

from models import baseline, cnn, resnet

from utils import training

import numpy as np

import json

import os


def run_experiment(name, model, root_dir='', split=.9, size=64, random=True, seed=None, n_epochs=300,
                   wall_time=7.5, batch_size=100, learning_rate=1e-3, log_interval=30, save=None):

    configuration = {
        'name': name,
        'split': split,
        'size': size,
        'random': random,
        'seed': seed,
        'n_epochs': n_epochs,
        'wall_time': wall_time,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'log_interval': log_interval
    }

    result_folder = root_dir + 'results/' + name + '/'
    result_log = result_folder + 'logs.json'

    try:
        os.mkdir(result_folder)
    except FileExistsError:
        pass

    with open(result_folder + 'config.json', 'w') as f:
        json.dump(configuration, f)

    ############################################
    #           Network architecture           #
    ############################################

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
        # transforms.Grayscale(),
        transforms.RandomCrop(size),
        transforms.ToTensor()
    ])

    dataset = KaggleDataset(
        data_dir='data/trainset/',
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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
        'resnet-3-128': resnet.ResNet(num_blocks=3, channels=128)
    }

    net = networks[m](size)

    run_experiment(model=net, **config)
