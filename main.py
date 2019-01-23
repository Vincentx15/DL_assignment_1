from model.net import TwoLayerNet
import model.data_loader as loader
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time


dataset = loader.ToyData()
batch_size = 10

device = torch.device("cuda:0")

# Create indices and sampler
test_split = .2
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)


start_time = time.time()



net = TwoLayerNet(2, 5, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader):

        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if not i % 100:  # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 100))
        #     running_loss = 0.0

print('Finished Training')

elapsed_time = time.time() - start_time

print('CPU time = ', elapsed_time)


device = torch.device("cuda:0")

start_time = time.time()

gpu_net = TwoLayerNet(2, 5, 1)
gpu_net.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader):

        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = gpu_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if not i % 100:  # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 100))
        #     running_loss = 0.0

print('Finished Training')

elapsed_time = time.time() - start_time

print('GPU time = ', elapsed_time)

running_loss = 0.0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        if not i % 100:  # print every 2000 mini-batches
            print('loss: %.3f' % (running_loss / 100))
            running_loss = 0.0

