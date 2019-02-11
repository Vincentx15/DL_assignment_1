import numpy as np
from model.ToyConvNet import ToyConvNet
import torch
import torch.nn as nn
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import time

# Get Data and DataLoader
mnist_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=mnist_transforms, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=mnist_transforms, download=True)

batch_size = 64

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create indices and sampler
validation_split = .2
dataset_size = len(mnist_train)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, sampler=validation_sampler)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)


# Tune Net
def train():
    net = ToyConvNet()
    if device:
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            if device:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if not i % 200:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

        net.eval()
        total = 0
        correct = 0
        for i, (inputs, labels) in enumerate(validation_loader):
            if device:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += int(predicted.eq(labels.data).cpu().sum())
        print('Epoch : %d Valid Acc : %.3f' % (epoch + 1, 100. * correct / total))
        net.train()


start_time = time.time()

train()

elapsed_time = time.time() - start_time
print('Finished Training')
print('GPU time = ', elapsed_time)

# start_time = time.time()
#
# device = None
# train()
#
# elapsed_time = time.time() - start_time
# print('Finished Training')
# print('CPU time = ', elapsed_time)
