import torch.nn as nn
import torch.nn.functional as functional


class Network(nn.Module):

    def __init__(self, input_size=64):

        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.max = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        size = input_size // 2

        self.fc1 = nn.Linear(128 * size * size, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):

        # Obtain the batch size
        n = x.size(0)

        x = self.conv1(x)
        x = functional.relu(x)

        x = self.conv2(x)
        x = functional.relu(x)

        x = self.conv3(x)
        x = functional.relu(x)

        x = self.max(x)

        x = self.conv4(x)
        x = functional.relu(x)

        # Flatten the tensor for dense layers
        x = x.view(n, -1)

        x = functional.relu(self.fc1(x))

        x = self.fc2(x)

        return x
