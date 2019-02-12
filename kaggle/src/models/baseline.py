import torch.nn as nn
import torch.nn.functional as functional


class Baseline(nn.Module):

    def __init__(self, size=64):

        super(Baseline, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.max = nn.MaxPool2d(2)

        out_size = size // 2

        self.fc1 = nn.Linear(32 * out_size * out_size, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):

        # Obtain the batch size
        n = x.size(0)

        x = self.conv1(x)
        x = functional.relu(x)

        x = self.conv2(x)
        x = functional.relu(x)

        x = self.max(x)

        # Flatten the tensor for dense layers
        x = x.view(n, -1)

        x = functional.relu(self.fc1(x))

        x = self.fc2(x)

        return x
