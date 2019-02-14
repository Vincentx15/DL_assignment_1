import torch.nn as nn
import torch.nn.functional as functional


class Block(nn.Module):
    """
    Block within a residual network. Note that the double convolution is bypassed by the input.
    """

    def __init__(self, channels=16):
        """
        Define the Pytorch functions used inside the model.
        """

        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Performs the forward pass. Automated process by the torch.nn.Module class.

        Args:
            x: input tensor.

        Returns:
            x: output tensor.
        """

        identity = x

        out = self.conv1(x)
        out = functional.relu(out)

        out = self.conv2(out)
        out = functional.relu(out)

        out += identity

        out = functional.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, size=64, num_blocks=3, channels=128):
        """
        Define the Pytorch functions used inside the model.
        """

        super(ResNet, self).__init__()

        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)

        self.blocks = nn.Sequential(*[
            Block(channels) for _ in range(self.num_blocks)
        ])

        self.max = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)

        end_size = size // 2

        self.fc1 = nn.Linear(32 * end_size * end_size, 100)
        self.classifier = nn.Linear(100, 2)

    def forward(self, x):
        """
        Performs the forward pass. Automated process by the torch.nn.Module class.

        Args:
            x: input tensor.

        Returns:
            x: output tensor.
        """

        # Obtain the batch size
        n = x.size(0)

        x = self.conv1(x)
        x = functional.relu(x)

        x = self.blocks(x)

        x = self.max(x)

        x = self.conv2(x)
        x = functional.relu(x)

        # Flatten the tensor for dense layers
        x = x.view(n, -1)

        x = functional.relu(self.fc1(x))

        x = self.classifier(x)

        return x
