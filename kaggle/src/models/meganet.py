import torch.nn as nn


class MegaNet(nn.Module):
    """
    Deep model with spatial preservation in conv layers.
    """

    def __init__(self, size=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True))

        out_size = size // 2

        self.net2 = nn.Sequential(
            nn.Linear(64*out_size*out_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2))

    def forward(self, input):

        # Obtain the batch size
        n = input.size(0)

        out = self.net(input)
        out = out.view(n, -1)
        out = self.net2(out)
        return out
