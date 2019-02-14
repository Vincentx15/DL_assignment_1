import torch.nn as nn



class ToyConvNet(nn.Module):
    def __init__(self):
        """

        D_in: input dimension
        H: dimension of hidden layer
        D_out: output dimension
        """
        super(ToyConvNet, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # Logistic Regression
        self.clf = nn.Linear(128 * 7 * 7, 10)

    def layer_resNet(self):
        nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))

    def forward(self, x):

        n = x.size(0)

        x = self.conv(x)

        x = x.view(n, -1)

        x = self.clf(x)

        return x
