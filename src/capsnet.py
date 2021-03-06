import torch
from torch.nn import Module, Conv2d, Sequential, Linear, ReLU, Sigmoid
from torch.nn.functional import one_hot

from layers import PrimaryCapsules, RoutingCapsules


class CapsNetMnist(Module):
    def __init__(self, device):
        super().__init__()

        self.conv1 = Conv2d(1, 256, 9)
        self.relu = ReLU()
        self.primary = PrimaryCapsules(256, 256, 8, 9)
        self.dense = RoutingCapsules(8, 16, 1152, 10, 3, device)

        self.decoder = Sequential(
            Linear(16 * 10, 512),
            ReLU(inplace=True),
            Linear(512, 1024),
            ReLU(inplace=True),
            Linear(1024, 1 * 28 * 28),
            Sigmoid()
        )

    def forward(self, input, label=None):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.primary(x)
        x = self.dense(x)
        length = torch.norm(x, dim=-1)

        x = x * one_hot(label, 10)[:, :, None]
        reconstruction = self.decoder(x.view(x.size(0), -1))

        return length, reconstruction.view(*input.shape)
