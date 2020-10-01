import torch
from torch.nn import Module, Conv2d, Parameter


class PrimaryCapsules(Module):
    def __init__(self, in_channels, out_channels, caps_channels, kernel_size=9):
        super().__init__()
        self.caps_channels = caps_channels
        self.conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2)
        self.squash = Squash()

    def forward(self, x):
        output = self.conv(x)
        output = output.view(x.size(0), -1, self.caps_channels)
        # output = output.view(x.size(0), self.caps_channels, -1)
        output = self.squash(output)

        return output


class RoutingCapsules(Module):
    def __init__(self, in_channels, out_channels, in_capsules_number, capsules_number, routings_number, device):
        super().__init__()
        self.capsules_number = capsules_number
        self.in_capsules_number = in_capsules_number
        self.routings_number = routings_number
        self.device = device

        self.w = Parameter(torch.zeros(1, capsules_number, in_capsules_number, out_channels, in_channels))
        self.squash = Squash()

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(self.w, x)
        u_hat = u_hat.squeeze(-1)
        u_hat_detached = u_hat.detach()

        b = torch.zeros(x.size(0), self.capsules_number, self.in_capsules_number, 1).to(self.device)

        for i in range(self.routings_number - 1):
            c = torch.softmax(b, dim=1)
            s = (c * u_hat_detached).sum(dim=2)
            v = self.squash(s)  # v

            uv = torch.matmul(u_hat_detached, v.unsqueeze(-1))
            b += uv

        c = torch.softmax(b, dim=1)
        s = (c * u_hat).sum(dim=2)
        v = self.squash(s)

        return v


class Squash(Module):
    def forward(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-9)

        return scale * x
