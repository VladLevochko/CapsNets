import torch
from torch.nn import Module, MSELoss
from torch.nn.functional import one_hot


class MarginLoss(Module):
    def __init__(self, lambda_reconstruction):
        super().__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.m_plus = 0.9

    def forward(self, y_predicted, y_true):
        value = y_true * torch.clamp(self.m_plus - y_predicted, min=0.) ** 2 \
            + self.lambda_reconstruction * (1 - y_true) * torch.clamp(y_predicted - (1 - self.m_plus), min=0.) ** 2
        return value.sum(dim=1).mean()


class CapsuleLoss(Module):
    def __init__(self):
        super().__init__()
        self.margin_loss = MarginLoss(0.5)
        self.reconstruction_loss = MSELoss()
        self.lambda_reconstruction = 0.001

    def forward(self, y_predicted, y_true, reconstruction, image):
        y_true = one_hot(y_true, 10)
        margin_loss = self.margin_loss(y_predicted, y_true)
        reconstruction_loss = self.lambda_reconstruction * self.reconstruction_loss(reconstruction, image)
        return margin_loss + reconstruction_loss
