import torch
from torch.nn import Module, Linear, Sequential, BatchNorm1d, ReLU
from torch.nn.functional import normalize

from reid.resnet_cut import resnet18_cut


class Baseline(Module):
    def __init__(self, num_classes=256, embedding_size=256, pretrained=False):
        super().__init__()
        self.model = resnet18_cut(pretrained=pretrained)
        in_features = 512 * 3 * 4
        self.embedding = Linear(in_features, embedding_size)
        self.fc = Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        embedding = normalize(x, p=2, dim=1)
        logits = self.fc(x)

        return embedding, logits
