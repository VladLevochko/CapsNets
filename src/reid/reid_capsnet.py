import torch
from torch.nn import Module, Sequential, Linear, BatchNorm1d, ReLU
from torch.nn.functional import normalize


from layers import PrimaryCapsules, RoutingCapsules
from reid.resnet_cut import resnet18_cut


class ReidCapsnet(Module):
    def __init__(self, num_classes=256, embedding_size=256, device="cpu", pretrained=False):
        super().__init__()
        self.model = resnet18_cut(pretrained=pretrained)
        self.primary = PrimaryCapsules(512, 512, 8, 3)
        self.routing = RoutingCapsules(64, 32, 8, embedding_size // 32, 3, device)
        self.relu = ReLU()
        self.fc = Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.primary(x)
        x = self.routing(x)
        x = x.view(x.size(0), -1)
        embedding = normalize(x, p=2, dim=1)
        x = self.relu(x)
        logits = self.fc(x)

        return embedding, logits





if __name__ == "__main__":
    m = ReidCapsnet()
    print(m)
    data = torch.randn((1, 3, 120, 120))
    result = m(data)
