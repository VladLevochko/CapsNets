from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Linear, Softmax


class ConvNet(Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.conv1 = Conv2d(1, 32, kernel_size=3)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(32, 64, kernel_size=3)
        self.relu = ReLU()
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.fc = Linear(9216, 10)
        self.softmax = Softmax()

    def forward(self, input, *args, **kwargs):
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)

        return x
