from torch.nn import Module, Conv2d, MaxPool2d, Linear, Softmax, ReLU, Dropout
import torch.nn.functional as F


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


class ConvNetLarge(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 256, 5)
        self.relu = ReLU()
        p = 0.7
        self.dropout1 = Dropout(p)
        self.conv2 = Conv2d(256, 512, 5)
        self.dropout2 = Dropout()
        self.maxpool1 = MaxPool2d(2, 2)
        self.maxpool2 = MaxPool2d(2, 2)
        self.fc1 = Linear(8192, 2048)
        self.dropout3 = Dropout(p)
        self.fc2 = Linear(2048, 10)
        self.softmax = Softmax()

    def forward(self, x, *args, **kwargs):
        x = self.maxpool1(self.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.maxpool2(self.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
