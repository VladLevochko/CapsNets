from collections import defaultdict

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import RandomCrop, ToTensor, Compose


class DataProvider:
    DATA_TYPES = {"mnist", "mnist_single", "mnist_reduced"}

    def __init__(self):
        pass

    def get_data_loaders(self, data_type="mnist", **kwargs):
        if data_type == "mnist":
            return self.load_data(**kwargs)
        elif data_type == "mnist_single":
            return self.load_single(**kwargs)
        elif data_type == "mnist_reduced":
            return self.load_mnist_reduced(**kwargs)

    def load_single(self, data_path="data", download=True, **kwargs):
        batch_size = 1
        dataset = MNIST(data_path, train=True, download=download,
                        transform=Compose([RandomCrop(size=28, padding=2), ToTensor()]))
        dataset.data = dataset.data[0][None]
        dataset.targets = dataset.targets[0][None]
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def load_data(self, data_path="data", download=True, batch_size=128, **kwargs):
        train_loader = DataLoader(MNIST(data_path, train=True, download=download,
                                        transform=Compose([
                                            RandomCrop(size=28, padding=2),
                                            ToTensor()])),
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(MNIST(data_path, train=False, download=download,
                                       transform=Compose([ToTensor()])),
                                 batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    def load_mnist_reduced(self, data_path="data", download=True, batch_size=128, reduction_factor=0.1, **kwargs):
        train_dataset = MNIST(data_path, train=True, download=download,
                        transform=Compose([RandomCrop(size=28, padding=2), ToTensor()]))
        train_dataset = self.reduce_dataset(train_dataset, reduction_factor)
        test_dataset = MNIST(data_path, train=False, download=download,
                        transform=Compose([RandomCrop(size=28, padding=2), ToTensor()]))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def reduce_dataset(self, dataset, reduction_factor=0.1):
        data = dataset.data
        targets = dataset.targets

        classes = defaultdict(list)
        for i, target in enumerate(targets):
            classes[target].append(i)
        selected = []
        for _, indices in classes.items():
            selected.extend(indices[:len(indices) * reduction_factor])
        selected.sort()

        dataset.data, dataset.targets = data[selected], targets[selected]

        return dataset
