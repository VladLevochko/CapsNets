import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData, MNIST
from torchvision.transforms import ToTensor, RandomCrop

from train import Trainer


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate = 0.1
    epochs_number = 10

    # train_loader, test_loader = load_data()
    train_loader, test_loader = load_fake_data()

    trainer = Trainer(train_loader, test_loader, device, learning_rate)
    trainer.run(epochs_number)


def load_fake_data():
    batch_size = 1
    train_loader = DataLoader(FakeData(size=100, image_size=(1, 28, 28),
                                       transform=transforms.Compose([
                                        RandomCrop(size=28, padding=2),
                                        ToTensor()])),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(FakeData(size=10, image_size=(1, 28, 28),
                                      transform=transforms.Compose([transforms.ToTensor()])),
                             batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def load_data():
    data_path = "data"
    download = True
    batch_size = 1
    train_loader = DataLoader(MNIST(data_path, train=True, download=download,
                                    transform=transforms.Compose([
                                        RandomCrop(size=28, padding=2),
                                        ToTensor()])),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MNIST(data_path, train=False, download=download,
                                   transform=transforms.Compose([transforms.ToTensor()])),
                             batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    main()

