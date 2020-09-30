import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData, MNIST
from torchvision.transforms import ToTensor, RandomCrop
from argparse import ArgumentParser

from train import Trainer


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--lr", default=0.1)

    return parser.parse_args()


def main(arguments):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = load_data(arguments)
    # train_loader, test_loader = load_fake_data()

    trainer = Trainer(train_loader, test_loader, device, arguments.lr)
    trainer.run(arguments.epochs)


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


def load_data(arguments):
    data_path = "../data"
    download = True
    train_loader = DataLoader(MNIST(data_path, train=True, download=download,
                                    transform=transforms.Compose([
                                        RandomCrop(size=28, padding=2),
                                        ToTensor()])),
                              batch_size=arguments.batch_size, shuffle=True)
    test_loader = DataLoader(MNIST(data_path, train=False, download=download,
                                   transform=transforms.Compose([transforms.ToTensor()])),
                             batch_size=arguments.batch_size, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)

