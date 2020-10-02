import torch
from argparse import ArgumentParser

from data_provider import DataProvider
from train import Trainer


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--data_type", type=str, choices=DataProvider.DATA_TYPES, default="mnist")

    return parser.parse_args()


def main(arguments):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_provider = DataProvider()
    train_loader, test_loader = data_provider.get_data_loaders(**dict(arguments.__dict__))

    trainer = Trainer(train_loader, test_loader, device, arguments.lr)
    trainer.run(arguments.epochs)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)

