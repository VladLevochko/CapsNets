import torch
from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter

from data_provider import DataProvider
from train import TrainerFactory


# TODO: try different kernel sizes


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, choices=["caps", "conv"], default="caps")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.9)

    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--download", type=bool, default=False)
    parser.add_argument("--data_type", type=str, choices=DataProvider.DATA_TYPES, default="mnist")
    parser.add_argument("--reduction_factor", type=float, default=0.1)

    return parser.parse_args()


def main(parameters):
    data_provider = DataProvider()
    train_loader, test_loader = data_provider.get_data_loaders(**parameters)

    writer = SummaryWriter()

    trainer_type = parameters["experiment"] + "net"
    trainer = TrainerFactory.create_trainer(trainer_type, train_loader, test_loader, writer, **parameters)
    trainer.run(parameters["epochs"])

    writer.close()


if __name__ == "__main__":
    arguments = parse_arguments()
    parameters = arguments.__dict__
    parameters["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print("Parameters\n", parameters)

    main(parameters)

