import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, CenterCrop, Normalize, RandomHorizontalFlip

from reid.celeba_dataset import CelebADataset
from reid.lfw_dataset import LfwDataset
from reid.reid_baseline import Baseline
from reid.reid_capsnet import ReidCapsnet
from reid.trainer import ReidTrainer


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--celeba_np_root", type=str, default="data/celeba_np")
    parser.add_argument("--lfw_root", type=str, default="data/lfw")
    parser.add_argument("--local", action="store_true")

    parser.add_argument("--lamda", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-3)

    parser.add_argument("--model_type", type=str, default="caps")
    parser.add_argument("--pretrained", action="store_true")

    return parser.parse_args()


def main(parameters):
    train_data = CelebADataset(root=parameters["celeba_np_root"],
                               transform=Compose([
                                   RandomHorizontalFlip(),
                                   ToTensor(),
                                   Normalize(127.5, 128)
                               ]))
    lfw_data = LfwDataset(parameters["lfw_root"], transform=Compose([
        ToTensor(),
        Normalize(127.5, 128)
    ]))

    num_workers = parameters["num_workers"]
    pin_memory = True if parameters["device"] == "cuda" else False
    parameters["device"] = torch.device(parameters["device"])

    if not parameters["local"]:
        train_loader = DataLoader(train_data, batch_size=parameters["batch_size"], shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        lfw_loader = DataLoader(lfw_data, batch_size=parameters["batch_size"], shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory)
    else:
        train_data = Subset(train_data, range(0, 100))
        train_loader = DataLoader(train_data, batch_size=parameters["batch_size"], shuffle=True)
        lfw_loader = DataLoader(lfw_data, batch_size=parameters["batch_size"], shuffle=False)

    writer = SummaryWriter()

    model = create_model(parameters)

    trainer = ReidTrainer(model, train_loader, None, lfw_loader, writer,
                          classes_number=10177, embedding_size=256, **parameters)
    trainer.run(parameters["epochs"])

    writer.close()


def create_model(parameters):
    model_type = parameters["model_type"]
    classes_number = 10177
    embedding_size = 256
    pretrained = parameters["pretrained"]

    if model_type == "conv":
        model = Baseline(num_classes=classes_number, embedding_size=embedding_size, pretrained=pretrained)
    elif model_type == "caps":
        model = ReidCapsnet(num_classes=classes_number, embedding_size=embedding_size,
                            device=parameters["device"], pretrained=pretrained)
    else:
        raise Exception("Unknown model type: " + model_type)

    return model


if __name__ == "__main__":
    arguments = parse_arguments()
    parameters = arguments.__dict__
    parameters["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if parameters["num_workers"] == 0:
        parameters["num_workers"] = os.cpu_count()

    print("Parameters\n", parameters)

    main(parameters)
