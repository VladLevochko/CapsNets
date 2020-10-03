import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from capsnet import CapsNetMnist
from loss import CapsuleLoss
from tqdm import tqdm


class Trainer:
    def __init__(self, train_loader, test_loader, writer=None, device="cpu", lr=0.1, lr_decay=0.9, **kwargs):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.writer = writer

        self.model = CapsNetMnist(device).to(device)
        self.loss = CapsuleLoss().to(device)

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_decay)

    def run(self, epochs_number):
        for epoch in range(epochs_number):
            print("\nEpoch", epoch)
            self.writer.add_scalar("lr", self.scheduler.get_lr(), epoch)

            self.train_step(epoch)
            self.eval_step(epoch)

            self.scheduler.step()

    def train_step(self, epoch_number):
        self.model.train()

        correct_predictions = 0
        total_predictions = 0
        total_loss = 0
        for images, labels in tqdm(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            predictions, reconstructions = self.model(images, labels)
            loss = self.loss(predictions, labels, reconstructions, images)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()

            _, predictions = predictions.max(dim=1)
            correct_predictions += (predictions == labels).sum()
            total_predictions += images.size(0)

        epoch_accuracy = torch.true_divide(correct_predictions, total_predictions)

        self.writer.add_scalar("Accuracy/Train", epoch_accuracy, epoch_number)
        self.writer.add_scalar("Loss/Train", total_loss, epoch_number)

        print("Train loss {} accuracy {}".format(total_loss, epoch_accuracy))

    def eval_step(self, epoch_number):
        self.model.eval()

        correct_predictions = 0
        total_predictions = 0
        total_loss = 0
        for images, labels in tqdm(self.test_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            predictions, reconstructions = self.model(images, labels)
            loss = self.loss(predictions, labels, reconstructions, images)
            total_loss += loss.item()

            _, predictions = predictions.max(dim=1)
            correct_predictions += (predictions == labels).sum()
            total_predictions += images.size(0)

        epoch_accuracy = torch.true_divide(correct_predictions, total_predictions)

        self.writer.add_scalar("Accuracy/Test", epoch_accuracy, epoch_number)
        self.writer.add_scalar("Loss/Test", total_loss, epoch_number)

        print("Test accuracy {}".format(epoch_accuracy))
