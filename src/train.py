import torch
from torch.nn.functional import one_hot
from torch.optim import Adam

from capsnet import CapsNetMnist
from loss import CapsuleLoss
from tqdm import tqdm


class Trainer:
    def __init__(self, train_loader, test_loader, device, learning_rate):
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = CapsNetMnist().to(device)
        self.loss = CapsuleLoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def run(self, epochs_number):
        for epoch in range(epochs_number):
            print("\nEpoch", epoch)
            self.train_step()
            self.eval_step()

    def train_step(self):
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0
        for images, labels in tqdm(self.train_loader):
            self.optimizer.zero_grad()

            predictions, reconstructions = self.model(images, labels)
            loss = self.loss(predictions, one_hot(labels, 10), reconstructions, images)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()

            _, predictions = predictions.max(dim=1)
            correct_predictions += (predictions == labels).sum()
            total_predictions += images.size(0)

        epoch_accuracy = torch.true_divide(correct_predictions, total_predictions)

        print("Train loss {} accuracy {}".format(total_loss, epoch_accuracy))

    def eval_step(self):
        correct_predictions = 0
        total_predictions = 0
        for images, labels in tqdm(self.test_loader):
            predictions, reconstruction = self.model(images, labels)

            _, predictions = predictions.max(dim=1)
            correct_predictions += (predictions == labels).sum()
            total_predictions += images.size(0)

        epoch_accuracy = torch.true_divide(correct_predictions, total_predictions)

        print("Test accuracy {}".format(epoch_accuracy))