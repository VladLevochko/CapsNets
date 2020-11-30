import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        self.images = []
        self.ids = []
        self.read()

    def read(self):
        images_path = os.path.join(self.root, "celeba_images.npy")
        labels_path = os.path.join(self.root, "celeba_labels.npy")
        self.images = np.load(images_path)
        self.ids = np.load(labels_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        if self.transform is not None:
            image = self.transform(image)

        return image, self.ids[index] - 1
