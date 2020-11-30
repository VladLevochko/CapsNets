import os
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset, ImageFolder


class LfwDataset(VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

        self.images_path = os.path.join(self.root, "lfw_images.npy")
        self.pairs_path = os.path.join(self.root, "lfw_pairs.npy")

        self.images = np.load(self.images_path)
        self.pairs = np.load(self.pairs_path)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, index

    def __len__(self):
        return len(self.images)

    def get_pairs(self):
        return self.pairs


if __name__ == "__main__":
    dataset = LfwDataset("/Users/vlad/PycharmProjects/data/lfw_np")

    image, _ = dataset[0]
    print(image.size)
