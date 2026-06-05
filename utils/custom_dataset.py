from torchvision.datasets import ImageFolder
from PIL import Image

def rgb_loader(path):
    return Image.open(path).convert("RGB")

class FruitDataset(ImageFolder):

    def __init__(self, root, transform=None):
        super().__init__(
            root=root,
            transform=transform,
            loader=rgb_loader
        )