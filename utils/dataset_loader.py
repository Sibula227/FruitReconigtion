from torchvision import datasets
from torch.utils.data import DataLoader

def create_dataloader(path, transform, batch_size=32, shuffle=True):

    dataset = datasets.ImageFolder(
        root=path,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataset, loader