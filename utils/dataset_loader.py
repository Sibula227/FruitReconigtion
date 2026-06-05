from torch.utils.data import DataLoader
from utils.custom_dataset import FruitDataset

def create_dataloader(
    path,
    transform,
    batch_size=32,
    shuffle=True
):

    dataset = FruitDataset(
        root=path,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )

    return dataset, loader