import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils.transforms import train_transform
from utils.transforms import test_transform

from utils.dataset_loader import create_dataloader

from models.resnet_model import create_model


# ======================
# Device
# ======================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")
SAVE_PATH = os.environ.get(
    "SAVE_PATH",
    "checkpoints/fruit_resnet18.pth"
)

# ======================
# Dataset Path
# ======================

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "FruitDataset"
)

TRAIN_PATH = os.path.join(
    DATASET_PATH,
    "train"
)

VAL_PATH = os.path.join(
    DATASET_PATH,
    "validation"
)


# ======================
# Load Dataset
# ======================

train_dataset, train_loader = create_dataloader(
    TRAIN_PATH,
    train_transform
)

val_dataset, val_loader = create_dataloader(
    VAL_PATH,
    test_transform,
    shuffle=False
)

print("\nClasses:")
print(train_dataset.classes)

num_classes = len(train_dataset.classes)

print(f"\nNumber of classes: {num_classes}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# ======================
# Model
# ======================

model = create_model(num_classes)

model.to(device)

print("\nModel Loaded Successfully")


# ======================
# Loss Function
# ======================

criterion = nn.CrossEntropyLoss()


# ======================
# Optimizer
# ======================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ======================
# Checkpoint Folder
# ======================

os.makedirs("checkpoints", exist_ok=True)


# ======================
# Training Config
# ======================

num_epochs = 30
best_accuracy = 0.0


# ======================
# Training Loop
# ======================

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # ======================
    # Validation
    # ======================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | "
        f"Loss: {avg_loss:.4f} | "
        f"Val Accuracy: {accuracy:.2f}%"
    )

    # ======================
    # Save Best Model
    # ======================

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            SAVE_PATH
        )

        print(
            f"Best model saved! "
            f"Accuracy = {accuracy:.2f}%"
        )


print("\nTraining Finished!")
print(
    f"Best Validation Accuracy: "
    f"{best_accuracy:.2f}%"
)