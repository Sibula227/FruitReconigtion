import os
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from utils.dataset_loader import create_dataloader
from utils.transforms import test_transform
from models.resnet_model import create_model


# ======================
# Device
# ======================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")


# ======================
# Dataset Path
# ======================

DATASET_ROOT = os.environ.get(
    "DATASET_PATH",
    "FruitDataset"
)

TEST_PATH = os.path.join(
    DATASET_ROOT,
    "test"
)


# ======================
# Load Test Dataset
# ======================

test_dataset, test_loader = create_dataloader(
    TEST_PATH,
    test_transform,
    shuffle=False
)

print("\nClasses:")
print(test_dataset.classes)

num_classes = len(test_dataset.classes)

print(f"\nNumber of classes: {num_classes}")
print(f"Test images: {len(test_dataset)}")


# ======================
# Load Model
# ======================

model = create_model(num_classes)

model.load_state_dict(
    torch.load(
        "checkpoints/fruit_resnet18.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("\nModel Loaded Successfully")


# ======================
# Evaluation
# ======================

correct = 0
total = 0

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

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

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )

accuracy = 100 * correct / total

print("\n==========================")
print(f"Test Accuracy: {accuracy:.2f}%")
print("==========================\n")


# ======================
# Classification Report
# ======================

print("Classification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes
    )
)


# ======================
# Confusion Matrix
# ======================

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)