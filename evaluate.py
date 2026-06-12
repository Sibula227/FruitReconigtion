import os
import torch
import matplotlib.pyplot as plt
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
# Confusion Matrix Visualization
# ======================
print("\nSaving Confusion Matrix Image...\n")
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=test_dataset.classes, 
            yticklabels=test_dataset.classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png') # Lưu ảnh để cho vào báo cáo
plt.show()

# ======================
# Visualizing Sample Predictions
# ======================
print("\nSaving Sample Predictions...\n")
# Lấy 1 batch từ test_loader để vẽ
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

# Chuyển tensor về numpy để hiển thị
images = images.cpu().numpy()
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    # Denormalize ảnh
    img = np.transpose(images[i], (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    true_cls = test_dataset.classes[labels[i].item()]
    pred_cls = test_dataset.classes[preds[i].item()]
    
    ax.imshow(img)
    ax.set_title(f"True: {true_cls}\nPred: {pred_cls}", 
                 color=("green" if true_cls == pred_cls else "red"))
    ax.axis('off')
plt.tight_layout()
plt.savefig('sample_predictions.png')
plt.show()