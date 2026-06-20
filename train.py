import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # Thư viện để vẽ đồ thị

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

# ĐÃ SỬA: Đổi từ "validation" thành "valid" để khớp chính xác với Dataset mới của bạn
VAL_PATH = os.path.join(
    DATASET_PATH,
    "valid"
)

TEST_PATH = os.path.join(
    DATASET_PATH,
    "test"
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

test_dataset, test_loader = create_dataloader(
    TEST_PATH,
    test_transform,
    shuffle=False
)

print("\nClasses:")
print(train_dataset.classes)

num_classes = len(train_dataset.classes)

print(f"\nNumber of classes: {num_classes}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")

# ======================
# Model
# ======================

model = create_model(num_classes)

model.to(device)

print("\nModel Loaded Successfully")

# ======================
# Loss Function
# ======================
# ĐÃ CẬP NHẬT: Giữ lại Label Smoothing chống học vẹt, tự động thích ứng với số lượng lớp mới
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ======================
# Optimizer
# ======================
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001,
    weight_decay=1e-4  
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

# Khởi tạo các list để lưu lịch sử phục vụ vẽ biểu đồ
train_losses = []
val_losses = []       
test_losses = []      
train_accuracies = []  
val_accuracies = []
test_accuracies = []  

# ======================
# Training Loop
# ======================

for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total

    # ======================
    # Validation
    # ======================

    model.eval()

    correct = 0
    total = 0
    running_val_loss = 0.0  

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            val_loss_batch = criterion(outputs, labels)
            running_val_loss += val_loss_batch.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_val_loss = running_val_loss / len(val_loader)  

    # ======================
    # Test Evaluation
    # ======================
    test_correct = 0
    test_total = 0
    running_test_loss = 0.0  

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            test_loss_batch = criterion(outputs, labels)
            running_test_loss += test_loss_batch.item()

            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    avg_test_loss = running_test_loss / len(test_loader)  

    # Lưu lại giá trị của epoch này vào list lịch sử
    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)  
    test_losses.append(avg_test_loss)  
    train_accuracies.append(train_accuracy)
    val_accuracies.append(accuracy)
    test_accuracies.append(test_accuracy)  

    # In đầy đủ log của cả 3 tập dữ liệu để theo dõi song song
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | "
        f"Loss -> Train: {avg_loss:.4f}, Val: {avg_val_loss:.4f}, Test: {avg_test_loss:.4f} | "
        f"Acc -> Train: {train_accuracy:.2f}%, Val: {accuracy:.2f}%, Test: {test_accuracy:.2f}%"
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
        print(f"Best model saved! Accuracy = {accuracy:.2f}%")


print("\nTraining Finished!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

# ======================
# Plot Training History
# ======================
print("\nSaving Training History Plot...")

plt.figure(figsize=(14, 5))

# Đồ thị 1: Hiển thị song hành Train Loss, Val Loss và Test Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='red', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss', color='magenta', marker='o') 
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', color='green', marker='s') 
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss per Epoch (Train / Val / Test)')
plt.grid(True)
plt.legend()

# Đồ thị 2: Hiển thị đồng thời cả đường Train Accuracy, Validation Accuracy và Test Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy', color='orange', marker='o')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='blue', marker='o')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', color='green', marker='s') 
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Epoch (Train / Val / Test)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("Saved as 'training_history.png'.")