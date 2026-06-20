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
# CẬP NHẬT: Thêm weight_decay=1e-4 để thực hiện kỹ thuật L2 Regularization phạt trọng số lớn
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
val_losses = []       # SẼ ĐƯỢC LƯU: List lưu lịch sử Validation Loss
train_accuracies = []  
val_accuracies = []

# ======================
# Training Loop
# ======================

for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    
    # Khởi tạo biến đếm số ảnh đoán đúng và tổng số ảnh của tập Train ở đầu mỗi Epoch
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
        
        # Tính toán số lượng dự đoán đúng thực tế trên tập Train trong khi chạy
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    
    # Tính toán chính xác phần trăm Train Accuracy sau khi kết thúc vòng lặp epoch
    train_accuracy = 100 * train_correct / train_total

    # ======================
    # Validation
    # ======================

    model.eval()

    correct = 0
    total = 0
    running_val_loss = 0.0  # ĐÃ THÊM MỚI: Biến cộng dồn loss của tập Validation

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            # ĐÃ THÊM MỚI: Tính toán hàm lỗ (loss) cho từng batch trên tập Validation
            val_loss_batch = criterion(outputs, labels)
            running_val_loss += val_loss_batch.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_val_loss = running_val_loss / len(val_loader)  # ĐÃ THÊM MỚI: Tính loss trung bình tập Val

    # Lưu lại giá trị loss và accuracy của epoch này vào list lịch sử
    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)  # ĐÃ THÊM MỚI
    train_accuracies.append(train_accuracy)
    val_accuracies.append(accuracy)

    # ĐÃ CẬP NHẬT: In thêm chỉ số Val Loss ra màn hình log để tiện theo dõi
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | "
        f"Train Loss: {avg_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
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
        print(f"Best model saved! Accuracy = {accuracy:.2f}%")


print("\nTraining Finished!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

# ======================
# Plot Training History
# ======================
print("\nSaving Training History Plot...")

plt.figure(figsize=(12, 5))

# Đồ thị 1: ĐẦ LÀM MỚI - Hiển thị song song cả đường Train Loss và Val Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='red', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss', color='magenta', marker='o') # Màu hồng tím cho Val Loss
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss per Epoch')
plt.grid(True)
plt.legend()

# Đồ thị 2: Hiển thị đồng thời cả đường Train Accuracy và Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy', color='orange', marker='o')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='blue', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training & Validation Accuracy per Epoch')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("Saved as 'training_history.png'.")