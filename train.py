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
# Loss Function
# ======================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ======================
# Checkpoint Folder
# ======================
os.makedirs("checkpoints", exist_ok=True)


# ============================================================
# GIAI ĐOẠN 1: HYPERPARAMETER TUNING (PHẠM VI: 5 EPOCHS)
# ============================================================
tuning_lr_list = [1e-4, 5e-5, 1e-5]
tuning_wd_list = [1e-4, 1e-3]
tuning_epochs = 5

best_tuning_acc = -1.0
best_lr = 1e-4
best_wd = 1e-4
best_model_state = None
best_optimizer_state = None

# Từ điển lưu lịch sử của cấu hình chiến thắng để vẽ biểu đồ
best_histories = {
    'train_losses': [], 'val_losses': [], 'test_losses': [],
    'train_accuracies': [], 'val_accuracies': [], 'test_accuracies': []
}

print("\n" + "="*60)
print(" ⏳ BẮT ĐẦU GIAI ĐOẠN TUNING TỰ ĐỘNG (5 EPOCHS / CẤU HÌNH) ")
print("="*60)

for lr in tuning_lr_list:
    for wd in tuning_wd_list:
        print(f"\n▶️ Đang thử nghiệm cấu hình: LR = {lr} | Weight Decay = {wd}")
        
        # Khởi tạo mô hình và bộ tối ưu tạm thời cho lượt thử nghiệm
        temp_model = create_model(num_classes).to(device)
        temp_optimizer = optim.Adam(temp_model.parameters(), lr=lr, weight_decay=wd)
        
        # List tạm thời lưu lịch sử của lượt chạy này
        t_train_losses, t_val_losses, t_test_losses = [], [], []
        t_train_accs, t_val_accs, t_test_accs = [], [], []
        
        for t_epoch in range(tuning_epochs):
            # Train 1 epoch nhanh
            temp_model.train()
            running_loss = 0.0
            t_correct, t_total = 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                temp_optimizer.zero_grad()
                outputs = temp_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                temp_optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                t_total += labels.size(0)
                t_correct += (predicted == labels).sum().item()
                
            avg_loss = running_loss / len(train_loader)
            train_acc = 100 * t_correct / t_total
            
            # Validation nhanh
            temp_model.eval()
            val_correct, val_total, running_val_loss = 0, 0, 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = temp_model(images)
                    running_val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            avg_val_loss = running_val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # Test nhanh
            test_correct, test_total, running_test_loss = 0, 0, 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = temp_model(images)
                    running_test_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            avg_test_loss = running_test_loss / len(test_loader)
            test_acc = 100 * test_correct / test_total
            
            # Lưu log tạm thời
            t_train_losses.append(avg_loss)
            t_val_losses.append(avg_val_loss)
            t_test_losses.append(avg_test_loss)
            t_train_accs.append(train_acc)
            t_val_accs.append(val_acc)
            t_test_accs.append(test_acc)
            
            print(f"   Epoch [{t_epoch + 1}/{tuning_epochs}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
        # Kiểm tra xem cấu hình này có tối ưu nhất tại epoch số 5 không
        if val_acc > best_tuning_acc:
            best_tuning_acc = val_acc
            best_lr = lr
            best_wd = wd
            best_model_state = temp_model.state_dict()
            best_optimizer_state = temp_optimizer.state_dict()
            best_histories = {
                'train_losses': list(t_train_losses),
                'val_losses': list(t_val_losses),
                'test_losses': list(t_test_losses),
                'train_accuracies': list(t_train_accs),
                'val_accuracies': list(t_val_accs),
                'test_accuracies': list(t_test_accs)
            }

print("\n" + "="*60)
print(f" 🏆 TUNING THÀNH CÔNG! Cấu hình tối ưu nhất: LR={best_lr} | WD={best_wd}")
print(f" 📊 Đạt Accuracy tại Epoch 5: {best_tuning_acc:.2f}%")
print("="*60)


# ============================================================
# GIAI ĐOẠN 2: TIẾP TỤC HUẤN LUYỆN CHÍNH THỨC (30 EPOCHS)
# ============================================================
num_main_epochs = 30
total_epochs = tuning_epochs + num_main_epochs # Tổng cộng 35 Epochs

# Khởi tạo mô hình chính thức và nạp lại trạng thái của cấu hình thắng cuộc
model = create_model(num_classes).to(device)
model.load_state_dict(best_model_state)

optimizer = optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_wd)
optimizer.load_state_dict(best_optimizer_state)

# Kế thừa dữ liệu lịch sử 5 epoch đầu từ Tuning sang danh sách chính thức
train_losses = list(best_histories['train_losses'])
val_losses = list(best_histories['val_losses'])
test_losses = list(best_histories['test_losses'])
train_accuracies = list(best_histories['train_accuracies'])
val_accuracies = list(best_histories['val_accuracies'])
test_accuracies = list(best_histories['test_accuracies'])

best_accuracy = best_tuning_acc

print(f"\n🚀 Tiếp tục huấn luyện mô hình chính thức từ Epoch {tuning_epochs + 1} đến {total_epochs}...")

# Vòng lặp chạy tiếp từ epoch số 5 đến epoch số 35
for epoch in range(tuning_epochs, total_epochs):

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

    # Validation
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

    # Test Evaluation
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

    # Thêm giá trị mới vào chuỗi lịch sử liên tục
    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)  
    test_losses.append(avg_test_loss)  
    train_accuracies.append(train_accuracy)
    val_accuracies.append(accuracy)
    test_accuracies.append(test_accuracy)  

    # In log tiến trình kế thừa (Epoch 6/35 -> 35/35)
    print(
        f"Epoch [{epoch + 1}/{total_epochs}] | "
        f"Loss -> Train: {avg_loss:.4f}, Val: {avg_val_loss:.4f}, Test: {avg_test_loss:.4f} | "
        f"Acc -> Train: {train_accuracy:.2f}%, Val: {accuracy:.2f}%, Test: {test_accuracy:.2f}%"
    )

    # Save Best Model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(
            model.state_dict(),
            SAVE_PATH
        )
        print(f"🔥 Tìm thấy mô hình tốt hơn! Đã lưu checkpoint với Accuracy = {accuracy:.2f}%")


print("\nTraining Finished!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

# ======================
# Plot Training History (VẼ FULL 35 EPOCHS)
# ======================
print("\nSaving Training History Plot...")

plt.figure(figsize=(14, 5))

# Đồ thị 1: Biểu đồ Loss liên tục 35 Epochs
plt.subplot(1, 2, 1)
plt.plot(range(1, total_epochs + 1), train_losses, label='Train Loss', color='red', marker='o')
plt.plot(range(1, total_epochs + 1), val_losses, label='Val Loss', color='magenta', marker='o') 
plt.plot(range(1, total_epochs + 1), test_losses, label='Test Loss', color='green', marker='s') 
plt.axvline(x=5, color='gray', linestyle='--', label='End of Tuning') # Đường kẻ phân tách giai đoạn
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss per Epoch (Full 35 Epochs)')
plt.grid(True)
plt.legend()

# Đồ thị 2: Biểu đồ Accuracy liên tục 35 Epochs
plt.subplot(1, 2, 2)
plt.plot(range(1, total_epochs + 1), train_accuracies, label='Train Accuracy', color='orange', marker='o')
plt.plot(range(1, total_epochs + 1), val_accuracies, label='Validation Accuracy', color='blue', marker='o')
plt.plot(range(1, total_epochs + 1), test_accuracies, label='Test Accuracy', color='green', marker='s') 
plt.axvline(x=5, color='gray', linestyle='--', label='End of Tuning') # Đường kẻ phân tách giai đoạn
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Epoch (Full 35 Epochs)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("Saved as 'training_history.png'.")