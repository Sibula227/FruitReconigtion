import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.transforms import train_transform, test_transform
from utils.dataset_loader import create_dataloader
from models.resnet_model import create_model

# ==========================================
# 1. Khởi tạo thiết bị và đường dẫn dữ liệu
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Thiết bị chạy Tuning: {device}")

DATASET_PATH = os.environ.get("DATASET_PATH", "FruitDataset")
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "valid") # Khớp với thư mục "valid" mới của bạn

# Nạp dữ liệu một lần duy nhất để tối ưu bộ nhớ
train_dataset, train_loader = create_dataloader(TRAIN_PATH, train_transform)
val_dataset, val_loader = create_dataloader(VAL_PATH, test_transform, shuffle=False)
num_classes = len(train_dataset.classes)

# ==========================================
# 2. Định nghĩa không gian tham số cần Tunning (Grid Search)
# ==========================================
learning_rates = [1e-4, 5e-5, 1e-5]   # Các mức tốc độ học cần thử nghiệm
weight_decays = [1e-4, 1e-3]          # Các mức phạt trọng số lớn chống Overfitting
tuning_epochs = 5                     # Chạy thử 5 epoch cho mỗi cấu hình để tìm xu hướng

best_val_acc = 0.0
best_config = {}

print("\n=== BẮT ĐẦU QUÁ TRÌNH HYPERPARAMETER TUNING ===")
print(f"Tổng số tổ hợp cấu hình cần chạy thử: {len(learning_rates) * len(weight_decays)}\n")

# Bảng lưu kết quả lịch sử tuning
tuning_results = []

for lr in learning_rates:
    for wd in weight_decays:
        print(f"╔════════════════════════════════════════════════════════════╗")
        print(f" ⏳ Đang chạy thử nghiệm: Learning Rate = {lr} | Weight Decay = {wd}")
        print(f"╚════════════════════════════════════════════════════════════╝")
        
        # Khởi tạo lại mô hình trống cho mỗi lượt thử nghiệm
        model = create_model(num_classes).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        
        # Vòng lặp train thử nghiệm ngắn hạn
        for epoch in range(tuning_epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
            # Đánh giá nhanh trên tập Validation sau khi kết thúc lượt chạy thử
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = 100 * correct / total
            print(f"   -> Epoch [{epoch+1}/{tuning_epochs}] | Validation Accuracy: {val_acc:.2f}%")
        
        # Lưu kết quả lượt chạy này vào bảng thống kê
        tuning_results.append({"lr": lr, "wd": wd, "final_val_acc": val_acc})
        
        # Cập nhật nếu đây là cấu hình tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = {"lr": lr, "wd": wd}

# ==========================================
# 3. In báo cáo kết quả Tunning tổng hợp
# ==========================================
print("\n" + "="*60)
print("             KẾT QUẢ TỔNG HỢP QUÁ TRÌNH TUNING             ")
print("="*60)
print(f" STT |   Learning Rate   |   Weight Decay   |  Val Accuracy ")
print("-"*60)
for idx, res in enumerate(tuning_results):
    print(f"  {idx+1:02d} |      {res['lr']:<12} |      {res['wd']:<12} |     {res['final_val_acc']:.2f}%")
print("="*60)

print(f"🏆 CẤU HÌNH TỐI ƯU NHẤT CHO DATASET MỚI:")
print(f" 👉 Learning Rate (Tốc độ học): {best_config['lr']}")
print(f" 👉 Weight Decay (L2 Regularization): {best_config['wd']}")
print(f" 📊 Độ chính xác Validation đạt được: {best_val_acc:.2f}%")
print("="*60)
print("\n💡 Bạn hãy lấy cặp số tối ưu này điền ngược vào file train.py để train chính thức 30 Epochs nhé!")