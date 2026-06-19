import os
import torch
import torch.nn as nn
from utils.dataset_loader import create_dataloader
from utils.transforms import test_transform  # Dùng bộ transform sạch (không lật/xoay) để đánh giá khách quan ảnh gốc
from models.resnet_model import create_model

# ======================
# 1. Khởi tạo Thiết bị (Device)
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị đang sử dụng để kiểm tra: {device}")

# ======================
# 2. Thiết lập Đường dẫn Dataset
# ======================
DATASET_ROOT = os.environ.get("DATASET_PATH", "FruitDataset")
TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
VAL_PATH = os.path.join(DATASET_ROOT, "validation")

# ĐIỂM QUAN TRỌNG: Đánh giá mô hình tĩnh thì dùng test_transform cho cả hai tập 
# để xem khả năng nhận diện trên ảnh gốc thực tế chuẩn xác nhất.
train_dataset, train_loader = create_dataloader(TRAIN_PATH, test_transform, shuffle=False)
val_dataset, val_loader = create_dataloader(VAL_PATH, test_transform, shuffle=False)

num_classes = len(train_dataset.classes)
print(f"\n[Thông tin] Số lượng Class: {num_classes}")
print(f"[Thông tin] Tổng số ảnh tập Train: {len(train_dataset)}")
print(f"[Thông tin] Tổng số ảnh tập Validation: {len(val_dataset)}")

# ======================
# 3. Nạp Mô Hình Từ Checkpoint Tốt Nhất
# ======================
model = create_model(num_classes)
model_path = "checkpoints/fruit_resnet18.pth"

if not os.path.exists(model_path):
    print(f"❌ LỖI: Không tìm thấy file trọng số tại đường dẫn: {model_path}")
    print("Vui lòng kiểm tra lại file .pth của bạn.")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # Chuyển sang chế độ đánh giá
print("🟢 [OK] Đã tải thành công trọng số mô hình tốt nhất!")

# ======================
# 4. Hàm Quét và Tính Toán Chỉ Số
# ======================
criterion = nn.CrossEntropyLoss()

def calculate_metrics(dataloader):
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    return accuracy, avg_loss

# Tiến hành quét dữ liệu
print("\n🔄 Đang tính toán hiệu năng trên tập Train (vui lòng đợi)...")
train_acc, train_loss = calculate_metrics(train_loader)

print("🔄 Đang tính toán hiệu năng trên tập Validation (vui lòng đợi)...")
val_acc, val_loss = calculate_metrics(val_loader)

# ======================
# 5. In Bảng Đối Chiếu Kết Quả & Chẩn Đoán Overfitting
# ======================
print("\n" + "="*55)
print("           BẢNG ĐỐI CHIẾU CHỈ SỐ KIỂM TRA OVERFITTING          ")
print("="*55)
print(f" Tập dữ liệu   |  Độ chính xác (Accuracy)  |   Hàm Lỗ (Loss)   ")
print("-"*55)
print(f" TRAIN         |          {train_acc:.2f}%          |       {train_loss:.4f}      ")
print(f" VALIDATION    |          {val_acc:.2f}%          |       {val_loss:.4f}      ")
print("="*55)

# Thuật toán tự động chẩn đoán sức khỏe mô hình dựa trên độ chênh lệch (Gap)
gap_accuracy = train_acc - val_acc

print("\n🤖 [CHẨN ĐOÁN TỪ HỆ THỐNG]:")
if gap_accuracy > 12:
    print(f"⚠️ CẢNH BÁO: Mô hình đang bị OVERFITTING NẶNG!")
    print(f"-> Độ chính xác tập Train vượt trội tập Val tới {gap_accuracy:.2f}%.")
    print("-> Hướng xử lý: Bạn cần bổ sung Dropout vào Model hoặc tăng cường thêm kỹ thuật lật/xoay ở transforms.")
elif gap_accuracy > 5:
    print(f"⚡ CHÚ Ý: Mô hình có dấu hiệu OVERFITTING NHẸ (Chênh lệch: {gap_accuracy:.2f}%).")
    print("-> Kết quả này ở mức chấp nhận được đối với các bài toán phân loại ảnh cơ bản.")
elif gap_accuracy < -5:
    print(f"❓ BẤT THƯỜNG: Validation Accuracy lại cao hơn hẳn Train Accuracy ({abs(gap_accuracy):.2f}%).")
    print("-> Hãy kiểm tra lại xem tập Validation có vô tình bị trùng lặp ảnh hay không.")
else:
    print(f"✅ HOÀN HẢO: Mô hình cực kỳ khỏe mạnh và KHÔNG BỊ OVERFITTING!")
    print(f"-> Độ chênh lệch giữa hai tập vô cùng lý tưởng, chỉ khoảng {abs(gap_accuracy):.2f}%.")
print("="*55 + "\n")