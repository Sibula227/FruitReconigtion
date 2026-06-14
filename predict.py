import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os

from utils.transforms import test_transform
from models.resnet_model import create_model

# ======================
# Cấu hình cơ bản
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Danh sách 8 class của bạn (Phải đúng thứ tự chữ cái như lúc train)
classes = ['apple', 'banana', 'cucumber', 'grape', 'mango', 'orange', 'pear', 'tomato']
num_classes = len(classes)

# ======================
# Tải Mô Hình Đã Train
# ======================
model = create_model(num_classes)
model_path = "checkpoints/fruit_resnet18.pth"

if not os.path.exists(model_path):
    print(f"Lỗi: Không tìm thấy file weights tại {model_path}!")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval() # Chuyển sang chế độ dự đoán
print("Đã tải mô hình thành công!")

# ======================
# Hàm Dự Đoán 1 Bức Ảnh
# ======================
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {image_path}")
        return

    # 1. Đọc ảnh và áp dụng các bước tiền xử lý giống hệt lúc test
    image = Image.open(image_path).convert('RGB')
    image_tensor = test_transform(image).unsqueeze(0).to(device) # Thêm dimension cho batch_size = 1

    # 2. Đưa qua mô hình dự đoán
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1) # Chuyển output thành % xác suất
        
        # Lấy class có xác suất cao nhất
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    predicted_class = classes[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    # 3. Hiển thị kết quả trực quan
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    
    # In chữ kết quả lên hình
    title_color = 'green' if confidence_score > 80 else 'red'
    plt.title(f"Dự đoán: {predicted_class.upper()}\nĐộ tự tin: {confidence_score:.2f}%", 
              color=title_color, fontsize=14, fontweight='bold')
    
    plt.savefig('prediction_result.png')
    print(f"\n=> Kết quả: {predicted_class.upper()} ({confidence_score:.2f}%)")
    plt.show()

# ======================
# Chạy Thử Nghiệm
# ======================
# Bạn hãy tải 1 bức ảnh trái cây từ mạng về, up lên Kaggle và sửa tên file ở dưới đây:
IMAGE_TO_TEST = "test_image.jpg" # <--- THAY ĐỔI ĐƯỜNG DẪN ẢNH CỦA BẠN VÀO ĐÂY

print(f"\nĐang tiến hành dự đoán ảnh: {IMAGE_TO_TEST}...")
predict_image(IMAGE_TO_TEST)