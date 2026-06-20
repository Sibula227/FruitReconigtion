import torch
import os

# Đường dẫn tới file .pth bạn đang có
file_path = r"model_fruit_8classes.pth"  

if not os.path.exists(file_path):
    print(f"❌ LỖI: Không tìm thấy file '{file_path}' ở thư mục hiện tại!")
    print(f"Các file đang có ở đây là: {os.listdir('.')}")
else:
    # Tải từ điển trọng số lên CPU
    weights = torch.load(file_path, map_location="cpu", weights_only=True)

    # GIẢI PHÁP TỰ ĐỘNG: Kiểm tra xem mô hình dùng cấu trúc cũ hay mới để lấy đúng khóa
    fc_key = None
    if 'fc.weight' in weights:
        fc_key = 'fc.weight'      # Cấu trúc cũ (không Dropout)
    elif 'fc.1.weight' in weights:
        fc_key = 'fc.1.weight'    # Cấu trúc mới nâng cấp (có Dropout)

    # Tiến hành phân tích nếu tìm thấy tầng phân loại cuối
    if fc_key is not None:
        fc_shape = weights[fc_key].shape
        print(f"\n🟢 Đã tìm thấy cấu trúc lớp phân loại cuối: '{fc_key}'")
        print(f"Kích thước lớp cuối (Shape): {fc_shape}")
        
        # Kiểm tra số lượng class đầu ra
        num_classes = fc_shape[0]
        if num_classes == 10:
            print("=> KẾT LUẬN: Đây là file CŨ (nhận diện 10 loại trái cây). XÓA NGAY!")
        elif num_classes == 8:
            print("=> KẾT LUẬN: Đây là file MỚI CHUẨN XỊN (nhận diện 8 loại trái cây + chống Overfitting)!")
        else:
            print(f"=> KẾT LUẬN: File này nhận diện {num_classes} loại trái cây.")
    else:
        print("❌ LỖI: File .pth này có cấu trúc lạ, không tìm thấy tầng phân loại fc.")
        print("5 khóa cuối cùng trong file để bạn kiểm tra là:")
        for k in list(weights.keys())[-5:]:
            print(f"  - {k}")