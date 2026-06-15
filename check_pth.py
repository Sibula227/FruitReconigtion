import torch

# Đường dẫn tới file .pth bạn đang có
file_path = r"checkpoints/model_fruit_8classes.pth" 

# Tải từ điển trọng số lên CPU
weights = torch.load(file_path, map_location="cpu", weights_only=True)

# Lấy kích thước của lớp cuối cùng (Fully Connected Layer)
fc_shape = weights['fc.weight'].shape

print(f"Kích thước lớp cuối: {fc_shape}")
if fc_shape[0] == 10:
    print("=> KẾT LUẬN: Đây là file CŨ (nhận diện 10 loại trái cây). XÓA NGAY!")
elif fc_shape[0] == 8:
    print("=> KẾT LUẬN: Đây là file MỚI (nhận diện 8 loại trái cây). CHUẨN RỒI!")
else:
    print(f"=> KẾT LUẬN: File này nhận diện {fc_shape[0]} loại trái cây.")