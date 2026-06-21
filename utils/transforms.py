from torchvision import transforms

# ĐÃ SỬA: Loại bỏ RandomHorizontalFlip, RandomRotation, ColorJitter...
# Do dataset mới đã được Augmentation sẵn, ta chỉ thực hiện tiền xử lý cơ bản.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Ép về kích thước chuẩn của ResNet18
    transforms.ToTensor(),         # Chuyển sang Tensor PyTorch [0, 1]
    transforms.Normalize(          # Chuẩn hóa ma trận màu ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Tập test_transform giữ nguyên như cũ
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])