from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), # ĐÃ THÊM: Ngẫu nhiên lật ảnh theo chiều dọc
    transforms.RandomRotation(25),   # ĐÃ SỬA: Tăng góc xoay tối đa lên 25 độ để đa dạng góc nhìn
    
    # ĐÃ THÊM: Biến đổi ngẫu nhiên độ sáng, độ tương phản, độ bão hòa màu để chống nhiễu ánh sáng thực tế
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Tập test_transform GIỮ NGUYÊN (Không được thêm Augmentation vào tập test/val)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])