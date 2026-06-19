from torchvision import models
import torch.nn as nn

def create_model(num_classes):

    # Tải mô hình ResNet18 với trọng số pre-trained mặc định
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # ĐÃ CẬP NHẬT: Thay thế tầng fc đơn lẻ bằng một chuỗi tuần tự (Sequential)
    # bao gồm tầng Dropout và tầng Tuyến tính (Linear)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Ngẫu nhiên ngắt kết nối 50% nơ-ron trong quá trình huấn luyện nhằm chống overfitting
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model