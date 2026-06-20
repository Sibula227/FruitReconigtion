from torchvision import models
import torch.nn as nn

def create_model(num_classes):

    # Tải mô hình ResNet18 với trọng số pre-trained mặc định
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # ĐÃ ĐƯA VỀ CHUẨN: Sử dụng một tầng tuyến tính (Linear) đơn lẻ cho lớp phân loại cuối
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model