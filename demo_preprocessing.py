import cv2
import matplotlib.pyplot as plt
import os

# Đọc thử 1 ảnh bất kỳ trong dataset (Sửa đường dẫn tới 1 ảnh cụ thể trên máy bạn)
img_path = 'Dataset_New/train/apple/1_100.jpg' # Cập nhật lại đường dẫn thực tế của bạn
img = cv2.imread(img_path)

if img is not None:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Histogram Equalization (Cân bằng sáng)
    img_eq = cv2.equalizeHist(img_gray)

    # 2. Canny Edge Detection (Nhận diện biên cạnh)
    edges = cv2.Canny(img_gray, 100, 200)

    # Trực quan hóa
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_rgb); axes[0].set_title('Original RGB')
    axes[1].imshow(img_gray, cmap='gray'); axes[1].set_title('Grayscale')
    axes[2].imshow(img_eq, cmap='gray'); axes[2].set_title('Histogram Equalization')
    axes[3].imshow(edges, cmap='gray'); axes[3].set_title('Canny Edges')

    for ax in axes: ax.axis('off')
    plt.show()
else:
    print("Vui lòng cập nhật lại đường dẫn ảnh hợp lệ!")