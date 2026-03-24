from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
DATASET_DIR = Path('Garbage_Image_Dataset')


def calculate_mean_std(folder_path):
    """
    计算图像数据集的 RGB 三通道均值和标准差。
    """
    image_paths = [
        path for path in folder_path.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        raise FileNotFoundError(f'在 {folder_path} 下没有找到可用图片文件。')

    pixel_count = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_squared = np.zeros(3, dtype=np.float64)

    for image_path in image_paths:
        # 统一将图像转换为 RGB，避免灰度图或带透明通道图像影响统计。
        image = Image.open(image_path).convert('RGB')
        image_array = np.asarray(image, dtype=np.float32) / 255.0

        height, width, _ = image_array.shape
        pixel_count += height * width

        # 统计每个通道的像素和以及平方和，用于后续计算均值与标准差。
        channel_sum += image_array.sum(axis=(0, 1))
        channel_sum_squared += (image_array ** 2).sum(axis=(0, 1))

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_sum_squared / pixel_count - mean ** 2)
    return mean, std, len(image_paths)


if __name__ == '__main__':
    mean, std, image_count = calculate_mean_std(DATASET_DIR)
    print(f'统计图片数量: {image_count}')
    print('mean =', mean.tolist())
    print('std =', std.tolist())
