import random
import shutil
from pathlib import Path
from shutil import copy2


SOURCE_DIR = Path('Garbage_Image_Dataset')
TARGET_DIR = Path('data')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


def make_dir(path):
    """
    如果目录不存在则创建目录。
    """
    path.mkdir(parents=True, exist_ok=True)


def split_images(images):
    """
    按照预设比例切分单个类别下的图像路径列表。
    """
    total_num = len(images)
    train_end = int(total_num * TRAIN_RATIO)
    val_end = train_end + int(total_num * VAL_RATIO)
    return images[:train_end], images[train_end:val_end], images[val_end:]


def main():
    """
    将原始数据集按类别划分为 train / val / test 三部分。
    """
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f'未找到原始数据集目录: {SOURCE_DIR}')

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
        raise ValueError('训练集、验证集、测试集比例之和必须为 1。')

    random.seed(SEED)
    class_dirs = sorted([path for path in SOURCE_DIR.iterdir() if path.is_dir()])

    for split_name in ['train', 'val', 'test']:
        split_dir = TARGET_DIR / split_name
        if split_dir.exists():
            # 重新划分前先删除旧目录，避免重复复制历史文件。
            shutil.rmtree(split_dir)
        make_dir(split_dir)

    for class_dir in class_dirs:
        image_paths = [
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        random.shuffle(image_paths)

        train_images, val_images, test_images = split_images(image_paths)

        for split_name, split_images_list in {
            'train': train_images,
            'val': val_images,
            'test': test_images,
        }.items():
            target_class_dir = TARGET_DIR / split_name / class_dir.name
            make_dir(target_class_dir)

            for index, image_path in enumerate(split_images_list, start=1):
                copy2(image_path, target_class_dir / image_path.name)
                print(
                    f'[{split_name}] [{class_dir.name}] 正在处理 {index}/{len(split_images_list)}',
                    end='\r'
                )
        print()

    print('数据划分完成。')


if __name__ == '__main__':
    main()
