from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import build_model


TEST_DIR = Path('data/test')
MODEL_SAVE_PATH = 'best_model.pth'
BATCH_SIZE = 32
NUM_WORKERS = 2
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
TARGET_CLASS_A = 'glass'
TARGET_CLASS_B = 'plastic'
MISCLASSIFIED_SAVE_PATH = Path('misclassified_glass_plastic.txt')
MAX_VIS_SAMPLES_PER_DIRECTION = 12


def build_test_transform(mean, std):
    """
    构建测试阶段使用的图像预处理流程。
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def create_test_dataloader(mean, std):
    """
    加载测试集并构建 DataLoader。
    """
    test_dataset = ImageFolder(TEST_DIR, transform=build_test_transform(mean, std))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return test_dataset, test_loader


def evaluate_model(model, dataloader, device):
    """
    在测试集上评估模型，并返回预测结果与真实标签。
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def get_class_index(class_names, target_name):
    """
    根据类别名（忽略大小写）查找对应的类别索引。
    """
    normalized = [name.lower() for name in class_names]
    if target_name.lower() not in normalized:
        return None
    return normalized.index(target_name.lower())


def collect_bidirectional_confusions(all_labels, all_preds, test_dataset, class_names, class_a, class_b):
    """
    收集两类之间的双向混淆样本路径。
    """
    class_a_idx = get_class_index(class_names, class_a)
    class_b_idx = get_class_index(class_names, class_b)
    if class_a_idx is None or class_b_idx is None:
        return None, None

    a_to_b = []
    b_to_a = []
    for sample_idx, (label, pred) in enumerate(zip(all_labels, all_preds)):
        path, _ = test_dataset.samples[sample_idx]
        if label == class_a_idx and pred == class_b_idx:
            a_to_b.append(path)
        elif label == class_b_idx and pred == class_a_idx:
            b_to_a.append(path)
    return a_to_b, b_to_a


def print_and_save_confusions(class_a, class_b, a_to_b, b_to_a, save_path):
    """
    打印并保存双向混淆样本路径，便于后续人工查看。
    """
    print()
    print(f'真实为 {class_a} 但预测为 {class_b} 的样本数: {len(a_to_b)}')
    for path in a_to_b:
        print(path)

    print()
    print(f'真实为 {class_b} 但预测为 {class_a} 的样本数: {len(b_to_a)}')
    for path in b_to_a:
        print(path)

    with save_path.open('w', encoding='utf-8') as output_file:
        output_file.write(f'真实为 {class_a} 但预测为 {class_b} 的样本数: {len(a_to_b)}\n')
        for path in a_to_b:
            output_file.write(f'{path}\n')

        output_file.write('\n')
        output_file.write(f'真实为 {class_b} 但预测为 {class_a} 的样本数: {len(b_to_a)}\n')
        for path in b_to_a:
            output_file.write(f'{path}\n')

    print()
    print(f'双向混淆样本路径已保存到: {save_path}')


def plot_misclassified_samples(image_paths, title, max_samples):
    """
    可视化误分类样本图像，便于观察错误模式。
    """
    if not image_paths:
        print(f'{title}: 无样本可视化。')
        return

    selected_paths = image_paths[:max_samples]
    num_images = len(selected_paths)
    num_cols = min(4, num_images)
    num_rows = int(np.ceil(num_images / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(title, fontsize=14)

    for axis_index, axis in enumerate(axes):
        axis.axis('off')
        if axis_index >= num_images:
            continue

        image_path = selected_paths[axis_index]
        image = plt.imread(image_path)
        axis.imshow(image)
        axis.set_title(Path(image_path).name, fontsize=9)

    plt.tight_layout()
    plt.show()


def print_per_class_accuracy(conf_matrix, class_names):
    """
    输出每个类别的分类准确率。
    """
    print('各类别准确率:')
    for index, class_name in enumerate(class_names):
        class_total = conf_matrix[index].sum()
        class_correct = conf_matrix[index, index]
        class_acc = class_correct / class_total if class_total > 0 else 0.0
        print(f'{class_name}: {class_acc:.4f}')


def plot_confusion_matrix(conf_matrix, class_names):
    """
    绘制混淆矩阵热力图。
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location='cpu')
    class_names = checkpoint.get('class_names')
    mean = checkpoint.get('mean', DEFAULT_MEAN)
    std = checkpoint.get('std', DEFAULT_STD)

    if class_names is None:
        raise ValueError('模型文件中未找到类别名称信息，请重新训练并保存模型。')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    test_dataset, test_loader = create_test_dataloader(mean, std)
    print('测试集类别映射:', test_dataset.class_to_idx)
    print(f'测试集样本数: {len(test_dataset)}')

    all_labels, all_preds = evaluate_model(model, test_loader, device)
    test_acc = (all_labels == all_preds).mean()
    print(f'测试集整体准确率: {test_acc:.4f}')

    conf_matrix = confusion_matrix(all_labels, all_preds)
    print('混淆矩阵:')
    print(conf_matrix)
    print_per_class_accuracy(conf_matrix, class_names)

    a_to_b, b_to_a = collect_bidirectional_confusions(
        all_labels=all_labels,
        all_preds=all_preds,
        test_dataset=test_dataset,
        class_names=class_names,
        class_a=TARGET_CLASS_A,
        class_b=TARGET_CLASS_B,
    )
    if a_to_b is None or b_to_a is None:
        print(
            f'未找到目标类别，请确认类别名是否包含 "{TARGET_CLASS_A}" 与 "{TARGET_CLASS_B}"。'
        )
    else:
        print_and_save_confusions(
            class_a=TARGET_CLASS_A,
            class_b=TARGET_CLASS_B,
            a_to_b=a_to_b,
            b_to_a=b_to_a,
            save_path=MISCLASSIFIED_SAVE_PATH,
        )
        plot_misclassified_samples(
            image_paths=a_to_b,
            title=f'真实 {TARGET_CLASS_A} -> 预测 {TARGET_CLASS_B}',
            max_samples=MAX_VIS_SAMPLES_PER_DIRECTION,
        )
        plot_misclassified_samples(
            image_paths=b_to_a,
            title=f'真实 {TARGET_CLASS_B} -> 预测 {TARGET_CLASS_A}',
            max_samples=MAX_VIS_SAMPLES_PER_DIRECTION,
        )

    plot_confusion_matrix(conf_matrix, class_names)
