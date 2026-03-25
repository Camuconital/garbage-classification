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
    plot_confusion_matrix(conf_matrix, class_names)
