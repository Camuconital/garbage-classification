import copy
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import build_model


TRAIN_DIR = Path('data/train')
VAL_DIR = Path('data/val')
MODEL_SAVE_PATH = 'best_model.pth'
RESULT_SAVE_PATH = 'train_history.csv'
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42

# 这里的均值和标准差需要根据 mean_std.py 重新统计得到的结果进行替换。
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    """
    设置随机种子，尽量保证每次训练结果一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_transforms():
    """
    分别返回训练集和验证集的图像预处理流程。
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_transform, val_transform



def create_dataloaders():
    """
    加载训练集与验证集，并构建 DataLoader。
    """
    train_transform, val_transform = get_transforms()
    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = ImageFolder(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_dataset, val_dataset, train_loader, val_loader



def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    执行一轮训练，返回该轮训练损失和准确率。
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    sample_count = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        sample_count += inputs.size(0)

    epoch_loss = running_loss / sample_count
    epoch_acc = running_corrects / sample_count
    return epoch_loss, epoch_acc



def validate_one_epoch(model, dataloader, criterion, device):
    """
    在验证集上评估模型，返回验证损失和准确率。
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    sample_count = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            sample_count += inputs.size(0)

    epoch_loss = running_loss / sample_count
    epoch_acc = running_corrects / sample_count
    return epoch_loss, epoch_acc



def train_model_process(model, train_loader, val_loader, class_names, num_epochs=NUM_EPOCHS):
    """
    执行完整训练流程，并保存验证集表现最好的模型。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = []
    since = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 20)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
        })

        print(f'训练集损失: {train_loss:.4f} 训练集准确率: {train_acc:.4f}')
        print(f'验证集损失: {val_loss:.4f} 验证集准确率: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'model_state_dict': best_model_wts,
                'best_acc': best_acc,
                'class_names': class_names,
                'mean': MEAN,
                'std': STD,
            }, MODEL_SAVE_PATH)
            print(f'已更新最优模型，当前最佳验证准确率: {best_acc:.4f}')

        elapsed = time.time() - since
        print(f'累计训练时间: {elapsed // 60:.0f}m {elapsed % 60:.0f}s\n')

    model.load_state_dict(best_model_wts)
    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULT_SAVE_PATH, index=False)
    return history_df



def plot_acc_loss(train_process):
    """
    绘制训练集与验证集的损失和准确率曲线。
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process['train_loss'], 'ro-', label='训练损失')
    plt.plot(train_process['epoch'], train_process['val_loss'], 'bs-', label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process['train_acc'], 'ro-', label='训练准确率')
    plt.plot(train_process['epoch'], train_process['val_acc'], 'bs-', label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    set_seed(SEED)
    train_dataset, val_dataset, train_loader, val_loader = create_dataloaders()

    print('训练集类别映射:', train_dataset.class_to_idx)
    print(f'训练集样本数: {len(train_dataset)}')
    print(f'验证集样本数: {len(val_dataset)}')

    model = build_model(num_classes=len(train_dataset.classes), pretrained=True)
    history_df = train_model_process(model, train_loader, val_loader, train_dataset.classes, num_epochs=NUM_EPOCHS)
    plot_acc_loss(history_df)
