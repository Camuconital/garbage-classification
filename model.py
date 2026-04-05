import torch
from torch import nn
from torchvision import models


CLASS_NAMES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'mask', 'metal', 'paper', 'plastic', 'shoes', 'toothbrush', 'trash'
]


def build_model(num_classes=len(CLASS_NAMES), pretrained=True):
    """
    构建用于垃圾分类的 ResNet18 模型。

    参数:
        num_classes: 分类类别数量。
        pretrained: 是否加载 ImageNet 预训练权重。
    """
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    # 加载 torchvision 中已经实现好的 ResNet18。
    model = models.resnet18(weights=weights)

    # 将最后的全连接层替换为当前任务对应的分类层。
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print("模型输出形状:", output.shape)
