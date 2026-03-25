import torch
from torch import nn
from torchvision import models


CLASS_NAMES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation 注意力模块。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def build_model(num_classes=10, pretrained=True, attention_type='se'):
    """
    构建用于垃圾分类的 ResNet18 模型。

    参数:
        num_classes: 分类类别数量。
        pretrained: 是否加载 ImageNet 预训练权重。
        attention_type: 注意力模块类型，支持 'se' 或 None。
    """
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    # 加载 torchvision 中已经实现好的 ResNet18。
    model = models.resnet18(weights=weights)

    # 在 ResNet18 的最后一个残差阶段后增加注意力模块。
    if attention_type == 'se':
        model.layer4.add_module('se_attention', SEModule(channels=512))
    elif attention_type is not None:
        raise ValueError(f"不支持的注意力类型: {attention_type}，请使用 'se' 或 None。")

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
