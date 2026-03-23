# 垃圾分类项目

一个基于 **PyTorch** 的垃圾图像分类毕业设计项目。仓库中实现了数据集划分、数据集均值方差统计、自定义 ResNet18 模型训练以及测试推理流程，目标是对垃圾图片进行多类别分类。根据当前代码，模型最终输出 **10 个类别**。 

## 项目简介

本项目围绕垃圾图像分类任务展开，主要包含以下内容：

- 使用 `data_partitioning.py` 将原始图片数据集按训练集 / 测试集进行划分。
- 使用 `mean_std.py` 统计数据集的 RGB 通道均值与方差，用于图像归一化。
- 使用 `model.py` 定义基于残差块的自定义 `ResNet18` 网络。
- 使用 `model_train.py` 完成训练、验证以及最优模型权重保存。
- 使用 `model_test.py` 加载训练好的模型并在测试集上评估准确率。

## 项目结构

```text
.
├── README.md                # 项目说明文档
├── data_partitioning.py     # 原始数据集划分脚本
├── mean_std.py              # 计算数据集均值/方差
├── model.py                 # 自定义 ResNet18 网络结构
├── model_train.py           # 模型训练与验证脚本
└── model_test.py            # 模型测试脚本
```

## 任务类别

从 `model.py` 中最后一层全连接层的输出维度来看，当前任务是 **10 分类问题**。

`model_test.py` 中的注释给出了一组示例类别名称：

- battery
- biological
- cardboard
- clothes
- glass
- metal
- paper
- plastic
- shoes
- trash

> 注意：实际类别名称以你的数据集目录名为准。只要训练集与测试集目录结构一致，程序会自动根据文件夹名称建立标签映射。

## 数据集准备

代码默认使用以下目录约定：

### 1. 原始数据集目录

在项目根目录下准备原始数据集文件夹：

```text
Garbage_Image_Dataset/
├── class_1/
├── class_2/
├── ...
└── class_10/
```

每个子目录代表一个类别，目录中存放该类别的图片。

### 2. 划分后的数据目录

运行数据划分脚本后，会自动生成：

```text
data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

其中：

- `data/train`：训练数据
- `data/test`：测试数据

## 环境依赖

建议使用 Python 3.9 及以上版本，并安装以下依赖：

```bash
pip install torch torchvision torchsummary numpy pandas matplotlib pillow
```

如果你使用 GPU 训练，请根据自己的 CUDA 版本安装对应的 PyTorch。

## 使用流程

### 第一步：划分数据集

将原始数据集按 9:1 划分为训练集和测试集：

```bash
python data_partitioning.py
```

脚本默认：

- 原始数据目录为 `Garbage_Image_Dataset`
- 划分比例为测试集 `10%`
- 输出到 `data/train` 和 `data/test`

### 第二步：计算数据集均值和方差

为了保证图像归一化参数与数据集匹配，可以先统计数据集的 RGB 均值和方差：

```bash
python mean_std.py
```

运行后会输出类似：

- `Mean: [...]`
- `Variance: [...]`

你可以将统计结果填写到训练与测试脚本中的 `transforms.Normalize(...)` 中。

> 当前代码中已经写入了一组均值与方差参数，如果你更换了数据集，建议重新计算。

### 第三步：训练模型

执行训练脚本：

```bash
python model_train.py
```

训练脚本默认行为：

- 从 `data/train` 加载图片数据。
- 在脚本内部再按照 `8:2` 随机划分为训练集和验证集。
- 使用 `Adam` 优化器，学习率为 `0.001`。
- 批大小为 `32`。
- 默认训练 `30` 个 epoch。
- 在验证集上保存效果最好的模型参数到 `best_model.pth`。

训练过程中会输出：

- 每轮训练损失
- 每轮训练准确率
- 每轮验证损失
- 每轮验证准确率
- 总耗时

训练完成后，还会绘制训练/验证损失与准确率曲线。

### 第四步：测试模型

在测试集上评估模型：

```bash
python model_test.py
```

测试脚本会：

- 加载 `best_model.pth`
- 从 `data/test` 读取测试数据
- 输出最终测试准确率

## 模型说明

项目在 `model.py` 中实现了一个基于残差结构的自定义 `ResNet18`：

- 包含基础卷积层、批归一化层和最大池化层。
- 使用多个残差块提取图像特征。
- 最后通过自适应平均池化和全连接层输出分类结果。

模型输入尺寸默认为：

- `3 x 224 x 224`

如果需要查看网络结构摘要，可运行：

```bash
python model.py
```

## 代码实现说明

### `data_partitioning.py`

- 扫描 `Garbage_Image_Dataset` 下的类别文件夹。
- 自动创建 `data/train` 与 `data/test` 目录。
- 按随机采样方式划分训练集与测试集。

### `mean_std.py`

- 遍历整个原始图像数据集。
- 统计 RGB 三通道均值与方差。
- 用于后续 `Normalize` 预处理。

### `model_train.py`

- 使用 `ImageFolder` 加载训练数据。
- 执行 `Resize`、`ToTensor` 和 `Normalize`。
- 完成训练循环与验证循环。
- 保存验证准确率最高的模型参数。
- 返回训练记录并绘制曲线。

### `model_test.py`

- 加载测试数据。
- 加载训练好的模型权重。
- 在测试集上计算分类准确率。

## 注意事项

- 请先准备好数据集，否则训练和测试脚本会因为找不到目录而报错。
- `best_model.pth` 需要先通过训练生成，否则测试脚本无法直接运行。
- 当前训练脚本中使用的是固定归一化参数；若数据集发生变化，建议重新统计。
- `data_partitioning.py` 为随机划分脚本，多次运行可能生成不同的数据分布。
- 若在 Windows 与 Linux 之间切换，建议统一使用当前代码中的正斜杠路径写法。

## 可改进方向

如果后续你想继续完善这个毕业设计项目，可以考虑：

- 增加 `requirements.txt` 或 `environment.yml`，方便环境复现。
- 增加随机种子设置，提升实验可复现性。
- 将训练、验证、测试路径配置改成命令行参数。
- 保存更完整的训练日志，例如 loss / acc 到 CSV 文件。
- 增加单张图片预测脚本与可视化展示。
- 尝试使用官方预训练模型提升分类效果。

## 适用场景

该项目适合作为以下用途的基础代码：

- 垃圾分类课程设计 / 毕业设计
- 图像分类入门练习
- PyTorch 自定义 ResNet 网络结构学习
- 小规模多分类图像识别实验

## License

当前仓库未声明开源许可证。如需开源发布，建议补充 `LICENSE` 文件。
