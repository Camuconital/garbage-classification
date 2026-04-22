# garbage-classification

基于 PyTorch 的垃圾分类项目，适合在 Colab 环境中完成数据划分、训练、验证与测试。

## 项目文件说明

- `data_partitioning.py`：将原始数据集划分为训练集、验证集和测试集。
- `mean_std.py`：统计数据集的 RGB 三通道均值和标准差。（以弃用，直接官方预训练resnet的标准差和方差）
- `model.py`：构建用于垃圾分类的 ResNet18 模型。
- `model_train.py`：执行模型训练并保存最佳权重。
- `model_test.py`：在测试集上评估模型并输出混淆矩阵。
- `gradio_app.py`：启动可视化网页，上传图片后进行垃圾类别识别。

## 推荐使用流程

1. 准备原始数据集目录 `Garbage_Image_Dataset/类别名/*.jpg`。
2. 运行 `python data_partitioning.py` 划分数据。
3. 运行 `python mean_std.py` 统计均值和标准差，并同步更新到训练与测试脚本中（弃用）
4. 运行 `python model_train.py` 开始训练。
5. 运行 `python model_test.py` 查看测试结果。

## Gradio 可视化界面使用说明

> 你在 Colab 训练得到的 `best_model.pth` 可以直接用于本界面，只要文件结构与当前训练脚本保存格式一致（包含 `model_state_dict`、`class_names`、`mean`、`std`）。

### 1) 安装依赖

```bash
pip install torch torchvision gradio pillow
```

### 2) 准备模型权重

将最佳权重放在仓库根目录，默认文件名为：

```text
best_model.pth
```

如果你的权重文件名不同，可以修改 `gradio_app.py` 里的 `DEFAULT_MODEL_PATH`。

### 3) 启动界面

```bash
python gradio_app.py
```

终端会输出一个本地地址（例如 `http://127.0.0.1:7860`），浏览器打开即可。

### 4) 页面操作

1. 点击“上传图片”选择垃圾图片。
2. 点击“开始识别”。
3. 右侧会显示：
   - 预测类别（含置信度）
   - Top-3 概率分布
4. 点击“清空”可重置输入和输出。

## 你还需要准备什么？

对于推理网页来说，**必须项只有一个**：训练好的模型权重（`best_model.pth`）。

可选项：

- 一批测试图片（用于手工验证效果）。
- 如果你后续想在网页里展示“类别中文名”，可以额外准备一个类别映射表（例如 `class_map.json`）。
