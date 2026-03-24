# garbage-classification

基于 PyTorch 的垃圾分类项目，适合在 Colab 环境中完成数据划分、训练、验证与测试。

## 项目文件说明

- `data_partitioning.py`：将原始数据集划分为训练集、验证集和测试集。
- `mean_std.py`：统计数据集的 RGB 三通道均值和标准差。
- `model.py`：构建用于垃圾分类的 ResNet18 模型。
- `model_train.py`：执行模型训练并保存最佳权重。
- `model_test.py`：在测试集上评估模型并输出混淆矩阵。

## 推荐使用流程

1. 准备原始数据集目录 `Garbage_Image_Dataset/类别名/*.jpg`。
2. 运行 `python data_partitioning.py` 划分数据。
3. 运行 `python mean_std.py` 统计均值和标准差，并同步更新到训练与测试脚本中。
4. 运行 `python model_train.py` 开始训练。
5. 运行 `python model_test.py` 查看测试结果。
