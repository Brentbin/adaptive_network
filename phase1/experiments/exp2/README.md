# 实验二：节点专门化测试

## 实验目的
验证自适应神经网络中节点的专门化能力，观察网络如何在处理多任务序列预测时形成功能区。

## 文件结构
```
exp2/
├── README.md              # 本文档，实验说明
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
├── src/                  # 源代码
│   ├── data_gen.py      # 数据生成器
│   ├── network.py       # 网络模型定义
│   ├── train.py         # 训练脚本
│   └── utils.py         # 工具函数
├── notebooks/           # Jupyter notebooks
│   └── analysis.ipynb   # 数据分析和可视化
└── results/            # 实验结果
    ├── models/         # 保存的模型
    ├── logs/          # 训练日志
    └── figures/       # 生成的图表
```

## 实验说明

### 1. 序列预测任务设计
实验包含三种序列预测任务：
- 加法序列：预测序列中下一个数（前两个数的和）
- 乘法序列：预测序列中下一个数（前两个数的积）
- 模式序列：预测不同模式（正弦、二次、指数）下的下一个值

### 2. 网络结构
- 继承自基础自适应网络
- 输入：序列窗口（默认5个时间步）
- 输出：下一个时间步的预测值
- 初始节点数：50
- 动态连接形成
- 节点专门化机制
- 任务特定的输出层

### 3. 训练过程
- 阶段1：单任务预训练（每个任务独立训练）
- 阶段2：多任务联合训练（交替训练不同任务）
- 阶段3：新任务适应测试（测试模型对分布变化的适应能力）

### 4. 评估指标
- 预测准确性：MSE、MAE、RMSE
- 节点专门化度：基于Gini系数的专门化得分
- 功能区形成：节点活跃度热力图
- 适应性指标：任务切换效率

## 实验参数

### 数据参数
- 样本数量：1000个序列
- 序列长度：10个时间步
- 输入窗口：5个时间步
- 训练集比例：80%
- 噪声水平：0.01

### 模型参数
- 隐藏层大小：128
- 节点数量：50
- 最小连接数：3
- 最大思考深度：50

### 训练参数
- 预训练轮数：5
- 联合训练轮数：10
- 适应测试轮数：5
- 学习率：0.001
- 批次大小：32

## 使用说明

1. 环境准备：
```bash
conda activate SAGE_dev
```

2. 生成数据：
```bash
python src/data_gen.py
```

3. 运行实验：
```bash
python src/train.py
```

4. 分析结果：
打开 notebooks/analysis.ipynb 进行分析 