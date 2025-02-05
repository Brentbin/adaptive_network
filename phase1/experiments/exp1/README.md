# 实验一：基础动态适应能力测试

## 文件结构
```
exp1/
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

### 1. 数据生成
- 生成三种不同规则的序列数据
- 每种规则包含训练集和测试集
- 数据格式：时间序列数据

### 2. 网络结构
- 初始节点数：20
- 最小连接度：3
- 动态节点生成
- 连接强度自适应

### 3. 训练过程
- 阶段1：初始规则训练
- 阶段2：规则变更（算术→几何）
- 阶段3：规则再变更（倍数变化）

### 4. 评估指标
- 适应时间统计
- 准确率变化曲线
- 网络结构演化分析

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