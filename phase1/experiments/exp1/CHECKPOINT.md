# 实验一断点记录

## 项目状态

### 1. 核心实现 (根目录)
- network.py: 自适应网络基类，包含网络结构和前向传播
- node.py: 自适应节点实现，包含专门化机制
- subgraph.py: 子图形成机制，负责动态选择活跃节点
- thinking.py: 思考控制器，管理思考深度和收敛

### 2. 实验一实现 (experiments/exp1/)
已完成文件：
- src/data_gen.py: 序列数据生成器，生成三种模式的序列
- src/network.py: SequencePredictionNetwork 实现，继承自 AdaptiveNetwork
- src/train.py: 训练脚本，包含训练循环和评估
- src/utils.py: 工具函数，包含数据加载和指标计算
- config.py: 实验参数配置
- run_experiment.sh: 实验运行脚本
- EXPERIMENT_LOG.md: 实验记录模板

### 3. 目录结构
```
experiments/exp1/
├── data/
│   ├── raw/           # 原始数据目录
│   └── processed/     # 处理后的数据目录
├── results/
│   ├── models/        # 模型保存目录
│   ├── logs/         # 训练日志目录
│   └── figures/      # 可视化结果目录
├── src/              # 源代码目录
│   ├── data_gen.py
│   ├── network.py
│   ├── train.py
│   └── utils.py
├── notebooks/        # Jupyter notebooks
│   └── analysis.ipynb
├── config.py         # 实验配置
├── run_experiment.sh # 运行脚本
└── EXPERIMENT_LOG.md # 实验记录
```

### 4. 实验配置
```python
CONFIG = {
    # 数据参数
    'seq_length': 10,
    'input_window': 5,
    'train_samples': 1000,
    'test_samples': 200,
    'noise_level': 0.01,
    
    # 模型参数
    'hidden_size': 64,
    'num_nodes': 20,
    'max_thinking_depth': 50,
    
    # 训练参数
    'num_epochs': 10,
    'learning_rate': 0.001,
    'batch_size': 32
}
```

### 5. 待完成工作
1. 数据生成和验证
   - 运行 data_gen.py 生成三种序列数据
   - 验证数据格式和质量

2. 模型训练
   - 按顺序训练三种序列模式
   - 记录适应过程和性能变化

3. 结果分析
   - 评估模型在不同序列上的表现
   - 分析网络的适应性能力
   - 生成可视化结果

### 6. 关键实现细节
1. SequencePredictionNetwork:
   - 继承自 AdaptiveNetwork
   - 添加了序列预测特定的接口
   - 包含模式适应机制

2. 训练流程:
   - 依次训练三种序列模式
   - 每种模式训练10个epoch
   - 记录适应过程和性能指标

3. 评估指标:
   - MSE、MAE、RMSE
   - 适应时间
   - 网络结构变化

## 下一步计划
1. 运行数据生成器
2. 验证生成的数据格式和质量
3. 开始训练过程
4. 记录实验结果

## 注意事项
1. 确保使用 SAGE_dev conda环境
2. 检查数据生成的随机种子设置
3. 监控网络的适应过程
4. 保存关键节点的训练日志 