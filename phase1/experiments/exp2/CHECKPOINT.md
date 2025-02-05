# 实验二断点记录

## 项目状态

### 1. 核心实现 (根目录)
- network.py: 自适应网络基类，包含网络结构和前向传播
- node.py: 自适应节点实现，包含专门化机制
- subgraph.py: 子图形成机制，负责动态选择活跃节点
- thinking.py: 思考控制器，管理思考深度和收敛

### 2. 实验二实现 (experiments/exp2/)
已完成文件：
- src/data_gen.py: 序列数据生成器，生成三种序列预测任务的数据
- src/network.py: MultiTaskSequenceNetwork 实现，继承自基础自适应网络
- src/train.py: 训练脚本，包含三个训练阶段
- src/utils.py: 工具函数，包含评估和可视化功能
- config.py: 实验参数配置
- run_experiment.sh: 实验运行脚本
- EXPERIMENT_LOG.md: 实验记录模板

### 3. 目录结构
```
experiments/exp2/
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
    'num_samples': 1000,
    'test_ratio': 0.2,
    'seq_length': 10,        # 序列长度
    'input_window': 5,       # 输入窗口大小
    'noise_level': 0.01,
    
    # 任务参数
    'tasks': ['addition', 'multiplication', 'pattern'],
    'num_patterns': 3,
    
    # 模型参数
    'hidden_size': 128,
    'num_nodes': 50,
    'max_thinking_depth': 50,
    'min_connections': 3,
    
    # 训练参数
    'num_epochs': {
        'pretrain': 5,
        'joint': 10,
        'adapt': 5
    }
}
```

### 5. 待完成工作
1. 数据生成和验证
   - 运行 data_gen.py 生成三种序列数据
   - 验证序列生成的正确性
   - 检查序列分布和特性
   - 验证输入窗口的处理

2. 模型训练
   - 完成预训练阶段（每个序列任务）
   - 执行联合训练（多任务学习）
   - 进行适应性测试（分布变化）
   - 记录训练过程和指标

3. 结果分析
   - 评估序列预测性能
   - 分析节点专门化程度
   - 可视化功能区形成过程
   - 总结实验发现

### 6. 关键实现细节
1. MultiTaskSequenceNetwork:
   - 继承自基础自适应网络
   - 序列处理机制
   - 多任务输出层
   - 专门化追踪

2. 训练流程:
   - 序列预测训练
   - 多任务切换策略
   - 专门化度量和记录
   - 适应性评估方法

3. 评估指标:
   - 序列预测准确性
   - 专门化程度指标
   - 适应性指标

## 下一步计划
1. 运行序列数据生成器
2. 验证序列预测功能
3. 开始预训练阶段
4. 监控节点专门化

## 注意事项
1. 确保使用 SAGE_dev conda环境
2. 检查序列生成的随机种子设置
3. 监控序列预测性能
4. 保存关键节点的训练日志 