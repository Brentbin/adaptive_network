"""实验配置"""

from pathlib import Path

CONFIG = {
    # 数据参数
    'num_samples': 1000,
    'test_ratio': 0.2,
    'seq_length': 10,        # 序列长度
    'input_window': 5,       # 输入窗口大小
    'noise_level': 0.01,
    
    # 任务参数
    'tasks': ['addition', 'multiplication', 'pattern'],
    'num_patterns': 3,  # 模式识别任务中的模式数量
    
    # 模型参数
    'hidden_size': 128,
    'num_nodes': 50,
    'max_thinking_depth': 50,
    'min_connections': 3,
    
    # 训练参数
    'num_epochs': {
        'pretrain': 5,    # 单任务预训练轮数
        'joint': 10,      # 多任务联合训练轮数
        'adapt': 5        # 新任务适应轮数
    },
    'learning_rate': 0.001,
    'batch_size': 32,
    
    # 路径配置
    'data_dir': Path('./data'),
    'results_dir': Path('./results'),
    'model_dir': Path('./results/models'),
    'log_dir': Path('./results/logs'),
}

# 创建必要的目录
for path in [CONFIG['data_dir'], CONFIG['results_dir'], CONFIG['model_dir'], CONFIG['log_dir']]:
    path.mkdir(parents=True, exist_ok=True) 