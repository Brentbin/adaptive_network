"""实验配置"""

from pathlib import Path

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
    'batch_size': 32,
    
    # 路径配置
    'data_dir': Path('../data'),
    'results_dir': Path('../results'),
    'model_dir': Path('../results/models'),
}

# 创建必要的目录
for path in [CONFIG['data_dir'], CONFIG['results_dir'], CONFIG['model_dir']]:
    path.mkdir(parents=True, exist_ok=True) 