"""
实验一的工具函数
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch
from typing import Dict

def load_data() -> Dict:
    """加载实验数据"""
    data_path = Path("../data/processed")
    
    data_dict = {}
    for pattern in ['arithmetic', 'geometric', 'multiplier']:
        data = np.load(data_path / f"{pattern}_data.npz")
        data_dict.update({
            f"{pattern}_X_train": torch.FloatTensor(data['X_train']),
            f"{pattern}_y_train": torch.FloatTensor(data['y_train']),
            f"{pattern}_X_test": torch.FloatTensor(data['X_test']),
            f"{pattern}_y_test": torch.FloatTensor(data['y_test'])
        })
    
    # 加载参数
    with open(data_path / "params.json", 'r') as f:
        params = json.load(f)
    data_dict['input_window'] = params['input_window']
    
    return data_dict

def calculate_metrics(predictions: torch.Tensor, 
                     targets: torch.Tensor) -> Dict:
    """计算评估指标"""
    mse = torch.nn.functional.mse_loss(predictions, targets).item()
    mae = torch.nn.functional.l1_loss(predictions, targets).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse)
    }

def save_training_log(train_stats: Dict,
                     eval_stats: Dict,
                     pattern_type: str,
                     epoch: int) -> None:
    """保存训练日志"""
    log_path = Path("../results/logs")
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_path / f"{pattern_type}_training.jsonl"
    
    # 处理训练统计信息
    metrics = []
    for m in train_stats['metrics']:
        metric = {}
        for k, v in m.items():
            if k == 'network_stats':
                # 网络统计信息已经是字典格式
                metric[k] = v
            elif isinstance(v, (int, float, str, bool)):
                metric[k] = v
            elif isinstance(v, (list, tuple)):
                metric[k] = list(v)
            elif isinstance(v, dict):
                metric[k] = {str(dk): dv for dk, dv in v.items()}
            else:
                # 其他类型转换为字符串
                metric[k] = str(v)
        metrics.append(metric)
    
    # 创建日志条目
    log_entry = {
        'epoch': epoch,
        'train': {
            'avg_loss': train_stats['avg_loss'],
            'metrics': metrics
        },
        'eval': {
            'avg_loss': eval_stats['avg_loss'],
            # 不保存预测结果，因为太大了
            'num_predictions': len(eval_stats['predictions'])
        },
        'pattern': pattern_type
    }
    
    # 保存日志
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def visualize():
    """可视化函数"""
    # TODO: 实现可视化
    pass 