"""工具函数"""

import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
import sys

root_path = "/Users/libin/SAGE/src/sage/models/adaptive_network"
sys.path.append(str(root_path))
from experiments.exp2.config import CONFIG

def load_data() -> Dict:
    """加载实验数据"""
    data_path = CONFIG['data_dir'] / 'processed'
    
    data_dict = {}
    for task in CONFIG['tasks']:
        data = np.load(data_path / f"{task}_data.npz")
        data_dict.update({
            f"{task}_X_train": torch.FloatTensor(data['X_train']),
            f"{task}_y_train": torch.FloatTensor(data['y_train']),
            f"{task}_X_test": torch.FloatTensor(data['X_test']),
            f"{task}_y_test": torch.FloatTensor(data['y_test'])
        })
    
    # 加载参数
    with open(data_path / "params.json", 'r') as f:
        params = json.load(f)
    data_dict.update(params)
    
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

def analyze_node_specialization(node_activities: np.ndarray,
                              task_labels: List[str]) -> Tuple[np.ndarray, float]:
    """分析节点专门化程度
    
    Args:
        node_activities: 形状为 (num_nodes, num_tasks) 的节点活跃度矩阵
        task_labels: 任务标签列表
    
    Returns:
        specialization_matrix: 节点专门化矩阵
        specialization_score: 整体专门化得分
    """
    # 计算每个节点对每个任务的相对活跃度
    activity_sum = node_activities.sum(axis=1, keepdims=True)
    specialization_matrix = node_activities / (activity_sum + 1e-10)
    
    # 计算专门化得分 (使用Gini系数)
    n = len(task_labels)
    gini_scores = np.zeros(len(node_activities))
    for i in range(len(node_activities)):
        sorted_activities = np.sort(specialization_matrix[i])
        cumsum = np.cumsum(sorted_activities)
        gini_scores[i] = 1 - 2 * np.sum(cumsum) / (n * cumsum[-1])
    
    specialization_score = np.mean(gini_scores)
    
    return specialization_matrix, specialization_score

def plot_specialization_heatmap(specialization_matrix: np.ndarray,
                              task_labels: List[str],
                              save_path: Path = None):
    """绘制节点专门化热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(specialization_matrix, 
                xticklabels=task_labels,
                yticklabels=[f"Node {i+1}" for i in range(len(specialization_matrix))],
                cmap='YlOrRd')
    plt.title('Node Specialization Heatmap')
    plt.xlabel('Tasks')
    plt.ylabel('Nodes')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_training_log(train_stats: Dict,
                     eval_stats: Dict,
                     task_name: str,
                     epoch: int,
                     phase: str = 'pretrain'):
    """保存训练日志"""
    log_file = CONFIG['log_dir'] / f"{task_name}_{phase}_training.jsonl"
    
    log_entry = {
        'epoch': epoch,
        'phase': phase,
        'task': task_name,
        'train': train_stats,
        'eval': eval_stats
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n') 