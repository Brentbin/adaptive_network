"""实验二网络模型"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import sys

root_path = "/Users/libin/SAGE/src/sage/models/adaptive_network"
sys.path.insert(0, str(root_path))
from network import AdaptiveNetwork
from experiments.exp2.config import CONFIG

class MultiTaskSequenceNetwork(AdaptiveNetwork):
    """多任务序列预测网络，继承自基础自适应网络"""
    def __init__(self):
        # 保存网络参数
        self.hidden_size = CONFIG['hidden_size']
        self.num_nodes = CONFIG['num_nodes']
        
        # 初始化基础自适应网络
        super().__init__(
            input_size=CONFIG['input_window'],  # 输入窗口大小
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,  # 输出隐藏状态
            num_nodes=self.num_nodes,
            max_thinking_depth=CONFIG['max_thinking_depth']
        )
        
        # 最小连接数作为类属性
        self.min_connections = CONFIG['min_connections']
        
        # 任务特定的输出层
        self.task_outputs = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(CONFIG['input_window'], self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1)
            )
            for task in CONFIG['tasks']
        })
        
        # 节点活跃度记录
        self.node_activities = {
            task: np.zeros(self.num_nodes)
            for task in CONFIG['tasks']
        }
        
        # 专门化矩阵
        self.specialization_matrix = np.zeros((self.num_nodes, len(CONFIG['tasks'])))
        
        # 连接矩阵
        self.connections = np.ones((self.num_nodes, self.num_nodes)) * 0.5
    
    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        # 确保输入是二维的 [batch_size, seq_length]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [seq_length] -> [1, seq_length]
        
        # 使用基类的序列处理机制
        hidden = super().forward(x, task)  # 传递task作为task_type参数
        
        # 记录节点活跃度
        if self.current_state is not None:
            # 使用当前活跃节点更新活跃度
            for node_id in self.current_state.active_nodes:
                self.node_activities[task][node_id] += 1
            
            # 更新专门化矩阵
            total_activity = sum(self.node_activities[t] for t in CONFIG['tasks'])
            for i, t in enumerate(CONFIG['tasks']):
                self.specialization_matrix[:, i] = (
                    self.node_activities[t] / (total_activity + 1e-10)
                )
        
        # 使用任务特定的输出层进行预测
        output = self.task_outputs[task](x)  # 直接使用输入序列
        return output.squeeze(-1)  # 移除最后一个维度，使其与目标形状匹配
    
    def get_specialization_matrix(self) -> np.ndarray:
        """获取节点专门化矩阵"""
        return self.specialization_matrix.copy()
    
    def update_connections(self):
        """基于专门化程度更新节点间连接"""
        if self.current_state is None:
            return
            
        # 额外的基于专门化的连接调整
        with torch.no_grad():
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    # 计算节点专门化的相似度
                    similarity = np.dot(
                        self.specialization_matrix[i],
                        self.specialization_matrix[j]
                    )
                    # 根据相似度调整连接权重
                    self.connections[i, j] *= (1 + 0.1 * similarity)
    
    def get_active_nodes(self, task: str = None) -> List[int]:
        """重写获取活跃节点的方法，考虑任务专门化"""
        if task is None:
            # 如果没有指定任务，返回所有节点
            return list(range(self.num_nodes))
        
        # 获取对当前任务专门化程度最高的节点
        task_idx = CONFIG['tasks'].index(task)
        specialization_scores = self.specialization_matrix[:, task_idx]
        
        # 选择专门化程度最高的节点
        top_k = max(self.min_connections, int(0.3 * self.num_nodes))
        active_nodes = np.argsort(specialization_scores)[-top_k:]
        
        return active_nodes.tolist()
    
    def count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 