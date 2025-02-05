import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class ActivationPattern:
    task_type: str
    input_pattern: torch.Tensor
    output_pattern: torch.Tensor
    performance: float
    timestamp: float

class AdaptiveNode(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 output_size: int,
                 history_size: int = 1000):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.history_size = history_size
        
        # 核心神经网络 - 保持输入维度不变，只在最后一层转换到输出维度
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)  # 保持输入维度
        )
        
        # 最后的预测层
        self.predictor = nn.Linear(input_size, output_size)
        
        # 状态跟踪
        self.specialization = None
        self.activation_patterns: List[ActivationPattern] = []
        self.connection_strength: Dict[int, float] = {}
        self.activation_count = 0
        self.performance_history: List[float] = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，确保输入维度正确
        Args:
            x: 输入张量，形状为 [batch_size, input_size]
        Returns:
            output: 输出张量，形状为 [batch_size, output_size]
        """
        self.activation_count += 1
        
        # 确保输入是二维的 [batch_size, input_size]
        if x.dim() == 1:
            # 单个样本，添加批次维度
            x = x.view(1, -1)
        elif x.dim() > 2:
            # 高维输入，展平到二维
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            
        # 检查输入维度
        if x.size(-1) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.size(-1)}")
            
        # 特征提取（保持输入维度）
        features = self.feature_extractor(x)  # [batch_size, input_size]
        
        # 预测（转换到输出维度）
        output = self.predictor(features)  # [batch_size, output_size]
            
        return output
        
    def update_specialization(self, 
                            task_pattern: ActivationPattern) -> None:
        """更新节点的专门化方向"""
        self.activation_patterns.append(task_pattern)
        if len(self.activation_patterns) > self.history_size:
            self.activation_patterns.pop(0)
            
        self.specialization = self._analyze_patterns()
        
    def _analyze_patterns(self) -> Dict[str, float]:
        """分析历史激活模式，确定专门化方向"""
        if not self.activation_patterns:
            return {}
            
        task_performance = {}
        for pattern in self.activation_patterns:
            if pattern.task_type not in task_performance:
                task_performance[pattern.task_type] = []
            task_performance[pattern.task_type].append(pattern.performance)
            
        # 计算每种任务类型的平均性能
        specialization = {
            task: np.mean(perfs)
            for task, perfs in task_performance.items()
        }
        
        return specialization
        
    def update_connection(self, 
                         target_node_id: int, 
                         strength_delta: float) -> None:
        """更新与其他节点的连接强度"""
        current_strength = self.connection_strength.get(target_node_id, 0.0)
        self.connection_strength[target_node_id] = max(0.0, 
            min(1.0, current_strength + strength_delta)
        )
        
    def get_performance_stats(self) -> Dict[str, float]:
        """获取节点性能统计"""
        if not self.performance_history:
            return {'mean': 0.0, 'std': 0.0}
            
        return {
            'mean': np.mean(self.performance_history),
            'std': np.std(self.performance_history)
        }
        
    def reset_stats(self) -> None:
        """重置统计数据"""
        self.activation_count = 0
        self.performance_history.clear()
        
    def __repr__(self) -> str:
        return f"AdaptiveNode(spec={self.specialization}, act_count={self.activation_count})" 