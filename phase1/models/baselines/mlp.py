"""基准MLP模型"""

import torch
import torch.nn as nn
from typing import Optional

class MLPNetwork(nn.Module):
    """基准MLP网络，用于序列预测任务"""
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int = 1,
                 num_layers: int = 2):
        super().__init__()
        
        # 网络参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 构建MLP层
        layers = []
        current_size = input_size
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1)
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, task: Optional[str] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_length] 或 [batch_size, seq_length, features]
            task: 任务标识符（为了与自适应网络接口保持一致，但MLP不使用此参数）
        
        Returns:
            预测输出，形状为 [batch_size]
        """
        # 如果输入是3D张量，将其展平
        if x.dim() == 3:
            batch_size, seq_length, features = x.shape
            x = x.reshape(batch_size, seq_length * features)
        
        return self.network(x).squeeze(-1)  # 移除最后一个维度，使输出形状为[batch_size]
    
    def count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 