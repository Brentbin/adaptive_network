"""
实验一的网络实现 - 序列预测任务的特定适配
"""

import torch
from typing import Dict, Tuple
import sys

# 添加根目录到Python路径
root_path = "/Users/libin/SAGE/src/sage/models/adaptive_network"
sys.path.insert(0, str(root_path))

# 从根目录导入核心实现
from network import AdaptiveNetwork

class SequencePredictionNetwork(AdaptiveNetwork):
    """序列预测任务的自适应网络"""
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_nodes: int = 20,
                 max_thinking_depth: int = 50):
        # input_size 是输入窗口大小
        super().__init__(
            num_nodes=num_nodes,
            input_size=input_size,  # 输入窗口大小
            hidden_size=hidden_size,
            output_size=1,  # 序列预测输出单个值
            max_thinking_depth=max_thinking_depth
        )
        
        self.task_type = "sequence_prediction"
        self.input_size = input_size
        
    def predict(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """序列预测接口
        Args:
            sequence: 输入序列，形状为 [input_size] 或 [batch_size, input_size]
        Returns:
            prediction: 预测值，形状为 [1] 或 [batch_size]
            stats: 预测相关的统计信息
        """
        # 确保输入维度正确
        if sequence.dim() == 1:
            if sequence.size(0) != self.input_size:
                raise ValueError(f"Expected sequence length {self.input_size}, got {sequence.size(0)}")
            sequence = sequence.view(1, -1)  # [input_size] -> [1, input_size]
        elif sequence.dim() == 2:
            if sequence.size(1) != self.input_size:
                raise ValueError(f"Expected sequence length {self.input_size}, got {sequence.size(1)}")
        else:
            raise ValueError(f"Expected 1D or 2D input, got {sequence.dim()}D")
            
        # 使用基类的forward方法
        prediction = self.forward(sequence, self.task_type)
        
        # 收集预测相关的统计信息
        stats = {
            'thinking_depth': self.thinking_controller.current_depth,
            'confidence': self.current_state.confidence if self.current_state else 0.0,
            'active_nodes': len(self.current_state.active_nodes) if self.current_state else 0,
            'network_stats': self.get_network_stats()
        }
        
        return prediction.squeeze(), stats  # 移除不必要的维度
        
    def adapt_to_pattern(self, 
                        performance: float,
                        pattern_type: str) -> None:
        """适应新的序列模式"""
        # 更新网络
        self.update_network(performance)
        
        # 记录模式变化
        if self.current_state:
            self.current_state.task_type = f"{self.task_type}_{pattern_type}"
            
        # 调整思考控制器参数
        self.thinking_controller.adjust_parameters(performance) 