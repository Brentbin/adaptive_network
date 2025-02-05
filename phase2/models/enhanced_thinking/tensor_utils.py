"""张量维度自适应工具

提供一组工具函数，用于处理张量维度的自适应变换。
"""

import torch
from typing import Tuple, List, Optional

class TensorAdapter:
    """张量维度自适应工具类"""
    
    @staticmethod
    def ensure_batch_dim(x: torch.Tensor, target_dims: int = 2) -> Tuple[torch.Tensor, List[int]]:
        """确保张量具有指定的维度数
        
        如果维度数不足，在前面添加维度；如果维度数过多，在适当位置压缩维度。
        
        Args:
            x: 输入张量
            target_dims: 目标维度数
            
        Returns:
            处理后的张量和维度变换记录
        """
        original_shape = list(x.shape)
        current_dims = x.dim()
        
        if current_dims < target_dims:
            # 在前面添加维度
            for _ in range(target_dims - current_dims):
                x = x.unsqueeze(0)
        elif current_dims > target_dims:
            # 压缩多余维度
            dims_to_reduce = current_dims - target_dims
            for i in range(dims_to_reduce):
                # 选择要压缩的维度
                dim_to_reduce = i
                x = torch.mean(x, dim=dim_to_reduce)
                
        return x, original_shape
    
    @staticmethod
    def restore_shape(x: torch.Tensor, original_shape: List[int]) -> torch.Tensor:
        """恢复张量的原始形状
        
        Args:
            x: 输入张量
            original_shape: 原始形状
            
        Returns:
            恢复形状后的张量
        """
        target_dims = len(original_shape)
        current_dims = x.dim()
        
        if current_dims < target_dims:
            # 扩展维度
            for _ in range(target_dims - current_dims):
                x = x.unsqueeze(-1)
        elif current_dims > target_dims:
            # 压缩维度
            dims_to_reduce = current_dims - target_dims
            for _ in range(dims_to_reduce):
                x = x.squeeze(0)
                
        # 调整各个维度的大小
        return x.expand(*original_shape)
    
    @staticmethod
    def match_dimensions(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """使两个张量的维度匹配
        
        Args:
            x: 第一个张量
            y: 第二个张量
            
        Returns:
            维度匹配后的两个张量
        """
        x_dims = x.dim()
        y_dims = y.dim()
        
        if x_dims == y_dims:
            return x, y
            
        # 将维度较少的张量扩展到与另一个张量相同的维度
        if x_dims < y_dims:
            for _ in range(y_dims - x_dims):
                x = x.unsqueeze(0)
            # 扩展维度大小
            x = x.expand(*y.shape)
        else:
            for _ in range(x_dims - y_dims):
                y = y.unsqueeze(0)
            # 扩展维度大小
            y = y.expand(*x.shape)
            
        return x, y
    
    @staticmethod
    def apply_attention(query: torch.Tensor,
                       key: torch.Tensor,
                       value: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用注意力机制，自动处理维度
        
        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            mask: 掩码张量（可选）
            
        Returns:
            注意力输出和注意力权重
        """
        # 确保所有输入都是3D: [batch_size, sequence_length, hidden_size]
        query, _ = TensorAdapter.ensure_batch_dim(query, 3)
        key, _ = TensorAdapter.ensure_batch_dim(key, 3)
        value, _ = TensorAdapter.ensure_batch_dim(value, 3)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float32))
        
        # 应用掩码（如果有）
        if mask is not None:
            mask, _ = TensorAdapter.ensure_batch_dim(mask, attention_scores.dim())
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 应用注意力权重
        attention_output = torch.matmul(attention_weights, value)
        
        return attention_output, attention_weights
    
    @staticmethod
    def concat_features(features: List[torch.Tensor], dim: int = -1) -> torch.Tensor:
        """拼接特征，自动处理维度
        
        Args:
            features: 特征张量列表
            dim: 拼接维度
            
        Returns:
            拼接后的特征张量
        """
        if not features:
            return torch.tensor([])
            
        # 找出最大维度数和最大batch size
        max_dims = max(f.dim() for f in features)
        max_batch_size = max(f.size(0) if f.dim() > 0 else 1 for f in features)
        
        # 将所有特征扩展到相同维度
        aligned_features = []
        for feature in features:
            # 确保至少是2D
            if feature.dim() == 0:
                feature = feature.unsqueeze(0).unsqueeze(0)
            elif feature.dim() == 1:
                feature = feature.unsqueeze(0)
                
            # 扩展到相同的batch size
            if feature.size(0) == 1 and max_batch_size > 1:
                feature = feature.expand(max_batch_size, *feature.shape[1:])
                
            # 扩展到相同的维度数
            while feature.dim() < max_dims:
                if dim < 0:
                    # 在最后添加维度
                    feature = feature.unsqueeze(-1)
                else:
                    # 在指定维度之后添加维度
                    feature = feature.unsqueeze(dim + 1)
                    
            aligned_features.append(feature)
            
        # 拼接特征
        return torch.cat(aligned_features, dim=dim)
    
    @staticmethod
    def apply_mlp(x: torch.Tensor,
                 weight: torch.Tensor,
                 bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """应用多层感知机，自动处理维度
        
        Args:
            x: 输入张量
            weight: 权重张量
            bias: 偏置张量（可选）
            
        Returns:
            MLP输出
        """
        # 确保输入至少是2D: [batch_size, features]
        x, original_shape = TensorAdapter.ensure_batch_dim(x, 2)
        
        # 应用线性变换
        output = torch.matmul(x, weight.t())
        if bias is not None:
            output = output + bias
            
        return output
    
    @staticmethod
    def batch_average(x: torch.Tensor, dims: Optional[List[int]] = None) -> torch.Tensor:
        """批量平均，自动处理维度
        
        Args:
            x: 输入张量
            dims: 要平均的维度列表（可选）
            
        Returns:
            平均后的张量
        """
        if dims is None:
            # 默认对所有维度求平均，除了最后一维
            dims = list(range(x.dim() - 1))
            
        # 对指定维度求平均
        for dim in sorted(dims, reverse=True):
            x = torch.mean(x, dim=dim, keepdim=True)
            
        return x
    
    @staticmethod
    def ensure_same_device(*tensors: torch.Tensor) -> List[torch.Tensor]:
        """确保所有张量在同一设备上
        
        Args:
            tensors: 张量列表
            
        Returns:
            在同一设备上的张量列表
        """
        if not tensors:
            return []
            
        # 使用第一个张量的设备
        target_device = tensors[0].device
        
        # 将所有张量移动到目标设备
        return [t.to(target_device) for t in tensors] 