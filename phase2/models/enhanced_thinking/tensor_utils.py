"""张量工具模块

实现张量的转换、适配和优化功能。
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np

class TensorAdapter:
    """张量适配器
    
    负责:
    1. 张量格式转换
    2. 维度适配
    3. 特征变换
    4. 数据类型转换
    5. 设备迁移
    """
    
    def __init__(self):
        # 缓存常用的转换矩阵
        self.projection_cache = {}
        
        # 特征变换网络
        self.feature_transforms = {}
        
        # 设备映射
        self.device_map = {}
        
    def adapt_tensor(self,
                    tensor: torch.Tensor,
                    target_shape: Tuple[int, ...],
                    target_dtype: Optional[torch.dtype] = None,
                    target_device: Optional[torch.device] = None) -> torch.Tensor:
        """适配张量到目标规格
        
        维度处理规则:
        1. 如果输入是3维 [batch_size, 1, feature_size], 压缩到2维 [batch_size, feature_size]
        2. 如果输入是2维 [batch_size, feature_size], 保持不变
        3. 其他维度的输入会抛出异常
        
        Args:
            tensor: 输入张量
            target_shape: 目标形状
            target_dtype: 目标数据类型
            target_device: 目标设备
            
        Returns:
            适配后的张量
        """
        # 1. 数据类型转换
        if target_dtype is not None and tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)
            
        # 2. 设备迁移
        if target_device is not None and tensor.device != target_device:
            tensor = tensor.to(target_device)
            
        # 3. 维度适配
        if tensor.dim() == 3 and tensor.size(1) == 1:
            # 压缩3维到2维
            tensor = tensor.squeeze(1)
        elif tensor.dim() != 2:
            raise ValueError(f"输入维度错误: 期望2维或3维(中间维度为1), 实际是{tensor.dim()}维")
            
        # 4. 确保batch_size一致
        if tensor.size(0) != target_shape[0]:
            if tensor.size(0) == 1:
                # 扩展batch维度
                tensor = tensor.expand(target_shape[0], -1)
            else:
                raise ValueError(f"batch_size不匹配: 期望{target_shape[0]}, 实际是{tensor.size(0)}")
                
        # 5. 特征维度适配
        if tensor.size(1) != target_shape[1]:
            # 使用线性投影调整特征维度
            projection_key = f"{tensor.size(1)}->{target_shape[1]}"
            if projection_key not in self.projection_cache:
                self.projection_cache[projection_key] = nn.Linear(
                    tensor.size(1),
                    target_shape[1],
                    bias=False
                ).to(tensor.device)
            projection = self.projection_cache[projection_key]
            tensor = projection(tensor)
            
        return tensor
    
    def transform_features(self,
                         tensor: torch.Tensor,
                         transform_type: str,
                         **kwargs) -> torch.Tensor:
        """特征变换
        
        Args:
            tensor: 输入张量
            transform_type: 变换类型
            **kwargs: 额外参数
            
        Returns:
            变换后的张量
        """
        # 获取或创建变换网络
        if transform_type not in self.feature_transforms:
            self.feature_transforms[transform_type] = self._create_transform_net(
                transform_type,
                tensor.size(-1),
                **kwargs
            )
            
        transform_net = self.feature_transforms[transform_type]
        
        # 应用变换
        return transform_net(tensor)
    
    def merge_tensors(self,
                     tensors: List[torch.Tensor],
                     merge_type: str = 'concat',
                     dim: int = -1) -> torch.Tensor:
        """合并多个张量
        
        Args:
            tensors: 张量列表
            merge_type: 合并类型 ('concat', 'sum', 'mean', 'max')
            dim: 合并维度
            
        Returns:
            合并后的张量
        """
        if not tensors:
            raise ValueError("Empty tensor list")
            
        # 确保所有张量在同一设备上
        target_device = tensors[0].device
        tensors = [t.to(target_device) for t in tensors]
        
        # 根据合并类型处理
        if merge_type == 'concat':
            return torch.cat(tensors, dim=dim)
        elif merge_type == 'sum':
            return torch.stack(tensors, dim=dim).sum(dim=dim)
        elif merge_type == 'mean':
            return torch.stack(tensors, dim=dim).mean(dim=dim)
        elif merge_type == 'max':
            return torch.stack(tensors, dim=dim).max(dim=dim)[0]
        else:
            raise ValueError(f"Unsupported merge type: {merge_type}")
            
    def split_tensor(self,
                    tensor: torch.Tensor,
                    split_size_or_sections: Union[int, List[int]],
                    dim: int = -1) -> List[torch.Tensor]:
        """分割张量
        
        Args:
            tensor: 输入张量
            split_size_or_sections: 分割大小或分段列表
            dim: 分割维度
            
        Returns:
            分割后的张量列表
        """
        return list(torch.split(tensor, split_size_or_sections, dim))
    
    def optimize_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        """优化内存使用
        
        Args:
            tensor: 输入张量
            
        Returns:
            优化后的张量
        """
        # 1. 移除不必要的梯度历史
        if tensor.requires_grad:
            tensor = tensor.detach()
            
        # 2. 使用最适合的数据类型
        if tensor.dtype in [torch.float64, torch.float32]:
            tensor = tensor.to(torch.float16)
            
        # 3. 移到CPU以节省GPU内存
        if tensor.device.type == 'cuda' and not tensor.requires_grad:
            tensor = tensor.cpu()
            
        return tensor
    
    def _adapt_dimensions(self,
                         tensor: torch.Tensor,
                         target_shape: Tuple[int, ...]) -> torch.Tensor:
        """适配张量维度
        
        Args:
            tensor: 输入张量
            target_shape: 目标形状
            
        Returns:
            维度适配后的张量
        """
        current_shape = tensor.shape
        
        # 如果形状完全相同,直接返回
        if current_shape == target_shape:
            return tensor
            
        # 1. 处理维度数不同的情况
        while len(current_shape) < len(target_shape):
            tensor = tensor.unsqueeze(0)
            current_shape = tensor.shape
            
        while len(current_shape) > len(target_shape):
            tensor = tensor.squeeze(0)
            current_shape = tensor.shape
            
        # 2. 处理维度大小不同的情况
        for i, (current_dim, target_dim) in enumerate(zip(current_shape, target_shape)):
            if current_dim != target_dim:
                # 对于batch维度,使用复制或裁剪
                if i == 0:
                    if current_dim < target_dim:
                        # 复制batch
                        repeats = [1] * len(current_shape)
                        repeats[0] = target_dim // current_dim + 1
                        tensor = tensor.repeat(*repeats)[:target_dim]
                    else:
                        # 裁剪batch
                        tensor = tensor[:target_dim]
                        
                # 对于特征维度,使用投影
                elif i == len(current_shape) - 1:
                    projection_key = f"{current_dim}->{target_dim}"
                    if projection_key not in self.projection_cache:
                        self.projection_cache[projection_key] = nn.Linear(
                            current_dim,
                            target_dim,
                            bias=False
                        )
                    projection = self.projection_cache[projection_key]
                    tensor = projection(tensor)
                    
                # 对于其他维度,使用插值
                else:
                    tensor = nn.functional.interpolate(
                        tensor,
                        size=target_shape[i],
                        mode='linear' if i == 1 else 'nearest'
                    )
                    
        return tensor
    
    def _create_transform_net(self,
                            transform_type: str,
                            feature_dim: int,
                            **kwargs) -> nn.Module:
        """创建特征变换网络
        
        Args:
            transform_type: 变换类型
            feature_dim: 特征维度
            **kwargs: 额外参数
            
        Returns:
            变换网络
        """
        if transform_type == 'mlp':
            return nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.ReLU(),
                nn.Linear(feature_dim * 2, feature_dim)
            )
        elif transform_type == 'attention':
            num_heads = kwargs.get('num_heads', 4)
            return nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                batch_first=True
            )
        elif transform_type == 'conv1d':
            return nn.Sequential(
                nn.Conv1d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 1, 3, padding=1)
            )
        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")
            
    def to_device(self, tensor: torch.Tensor, device: str) -> torch.Tensor:
        """将张量迁移到指定设备
        
        Args:
            tensor: 输入张量
            device: 目标设备
            
        Returns:
            迁移后的张量
        """
        # 获取或创建设备映射
        if device not in self.device_map:
            self.device_map[device] = torch.device(device)
            
        target_device = self.device_map[device]
        
        # 如果已经在目标设备上,直接返回
        if tensor.device == target_device:
            return tensor
            
        return tensor.to(target_device)
    
    def clear_cache(self) -> None:
        """清理缓存"""
        self.projection_cache.clear()
        self.feature_transforms.clear()
        self.device_map.clear()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 