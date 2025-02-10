"""配置管理模块

负责系统配置的加载、保存和更新。
"""

import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import os

@dataclass
class SystemConfig:
    """系统配置类
    
    管理整个思维系统的配置参数
    """
    # 基础网络参数
    input_size: int
    hidden_size: int
    num_layers: int = 2
    dropout: float = 0.1
    attention_heads: int = 4
    
    # 资源管理参数
    resource_threshold: float = 0.2  # 资源分配阈值
    resource_buffer: float = 0.1     # 资源缓冲比例
    min_resource_ratio: float = 0.1  # 最小资源分配比例
    
    # 协调参数
    coordination_threshold: float = 0.3  # 层级协调阈值
    feedback_strength: float = 0.5       # 反馈强度
    integration_rate: float = 0.1        # 结果整合速率
    
    # 性能参数
    performance_window: int = 50      # 性能历史窗口大小
    confidence_threshold: float = 0.3  # 置信度阈值
    stability_threshold: float = 0.2   # 稳定性阈值
    
    # 学习参数
    base_learning_rate: float = 1e-3  # 基础学习率
    min_learning_rate: float = 1e-5   # 最小学习率
    learning_rate_decay: float = 0.95  # 学习率衰减
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'network': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'attention_heads': self.attention_heads
            },
            'resource': {
                'threshold': self.resource_threshold,
                'buffer': self.resource_buffer,
                'min_ratio': self.min_resource_ratio
            },
            'coordination': {
                'threshold': self.coordination_threshold,
                'feedback_strength': self.feedback_strength,
                'integration_rate': self.integration_rate
            },
            'performance': {
                'window': self.performance_window,
                'confidence_threshold': self.confidence_threshold,
                'stability_threshold': self.stability_threshold
            },
            'learning': {
                'base_rate': self.base_learning_rate,
                'min_rate': self.min_learning_rate,
                'decay': self.learning_rate_decay
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """从字典创建配置对象"""
        network = config_dict.get('network', {})
        resource = config_dict.get('resource', {})
        coordination = config_dict.get('coordination', {})
        performance = config_dict.get('performance', {})
        learning = config_dict.get('learning', {})
        
        return cls(
            # 网络参数
            input_size=network.get('input_size'),
            hidden_size=network.get('hidden_size'),
            num_layers=network.get('num_layers', 2),
            dropout=network.get('dropout', 0.1),
            attention_heads=network.get('attention_heads', 4),
            
            # 资源参数
            resource_threshold=resource.get('threshold', 0.2),
            resource_buffer=resource.get('buffer', 0.1),
            min_resource_ratio=resource.get('min_ratio', 0.1),
            
            # 协调参数
            coordination_threshold=coordination.get('threshold', 0.3),
            feedback_strength=coordination.get('feedback_strength', 0.5),
            integration_rate=coordination.get('integration_rate', 0.1),
            
            # 性能参数
            performance_window=performance.get('window', 50),
            confidence_threshold=performance.get('confidence_threshold', 0.3),
            stability_threshold=performance.get('stability_threshold', 0.2),
            
            # 学习参数
            base_learning_rate=learning.get('base_rate', 1e-3),
            min_learning_rate=learning.get('min_rate', 1e-5),
            learning_rate_decay=learning.get('decay', 0.95)
        )
    
    def update(self, updates: Dict[str, Any]) -> None:
        """更新配置参数
        
        Args:
            updates: 需要更新的参数字典
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def validate(self) -> bool:
        """验证配置参数的有效性
        
        Returns:
            配置是否有效
        """
        try:
            # 验证网络参数
            assert self.input_size > 0, "input_size must be positive"
            assert self.hidden_size > 0, "hidden_size must be positive"
            assert self.num_layers > 0, "num_layers must be positive"
            assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
            assert self.attention_heads > 0, "attention_heads must be positive"
            
            # 验证资源参数
            assert 0 <= self.resource_threshold <= 1, "resource_threshold must be in [0, 1]"
            assert 0 <= self.resource_buffer <= 1, "resource_buffer must be in [0, 1]"
            assert 0 <= self.min_resource_ratio <= 1, "min_resource_ratio must be in [0, 1]"
            
            # 验证协调参数
            assert 0 <= self.coordination_threshold <= 1, "coordination_threshold must be in [0, 1]"
            assert 0 <= self.feedback_strength <= 1, "feedback_strength must be in [0, 1]"
            assert 0 <= self.integration_rate <= 1, "integration_rate must be in [0, 1]"
            
            # 验证性能参数
            assert self.performance_window > 0, "performance_window must be positive"
            assert 0 <= self.confidence_threshold <= 1, "confidence_threshold must be in [0, 1]"
            assert 0 <= self.stability_threshold <= 1, "stability_threshold must be in [0, 1]"
            
            # 验证学习参数
            assert self.base_learning_rate > 0, "base_learning_rate must be positive"
            assert self.min_learning_rate > 0, "min_learning_rate must be positive"
            assert 0 < self.learning_rate_decay <= 1, "learning_rate_decay must be in (0, 1]"
            assert self.min_learning_rate <= self.base_learning_rate, "min_learning_rate must be <= base_learning_rate"
            
            return True
            
        except AssertionError as e:
            print(f"Configuration validation failed: {str(e)}")
            return False

class ConfigManager:
    """配置管理器"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> SystemConfig:
        """加载配置"""
        if not os.path.exists(self.config_path):
            # 创建默认配置
            default_config = SystemConfig(
                input_size=256,
                hidden_size=512,
                num_layers=2,
                dropout=0.1,
                attention_heads=4
            )
            self.save_config(default_config)
            return default_config
            
        # 根据文件扩展名选择加载方式
        ext = os.path.splitext(self.config_path)[1]
        if ext == '.json':
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")
            
        return SystemConfig(**config_dict)
        
    def save_config(self, config: Optional[SystemConfig] = None) -> None:
        """保存配置"""
        if config is None:
            config = self.config
            
        config_dict = asdict(config)
        
        # 根据文件扩展名选择保存方式
        ext = os.path.splitext(self.config_path)[1]
        if ext == '.json':
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
        elif ext in ['.yaml', '.yml']:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_dict, f, allow_unicode=True)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """更新配置
        
        Args:
            updates: 需要更新的配置项
        """
        config_dict = asdict(self.config)
        config_dict.update(updates)
        self.config = SystemConfig(**config_dict)
        self.save_config()
        
    def get_config(self) -> SystemConfig:
        """获取当前配置"""
        return self.config 