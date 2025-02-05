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
    """系统配置"""
    # 基础配置
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    attention_heads: int = 4
    
    # 资源管理配置
    resource_thresholds: Dict[str, float] = None
    
    # 状态控制配置
    state_transition_params: Dict[str, float] = None
    
    # 反馈系统配置
    feedback_thresholds: Dict[str, float] = None
    
    # 测试参数
    test_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化默认值"""
        if self.resource_thresholds is None:
            self.resource_thresholds = {
                'min_allocation': 0.1,
                'max_allocation': 0.8,
                'balance_factor': 0.5
            }
            
        if self.state_transition_params is None:
            self.state_transition_params = {
                'learning_rate': 0.1,
                'momentum': 0.9,
                'stability_threshold': 0.2
            }
            
        if self.feedback_thresholds is None:
            self.feedback_thresholds = {
                'performance_drop': 0.2,
                'efficiency_min': 0.3,
                'plasticity_min': 0.2,
                'confidence_min': 0.4
            }
            
        if self.test_params is None:
            self.test_params = {
                'num_test_cases': 100,
                'performance_threshold': 0.8,
                'max_test_time': 300,
                'error_tolerance': 0.1
            }

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
                learning_rate=0.001,
                batch_size=32,
                max_epochs=100,
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