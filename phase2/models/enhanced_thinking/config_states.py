"""配置状态系统的实现。

这个模块实现了思维系统的配置状态管理，包括：
1. 学习优化状态 (D⁻¹Pr⁰)：偏重于学习和适应
2. 平衡状态 (D⁰Pr⁰)：平衡各种资源利用
3. 记忆优化状态 (D¹Pr⁻β)：偏重于记忆和稳定性
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum, auto

from .base import ConfigurationState

class StateType(Enum):
    """配置状态类型"""
    LEARNING_OPTIMIZED = auto()  # D⁻¹Pr⁰
    BALANCED = auto()            # D⁰Pr⁰
    MEMORY_OPTIMIZED = auto()    # D¹Pr⁻β

@dataclass
class StateConfig:
    """状态配置参数"""
    learning_rate: float
    memory_weight: float
    attention_dropout: float
    hidden_dropout: float
    temperature: float = 1.0

class ConfigStateManager:
    """配置状态管理器"""
    
    def __init__(self):
        # 默认配置参数
        self.default_configs = {
            StateType.LEARNING_OPTIMIZED: StateConfig(
                learning_rate=1e-3,
                memory_weight=0.3,
                attention_dropout=0.1,
                hidden_dropout=0.1,
                temperature=1.2
            ),
            StateType.BALANCED: StateConfig(
                learning_rate=5e-4,
                memory_weight=0.5,
                attention_dropout=0.15,
                hidden_dropout=0.15,
                temperature=1.0
            ),
            StateType.MEMORY_OPTIMIZED: StateConfig(
                learning_rate=1e-4,
                memory_weight=0.8,
                attention_dropout=0.2,
                hidden_dropout=0.2,
                temperature=0.8
            )
        }
        
        self.current_state = StateType.BALANCED
        self.state_history = []
        self.performance_history = []
        
    def get_current_config(self) -> StateConfig:
        """获取当前状态的配置"""
        return self.default_configs[self.current_state]
    
    def record_performance(self, performance: float) -> None:
        """记录性能指标"""
        self.performance_history.append(performance)
        self.state_history.append(self.current_state)
        
    def should_transition(self) -> bool:
        """判断是否需要状态转换"""
        if len(self.performance_history) < 5:
            return False
            
        # 计算最近的性能变化趋势
        recent_perf = self.performance_history[-5:]
        trend = sum(b - a for a, b in zip(recent_perf[:-1], recent_perf[1:]))
        
        return abs(trend) > 0.1  # 如果性能变化显著，考虑转换状态
        
    def get_next_state(self) -> StateType:
        """决定下一个状态"""
        if not self.performance_history:
            return self.current_state
            
        recent_perf = self.performance_history[-5:]
        trend = sum(b - a for a, b in zip(recent_perf[:-1], recent_perf[1:]))
        
        if trend < -0.1:  # 性能下降
            if self.current_state == StateType.MEMORY_OPTIMIZED:
                return StateType.BALANCED
            else:
                return StateType.LEARNING_OPTIMIZED
        elif trend > 0.1:  # 性能提升
            if self.current_state == StateType.LEARNING_OPTIMIZED:
                return StateType.BALANCED
            else:
                return StateType.MEMORY_OPTIMIZED
        else:
            return self.current_state

class LearningOptimizedState(ConfigurationState):
    """学习优化状态 (D⁻¹Pr⁰)
    
    特点：
    1. 更高的学习率
    2. 更低的记忆权重
    3. 更低的dropout以促进学习
    """
    
    def __init__(self, config: StateConfig):
        self.config = config
        
    def apply(self, network: Any, context: Dict[str, Any]) -> None:
        """应用学习优化配置"""
        # 设置学习率
        if hasattr(network, 'optimizer'):
            for param_group in network.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate
                
        # 设置dropout
        for module in network.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.config.hidden_dropout
            elif isinstance(module, nn.MultiheadAttention):
                module.dropout = self.config.attention_dropout
                
        # 更新上下文
        context['memory_weight'] = self.config.memory_weight
        context['temperature'] = self.config.temperature
        
    def evaluate(self, performance: float, context: Dict[str, Any]) -> float:
        """评估状态效果"""
        # 在学习优化状态下，我们更关注性能提升的速度
        if len(context.get('performance_history', [])) > 1:
            prev_perf = context['performance_history'][-2]
            improvement_rate = (performance - prev_perf) / prev_perf
            return improvement_rate
        return 0.0

class BalancedState(ConfigurationState):
    """平衡状态 (D⁰Pr⁰)
    
    特点：
    1. 中等学习率
    2. 平衡的记忆权重
    3. 适中的dropout
    """
    
    def __init__(self, config: StateConfig):
        self.config = config
        
    def apply(self, network: Any, context: Dict[str, Any]) -> None:
        """应用平衡配置"""
        # 设置学习率
        if hasattr(network, 'optimizer'):
            for param_group in network.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate
                
        # 设置dropout
        for module in network.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.config.hidden_dropout
            elif isinstance(module, nn.MultiheadAttention):
                module.dropout = self.config.attention_dropout
                
        # 更新上下文
        context['memory_weight'] = self.config.memory_weight
        context['temperature'] = self.config.temperature
        
    def evaluate(self, performance: float, context: Dict[str, Any]) -> float:
        """评估状态效果"""
        # 在平衡状态下，我们同时关注性能和稳定性
        stability = 1.0
        if len(context.get('performance_history', [])) > 2:
            recent_perf = context['performance_history'][-3:]
            stability = 1.0 - torch.std(torch.tensor(recent_perf)).item()
        return performance * stability

class MemoryOptimizedState(ConfigurationState):
    """记忆优化状态 (D¹Pr⁻β)
    
    特点：
    1. 较低的学习率
    2. 较高的记忆权重
    3. 较高的dropout以促进泛化
    """
    
    def __init__(self, config: StateConfig):
        self.config = config
        
    def apply(self, network: Any, context: Dict[str, Any]) -> None:
        """应用记忆优化配置"""
        # 设置学习率
        if hasattr(network, 'optimizer'):
            for param_group in network.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate
                
        # 设置dropout
        for module in network.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.config.hidden_dropout
            elif isinstance(module, nn.MultiheadAttention):
                module.dropout = self.config.attention_dropout
                
        # 更新上下文
        context['memory_weight'] = self.config.memory_weight
        context['temperature'] = self.config.temperature
        
    def evaluate(self, performance: float, context: Dict[str, Any]) -> float:
        """评估状态效果"""
        # 在记忆优化状态下，我们更关注性能的稳定性
        if len(context.get('performance_history', [])) > 3:
            recent_perf = context['performance_history'][-4:]
            stability = 1.0 - torch.std(torch.tensor(recent_perf)).item()
            return performance * stability
        return performance 