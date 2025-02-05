"""增强思维模块

实现了一个多层次的认知系统，包括：
1. 快速直觉处理
2. 分析处理
3. 抽象推理
4. 元认知控制

系统特点：
1. 动态资源分配
2. 连续状态空间控制
3. 完整的反馈机制
4. 层次间协同工作
"""

from .thinking_levels import (
    ThinkingLevel,
    FastIntuitionLevel,
    AnalyticalLevel,
    AbstractReasoningLevel,
    MetaCognitiveLevel,
    ThinkingSystem
)

from .config_manager import (
    SystemConfig,
    ConfigManager
)

from .system_logger import SystemLogger
from .system_tester import SystemTester
from .system_monitor import SystemMonitor

__all__ = [
    'ThinkingLevel',
    'FastIntuitionLevel',
    'AnalyticalLevel',
    'AbstractReasoningLevel',
    'MetaCognitiveLevel',
    'ThinkingSystem',
    'SystemConfig',
    'ConfigManager',
    'SystemLogger',
    'SystemTester',
    'SystemMonitor'
] 