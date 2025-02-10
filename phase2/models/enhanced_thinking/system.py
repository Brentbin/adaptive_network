"""思维系统模块

实现多层次思维系统的集成。
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import time
from dataclasses import dataclass

from .thinking_levels import (
    FastIntuitionLevel,
    AnalyticalLevel,
    AbstractReasoningLevel,
    MetaCognitiveLevel,
    ThinkingLevel
)
from .config_manager import SystemConfig
from .communication import CommunicationManager, Message
from .performance import PerformanceMonitor, PerformanceMetrics
from .tensor_utils import TensorAdapter

@dataclass
class SystemState:
    """系统状态"""
    active_levels: set          # 当前活跃的层次
    processing_depth: float     # 当前处理深度
    resource_distribution: Dict[str, float]  # 资源分布
    performance_metrics: Dict[str, float]    # 性能指标
    timestamp: float           # 时间戳

class ThinkingSystem:
    """多层次思维系统
    
    整合各个组件，实现完整的思维处理系统。
    """
    
    def __init__(self, config: SystemConfig):
        # 验证配置
        if not config.validate():
            raise ValueError("Invalid system configuration")
            
        self.config = config
        
        # 初始化组件
        self.communication_manager = CommunicationManager()
        self.performance_monitor = PerformanceMonitor()
        self.tensor_adapter = TensorAdapter()
        
        # 初始化思维层次
        self.levels = {
            'fast_intuition': FastIntuitionLevel(0, config),
            'analytical': AnalyticalLevel(1, config),
            'abstract_reasoning': AbstractReasoningLevel(2, config),
            'metacognitive': MetaCognitiveLevel(3, config)
        }
        
        # 系统状态
        self.state = SystemState(
            active_levels=set(),
            processing_depth=0.0,
            resource_distribution={},
            performance_metrics={},
            timestamp=time.time()
        )
        
        # 性能历史
        self.performance_history = []
        
        # 处理历史
        self.processing_history = []
        
    def process(self, 
                input_data: torch.Tensor,
                context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """处理输入数据
        
        Args:
            input_data: 输入数据
            context: 上下文信息
            
        Returns:
            处理结果和系统状态
        """
        context = context or {}
        start_time = time.time()
        
        try:
            # 1. 准备阶段
            self._prepare_processing(input_data, context)
            
            # 2. 自下而上的处理
            bottom_up_results = self._bottom_up_processing(input_data, context)
            
            # 3. 自上而下的调控
            top_down_control = self._top_down_control(bottom_up_results, context)
            
            # 4. 整合结果
            final_result = self._integrate_results(bottom_up_results, top_down_control)
            
            # 5. 更新系统状态
            self._update_system_state(final_result, bottom_up_results)
            
            # 6. 记录性能指标
            processing_time = time.time() - start_time
            self._record_performance(final_result, processing_time)
            
            # 7. 清理缓存
            self.tensor_adapter.clear_cache()
            
            return final_result, dict(self.state.__dict__)
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            self._handle_processing_error(e)
            raise
            
    def update(self, feedback: Dict[str, float]) -> None:
        """更新系统
        
        Args:
            feedback: 外部反馈
        """
        try:
            # 1. 收集反馈
            performance_metrics = PerformanceMetrics(
                accuracy=feedback.get('accuracy', 0.5),
                efficiency=feedback.get('efficiency', 0.5),
                resource_usage=feedback.get('resource_usage', 0.5),
                response_time=feedback.get('response_time', 0.5),
                stability=feedback.get('stability', 0.5),
                adaptability=feedback.get('adaptability', 0.5),
                timestamp=time.time()
            )
            
            # 2. 记录性能
            self.performance_monitor.record_performance(performance_metrics)
            
            # 3. 分析性能
            analysis = self.performance_monitor.analyze_performance()
            
            # 4. 更新各层
            self._update_levels(analysis)
            
            # 5. 调整系统参数
            self._adjust_system_parameters(analysis)
            
            # 6. 处理警报
            self._handle_alerts()
            
        except Exception as e:
            print(f"Error during update: {str(e)}")
            self._handle_update_error(e)
            raise
            
    def get_state(self) -> Dict[str, Any]:
        """获取系统状态
        
        Returns:
            系统状态信息
        """
        return {
            'state': dict(self.state.__dict__),
            'performance': self.performance_monitor.get_performance_stats(),
            'communication': self.communication_manager.get_communication_stats(),
            'alerts': self.performance_monitor.get_alerts()
        }
        
    def _prepare_processing(self,
                          input_data: torch.Tensor,
                          context: Dict[str, Any]) -> None:
        """准备处理阶段
        
        Args:
            input_data: 输入数据
            context: 上下文信息
        """
        # 1. 评估任务复杂度
        task_complexity = self._evaluate_task_complexity(input_data)
        
        # 2. 激活相关层次
        self._activate_levels(task_complexity)
        
        # 3. 分配初始资源
        self._allocate_initial_resources(task_complexity)
        
        # 4. 广播准备消息
        self._broadcast_preparation_message(context)
        
    def _bottom_up_processing(self,
                            input_data: torch.Tensor,
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """自下而上的处理流程
        
        维度处理:
        1. 输入: [batch_size, input_size] 或 [batch_size, 1, input_size]
        2. 每层输入: [batch_size, input_size]
        3. 每层输出: [batch_size, input_size]
        
        Args:
            input_data: 输入数据
            context: 上下文信息
            
        Returns:
            各层处理结果
        """
        results = {}
        current_input = input_data
        
        # 确保初始输入是2维的
        if current_input.dim() == 3:
            current_input = current_input.squeeze(1)  # [batch_size, 1, input_size] -> [batch_size, input_size]
        elif current_input.dim() != 2:
            raise ValueError(f"输入维度错误: 期望2维或3维(中间维度为1), 实际是{current_input.dim()}维")
        
        # 按层次顺序处理
        for level_name in ['fast_intuition', 'analytical', 'abstract_reasoning', 'metacognitive']:
            if level_name not in self.state.active_levels:
                continue
                
            level = self.levels[level_name]
            
            # 1. 准备输入
            target_shape = (current_input.size(0), self.config.input_size)
            adapted_input = self.tensor_adapter.adapt_tensor(
                current_input,
                target_shape=target_shape,
                target_device=self._get_level_device(level)
            )
            
            # 2. 处理数据
            output, confidence = level.process(adapted_input, context)
            
            # 3. 确保输出是2维的
            if output.dim() == 3:
                output = output.squeeze(1)
            elif output.dim() != 2:
                raise ValueError(f"层级 {level_name} 输出维度错误: 期望2维或3维(中间维度为1), 实际是{output.dim()}维")
            
            # 4. 记录结果
            results[level_name] = {
                'output': output,
                'confidence': confidence,
                'state': level.state
            }
            
            # 5. 发送处理完成消息
            self._send_processing_message(level_name, results[level_name])
            
            # 6. 更新输入
            current_input = output
            
        return results
        
    def _top_down_control(self,
                         bottom_up_results: Dict[str, Any],
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """自上而下的控制
        
        Args:
            bottom_up_results: 自下而上的处理结果
            context: 上下文信息
            
        Returns:
            控制信号
        """
        control_signals = {}
        
        # 从高层到低层进行控制
        for level_name in reversed(['fast_intuition', 'analytical', 'abstract_reasoning', 'metacognitive']):
            if level_name not in self.state.active_levels:
                continue
                
            # 1. 生成控制信号
            control = self._generate_control_signal(
                level_name,
                bottom_up_results.get(level_name, {}),
                context
            )
            
            # 2. 发送控制消息
            self._send_control_message(level_name, control)
            
            # 3. 记录控制信号
            control_signals[level_name] = control
            
        return control_signals
        
    def _integrate_results(self,
                         bottom_up_results: Dict[str, Any],
                         top_down_control: Dict[str, Any]) -> torch.Tensor:
        """整合处理结果
        
        维度处理:
        1. 输入: 各层输出 [batch_size, input_size]
        2. 输出: [batch_size, input_size]
        
        Args:
            bottom_up_results: 自下而上的处理结果
            top_down_control: 自上而下的控制信号
            
        Returns:
            最终结果
        """
        # 1. 收集各层输出
        outputs = []
        weights = []
        target_shape = None
        
        for level_name, result in bottom_up_results.items():
            if level_name not in self.state.active_levels:
                continue
                
            # 获取输出和权重
            output = result['output']
            confidence = result['confidence']
            control = top_down_control.get(level_name, {})
            weight = confidence * control.get('weight', 1.0)
            
            # 确定目标维度
            if target_shape is None:
                target_shape = output.shape
            
            # 调整输出维度
            if output.shape != target_shape:
                output = self.tensor_adapter.adapt_tensor(
                    output,
                    target_shape=target_shape
                )
            
            outputs.append(output)
            weights.append(weight)
            
        # 2. 如果没有有效输出,返回空结果
        if not outputs:
            return torch.zeros(target_shape)
            
        # 3. 加权整合
        weights = torch.tensor(weights, device=outputs[0].device)
        weights = weights / (weights.sum() + 1e-10)
        
        integrated_output = sum(o * w for o, w in zip(outputs, weights))
        
        return integrated_output
        
    def _update_system_state(self,
                           final_result: torch.Tensor,
                           level_results: Dict[str, Any]) -> None:
        """更新系统状态
        
        Args:
            final_result: 最终结果
            level_results: 各层结果
        """
        # 1. 更新活跃层次
        self.state.active_levels = set(level_results.keys())
        
        # 2. 更新处理深度
        self.state.processing_depth = len(self.state.active_levels) / len(self.levels)
        
        # 3. 更新资源分布
        self.state.resource_distribution = {
            level_name: result['state'].resource_usage
            for level_name, result in level_results.items()
        }
        
        # 4. 更新性能指标
        self.state.performance_metrics = {
            level_name: {
                'confidence': result['confidence'],
                'output_norm': torch.norm(result['output']).item()
            }
            for level_name, result in level_results.items()
        }
        
        # 5. 更新时间戳
        self.state.timestamp = time.time()
        
    def _evaluate_task_complexity(self, input_data: torch.Tensor) -> float:
        """评估任务复杂度
        
        Args:
            input_data: 输入数据
            
        Returns:
            复杂度得分 (0.0 ~ 1.0)
        """
        # 1. 计算特征复杂度
        feature_complexity = torch.std(input_data).item()
        
        # 2. 考虑维度复杂度
        dim_complexity = min(1.0, input_data.dim() / 4)
        
        # 3. 考虑数据规模
        size_complexity = min(1.0, input_data.numel() / 1000)
        
        # 4. 综合评估
        complexity = (
            feature_complexity * 0.4 +
            dim_complexity * 0.3 +
            size_complexity * 0.3
        )
        
        return min(1.0, complexity)
        
    def _activate_levels(self, task_complexity: float) -> None:
        """激活相关层次
        
        Args:
            task_complexity: 任务复杂度
        """
        # 清除当前激活状态
        self.state.active_levels.clear()
        
        # 基于任务复杂度激活层次
        if task_complexity > self.config.resource_threshold:
            self.state.active_levels.add('fast_intuition')
            
        if task_complexity > self.config.resource_threshold * 2:
            self.state.active_levels.add('analytical')
            
        if task_complexity > self.config.resource_threshold * 3:
            self.state.active_levels.add('abstract_reasoning')
            
        # 元认知层始终保持激活
        self.state.active_levels.add('metacognitive')
        
    def _allocate_initial_resources(self, task_complexity: float) -> None:
        """分配初始资源
        
        Args:
            task_complexity: 任务复杂度
        """
        # 基础资源分配比例
        base_allocation = {
            'fast_intuition': 0.2,
            'analytical': 0.3,
            'abstract_reasoning': 0.4,
            'metacognitive': 0.1
        }
        
        # 根据任务复杂度调整分配
        adjusted_allocation = {}
        for level_name, ratio in base_allocation.items():
            if level_name in self.state.active_levels:
                # 复杂任务增加高层资源，简单任务增加低层资源
                if task_complexity > 0.7:
                    adjustment = 0.1 if level_name in ['abstract_reasoning', 'metacognitive'] else -0.05
                else:
                    adjustment = 0.1 if level_name in ['fast_intuition', 'analytical'] else -0.05
                    
                adjusted_ratio = max(
                    self.config.min_resource_ratio,
                    min(1.0, ratio + adjustment)
                )
                adjusted_allocation[level_name] = adjusted_ratio
                
        # 归一化分配比例
        total = sum(adjusted_allocation.values())
        self.state.resource_distribution = {
            level_name: ratio / total
            for level_name, ratio in adjusted_allocation.items()
        }
        
    def _broadcast_preparation_message(self, context: Dict[str, Any]) -> None:
        """广播准备消息
        
        Args:
            context: 上下文信息
        """
        message = Message(
            source='system',
            target='all',
            message_type='preparation',
            content={
                'active_levels': list(self.state.active_levels),
                'resource_distribution': dict(self.state.resource_distribution),
                'context': context
            },
            timestamp=time.time(),
            priority=0.8
        )
        
        self.communication_manager.broadcast_message(message)
        
    def _send_processing_message(self,
                               level_name: str,
                               result: Dict[str, Any]) -> None:
        """发送处理完成消息
        
        Args:
            level_name: 层级名称
            result: 处理结果
        """
        message = Message(
            source=level_name,
            target='metacognitive',
            message_type='processing_complete',
            content={
                'output_shape': result['output'].shape,
                'confidence': result['confidence'],
                'state': result['state']
            },
            timestamp=time.time(),
            priority=0.5
        )
        
        self.communication_manager.send_message(message)
        
    def _send_control_message(self,
                            level_name: str,
                            control: Dict[str, Any]) -> None:
        """发送控制消息
        
        Args:
            level_name: 层级名称
            control: 控制信号
        """
        message = Message(
            source='metacognitive',
            target=level_name,
            message_type='control',
            content=control,
            timestamp=time.time(),
            priority=0.7
        )
        
        self.communication_manager.send_message(message)
        
    def _generate_control_signal(self,
                               level_name: str,
                               level_result: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """生成控制信号
        
        Args:
            level_name: 层级名称
            level_result: 层级结果
            context: 上下文信息
            
        Returns:
            控制信号
        """
        control = {
            'weight': 1.0,
            'inhibition': 0.0,
            'modulation': {}
        }
        
        # 基于置信度调整权重
        if 'confidence' in level_result:
            control['weight'] *= level_result['confidence']
            
        # 基于性能调整抑制
        if level_name in self.state.performance_metrics:
            metrics = self.state.performance_metrics[level_name]
            if metrics['confidence'] < self.config.confidence_threshold:
                control['inhibition'] = 0.2
                
        # 添加调制信号
        control['modulation'] = {
            'learning_rate': context.get('learning_rate', self.config.base_learning_rate),
            'attention_focus': context.get('attention_focus', None)
        }
        
        return control
        
    def _get_level_input_shape(self, level: ThinkingLevel) -> Tuple[int, ...]:
        """获取层级的输入形状
        
        Args:
            level: 思维层级
            
        Returns:
            输入形状
        """
        return (1, self.config.input_size)
        
    def _get_level_device(self, level: ThinkingLevel) -> torch.device:
        """获取层级的设备
        
        Args:
            level: 思维层级
            
        Returns:
            设备
        """
        return next(level.parameters()).device
        
    def _record_performance(self,
                          final_result: torch.Tensor,
                          processing_time: float) -> None:
        """记录性能指标
        
        Args:
            final_result: 最终结果
            processing_time: 处理时间
        """
        metrics = PerformanceMetrics(
            accuracy=torch.mean(torch.abs(final_result)).item(),
            efficiency=1.0 / (processing_time + 1e-10),
            resource_usage=max(self.state.resource_distribution.values()),
            response_time=processing_time,
            stability=1.0 - torch.std(final_result).item(),
            adaptability=len(self.state.active_levels) / len(self.levels),
            timestamp=time.time()
        )
        
        self.performance_monitor.record_performance(metrics)
        
    def _update_levels(self, analysis: Dict[str, Any]) -> None:
        """更新各层级
        
        Args:
            analysis: 性能分析结果
        """
        for level_name, level in self.levels.items():
            if level_name not in self.state.active_levels:
                continue
                
            # 获取层级性能
            level_analysis = analysis['level_analysis'].get(level_name, {})
            performance = level_analysis.get('overall_performance', 0.5)
            
            # 更新层级
            level.update(performance)
            
    def _adjust_system_parameters(self, analysis: Dict[str, Any]) -> None:
        """调整系统参数
        
        Args:
            analysis: 性能分析结果
        """
        # 处理建议
        for suggestion in analysis['suggestions']:
            if suggestion['type'] == 'bottleneck_mitigation':
                self._handle_bottleneck(suggestion)
            elif suggestion['type'] == 'level_optimization':
                self._handle_level_optimization(suggestion)
            elif suggestion['type'] == 'trend_optimization':
                self._handle_trend_optimization(suggestion)
                
    def _handle_bottleneck(self, suggestion: Dict[str, Any]) -> None:
        """处理瓶颈
        
        Args:
            suggestion: 优化建议
        """
        target = suggestion['target']
        severity = suggestion['priority']
        
        # 调整资源分配
        if target == 'resource_usage':
            self._reallocate_resources(severity)
        # 调整学习率
        elif target == 'accuracy':
            self.config.base_learning_rate *= (1 + severity * 0.1)
        # 调整稳定性
        elif target == 'stability':
            self.config.stability_threshold *= (1 - severity * 0.1)
            
    def _handle_level_optimization(self, suggestion: Dict[str, Any]) -> None:
        """处理层级优化
        
        Args:
            suggestion: 优化建议
        """
        target = suggestion['target']
        priority = suggestion['priority']
        
        # 增加资源分配
        current_allocation = self.state.resource_distribution.get(target, 0.0)
        self.state.resource_distribution[target] = min(
            1.0,
            current_allocation * (1 + priority * 0.2)
        )
        
        # 归一化资源分配
        total = sum(self.state.resource_distribution.values())
        self.state.resource_distribution = {
            level_name: ratio / total
            for level_name, ratio in self.state.resource_distribution.items()
        }
        
    def _handle_trend_optimization(self, suggestion: Dict[str, Any]) -> None:
        """处理趋势优化
        
        Args:
            suggestion: 优化建议
        """
        target = suggestion['target']
        priority = suggestion['priority']
        
        # 调整相关参数
        if target == 'efficiency':
            self.config.integration_rate *= (1 - priority * 0.1)
        elif target == 'adaptability':
            self.config.coordination_threshold *= (1 - priority * 0.1)
            
    def _reallocate_resources(self, severity: float) -> None:
        """重新分配资源
        
        Args:
            severity: 问题严重程度
        """
        # 计算新的分配比例
        new_allocation = {}
        for level_name, ratio in self.state.resource_distribution.items():
            if level_name in ['abstract_reasoning', 'metacognitive']:
                # 减少高层资源
                new_allocation[level_name] = ratio * (1 - severity * 0.2)
            else:
                # 增加低层资源
                new_allocation[level_name] = ratio * (1 + severity * 0.2)
                
        # 归一化
        total = sum(new_allocation.values())
        self.state.resource_distribution = {
            level_name: ratio / total
            for level_name, ratio in new_allocation.items()
        }
        
    def _handle_alerts(self) -> None:
        """处理警报"""
        alerts = self.performance_monitor.get_alerts()
        
        for alert in alerts:
            severity = alert['severity']
            metric = alert['metric']
            
            if severity > 0.8:  # 严重警报
                if metric == 'stability':
                    # 增加稳定性控制
                    self.config.stability_threshold *= 1.2
                elif metric == 'resource_usage':
                    # 触发资源重分配
                    self._reallocate_resources(severity)
                elif metric == 'response_time':
                    # 减少活跃层次
                    self._reduce_active_levels()
                    
    def _reduce_active_levels(self) -> None:
        """减少活跃层次"""
        if 'abstract_reasoning' in self.state.active_levels:
            self.state.active_levels.remove('abstract_reasoning')
            
        if len(self.state.active_levels) > 2 and 'analytical' in self.state.active_levels:
            self.state.active_levels.remove('analytical')
            
    def _handle_processing_error(self, error: Exception) -> None:
        """处理处理过程中的错误
        
        Args:
            error: 错误对象
        """
        # 记录错误
        print(f"Processing error: {str(error)}")
        
        # 重置状态
        self.state.active_levels = {'fast_intuition', 'metacognitive'}
        self.state.processing_depth = 0.0
        
        # 清理缓存
        self.tensor_adapter.clear_cache()
        self.communication_manager.clear_buffers()
        
    def _handle_update_error(self, error: Exception) -> None:
        """处理更新过程中的错误
        
        Args:
            error: 错误对象
        """
        # 记录错误
        print(f"Update error: {str(error)}")
        
        # 恢复默认配置
        self.config = SystemConfig(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size
        ) 