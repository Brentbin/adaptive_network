"""具体ThinkingLevel类的实现。

这个模块实现了4个层次的思维处理能力:
1. 快速直觉处理 (FastIntuitionLevel)
2. 分析处理 (AnalyticalLevel) 
3. 抽象推理 (AbstractReasoningLevel)
4. 元认知控制 (MetaCognitiveLevel)

每个层次都继承自基础的ThinkingLevel类，并实现了各自特定的处理逻辑。

系统架构重构设计

核心目标：
1. 实现连续状态空间中的精确控制
2. 动态资源分配和平衡
3. 完整的反馈调节机制
4. 层次间的协同工作机制

TODO: 
1. 实现ResourceManager - 用于全局资源的动态分配和平衡
2. 实现StateController - 处理连续状态空间的调节
3. 实现FeedbackSystem - 构建完整的反馈机制
4. 重构现有ThinkingLevel基类
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
import time
import sys

from .base import ThinkingLevel, ThinkingState
from .config_states import ConfigStateManager
from .config_manager import SystemConfig
from .tensor_utils import TensorAdapter

@dataclass
class LevelConfig:
    """思维层次的配置参数"""
    input_size: int
    hidden_size: int
    num_layers: int = 2
    dropout: float = 0.1
    attention_heads: int = 4

class ResourceManager:
    """资源管理器
    
    负责系统级资源的动态分配和平衡。基于神经科学中的D∝1/Pr原理，
    实现资源密度与效率的动态平衡。
    """
    def __init__(self):
        # 全局资源池
        self.global_resources = {
            'computation': 1.0,  # 计算资源
            'memory': 1.0,      # 内存资源
            'attention': 1.0    # 注意力资源
        }
        
        # 资源分配历史
        self.allocation_history = []
        
        # 资源使用统计
        self.usage_stats = {
            'computation': [],
            'memory': [],
            'attention': []
        }
        
        # 资源阈值
        self.thresholds = {
            'min_allocation': 0.1,  # 最小分配比例
            'max_allocation': 0.8,  # 最大分配比例
            'balance_factor': 0.5   # 平衡因子
        }
    
    def allocate(self, 
                 demand: Dict[str, float], 
                 context: Dict[str, Any]) -> Dict[str, float]:
        """动态分配资源
        
        基于需求和上下文动态分配资源，遵循D∝1/Pr原理
        
        Args:
            demand: 资源需求字典
            context: 包含性能指标等上下文信息
            
        Returns:
            分配的资源字典
        """
        # 计算效率指标(Pr)
        efficiency = self._calculate_efficiency(context)
        
        # 根据D∝1/Pr计算理想资源密度
        ideal_density = 1.0 / (efficiency + 1e-6)  # 避免除零
        
        # 调整资源分配
        allocation = {}
        for resource_type, amount in demand.items():
            # 基础分配
            base_allocation = amount * ideal_density
            
            # 应用阈值约束
            allocation[resource_type] = np.clip(
                base_allocation,
                self.thresholds['min_allocation'],
                self.thresholds['max_allocation']
            )
            
            # 记录分配历史
            self.allocation_history.append({
                'type': resource_type,
                'amount': allocation[resource_type],
                'efficiency': efficiency,
                'timestamp': time.time()
            })
        
        return allocation
    
    def monitor_usage(self) -> Dict[str, float]:
        """监控资源使用情况
        
        Returns:
            各类资源的使用率统计
        """
        usage_rates = {}
        for resource_type in self.global_resources:
            recent_usage = self.usage_stats[resource_type][-50:] if self.usage_stats[resource_type] else []
            if recent_usage:
                usage_rates[resource_type] = np.mean(recent_usage)
            else:
                usage_rates[resource_type] = 0.0
                
        return usage_rates
    
    def _calculate_efficiency(self, context: Dict[str, Any]) -> float:
        """计算当前效率指标
        
        Args:
            context: 上下文信息，包含性能指标等
            
        Returns:
            效率得分 (0.0 ~ 1.0)
        """
        if 'performance' not in context:
            return 0.5  # 默认中等效率
            
        recent_perf = context['performance'][-5:] if isinstance(context['performance'], list) else [context['performance']]
        
        # 计算性能趋势
        if len(recent_perf) > 1:
            trend = sum(b - a for a, b in zip(recent_perf[:-1], recent_perf[1:]))
        else:
            trend = 0
            
        # 结合当前性能和趋势
        base_efficiency = np.mean(recent_perf)
        trend_factor = np.clip(trend + 0.5, 0, 1)  # 将趋势归一化到[0,1]
        
        return base_efficiency * 0.7 + trend_factor * 0.3  # 权重组合
    
    def optimize_allocation(self) -> None:
        """优化资源分配策略
        
        基于历史数据优化资源分配策略
        """
        if len(self.allocation_history) < 10:
            return
            
        recent_history = self.allocation_history[-10:]
        
        # 分析资源使用效率
        for resource_type in self.global_resources:
            type_history = [h for h in recent_history if h['type'] == resource_type]
            if type_history:
                # 计算平均效率
                avg_efficiency = np.mean([h['efficiency'] for h in type_history])
                
                # 根据效率调整阈值
                if avg_efficiency > 0.8:
                    # 效率高，可以适当降低最小阈值
                    self.thresholds['min_allocation'] = max(
                        0.05,
                        self.thresholds['min_allocation'] * 0.9
                    )
                elif avg_efficiency < 0.3:
                    # 效率低，提高最小阈值
                    self.thresholds['min_allocation'] = min(
                        0.2,
                        self.thresholds['min_allocation'] * 1.1
                    )

class StateController:
    """状态控制器
    
    负责在连续状态空间中进行状态的精确控制和平滑转换。基于神经科学中的
    突触配置原理，实现从高可塑性到高效率的连续调节。
    """
    def __init__(self):
        # 当前工作点
        self.working_point = {
            'efficiency': 0.5,    # 处理效率 (0.0 ~ 1.0)
            'plasticity': 0.5,    # 可塑性 (0.0 ~ 1.0)
            'stability': 0.5      # 稳定性 (0.0 ~ 1.0)
        }
        
        # 状态历史
        self.state_history = []
        
        # 状态转换参数
        self.transition_params = {
            'learning_rate': 0.1,     # 状态调整学习率
            'momentum': 0.9,          # 状态转换动量
            'stability_threshold': 0.2 # 稳定性阈值
        }
        
        # 状态监控指标
        self.stability_metrics = {
            'efficiency_std': [],     # 效率标准差
            'plasticity_std': [],     # 可塑性标准差
            'state_changes': []       # 状态变化记录
        }
    
    def adjust_state(self, 
                    feedback: Dict[str, float], 
                    context: Dict[str, Any]) -> Dict[str, float]:
        """在连续状态空间中调整工作点
        
        基于反馈和上下文信息，精确调节系统状态。遵循突触可塑性原理，
        在效率和可塑性之间寻找最优平衡。
        
        Args:
            feedback: 包含性能、学习效果等反馈信息
            context: 包含资源使用、任务需求等上下文信息
            
        Returns:
            更新后的工作点
        """
        # 计算目标状态
        target_state = self._compute_target_state(feedback, context)
        
        # 计算状态调整量
        state_delta = {}
        for key in self.working_point:
            current = self.working_point[key]
            target = target_state[key]
            
            # 应用动量
            if len(self.state_history) > 0:
                prev_delta = self.state_history[-1][key] - (self.state_history[-2][key] if len(self.state_history) > 1 else current)
                momentum_term = self.transition_params['momentum'] * prev_delta
            else:
                momentum_term = 0
                
            # 计算调整量
            delta = self.transition_params['learning_rate'] * (target - current) + momentum_term
            
            # 应用稳定性约束
            if self._check_stability(key):
                delta *= 0.5  # 降低调整幅度
                
            state_delta[key] = delta
        
        # 更新工作点
        new_state = {}
        for key in self.working_point:
            new_state[key] = np.clip(
                self.working_point[key] + state_delta[key],
                0.0,
                1.0
            )
            
        # 记录状态历史
        self.state_history.append(dict(self.working_point))
        if len(self.state_history) > 50:
            self.state_history.pop(0)
            
        # 更新工作点
        self.working_point = new_state
        
        # 更新状态变化记录
        self._update_stability_metrics()
        
        return self.working_point
    
    def _compute_target_state(self,
                            feedback: Dict[str, float],
                            context: Dict[str, Any]) -> Dict[str, float]:
        """计算目标状态
        
        基于反馈和上下文信息，计算理想的目标状态。
        
        Args:
            feedback: 反馈信息
            context: 上下文信息
            
        Returns:
            目标状态
        """
        # 提取关键指标
        performance = feedback.get('performance', 0.5)
        learning_progress = feedback.get('learning_progress', 0.5)
        task_complexity = context.get('task_complexity', 0.5)
        
        # 计算目标效率
        target_efficiency = performance * 0.7 + task_complexity * 0.3
        
        # 计算目标可塑性
        target_plasticity = learning_progress * 0.6 + (1 - task_complexity) * 0.4
        
        # 计算目标稳定性
        target_stability = max(0.3, min(0.8, 1 - abs(target_efficiency - target_plasticity)))
        
        return {
            'efficiency': target_efficiency,
            'plasticity': target_plasticity,
            'stability': target_stability
        }
    
    def _check_stability(self, state_key: str) -> bool:
        """检查特定状态维度的稳定性
        
        Args:
            state_key: 状态维度名称
            
        Returns:
            是否需要稳定性控制
        """
        if len(self.state_history) < 5:
            return False
            
        # 计算最近状态的标准差
        recent_states = [state[state_key] for state in self.state_history[-5:]]
        std = np.std(recent_states)
        
        # 更新稳定性指标
        if state_key == 'efficiency':
            self.stability_metrics['efficiency_std'].append(std)
        elif state_key == 'plasticity':
            self.stability_metrics['plasticity_std'].append(std)
            
        # 如果标准差超过阈值，需要稳定性控制
        return std > self.transition_params['stability_threshold']
    
    def _update_stability_metrics(self) -> None:
        """更新稳定性监控指标"""
        if len(self.state_history) < 2:
            return
            
        # 计算状态变化
        prev_state = self.state_history[-2]
        curr_state = self.state_history[-1]
        
        change = sum(abs(curr_state[k] - prev_state[k]) for k in curr_state)
        self.stability_metrics['state_changes'].append(change)
        
        # 维护历史长度
        for metric_list in self.stability_metrics.values():
            if len(metric_list) > 50:
                metric_list.pop(0)
    
    def monitor_stability(self) -> Dict[str, float]:
        """监控系统稳定性
        
        Returns:
            稳定性指标统计
        """
        if not self.stability_metrics['state_changes']:
            return {
                'overall_stability': 1.0,
                'efficiency_stability': 1.0,
                'plasticity_stability': 1.0
            }
            
        # 计算整体稳定性
        recent_changes = self.stability_metrics['state_changes'][-10:]
        overall_stability = 1.0 - min(1.0, np.mean(recent_changes) / 0.3)
        
        # 计算效率稳定性
        efficiency_std = self.stability_metrics['efficiency_std'][-10:]
        efficiency_stability = 1.0 - min(1.0, np.mean(efficiency_std) / 0.2)
        
        # 计算可塑性稳定性
        plasticity_std = self.stability_metrics['plasticity_std'][-10:]
        plasticity_stability = 1.0 - min(1.0, np.mean(plasticity_std) / 0.2)
        
        return {
            'overall_stability': overall_stability,
            'efficiency_stability': efficiency_stability,
            'plasticity_stability': plasticity_stability
        }

class FeedbackSystem:
    """反馈系统
    
    实现完整的反馈收集、分析和响应机制，包括：
    1. 多维度反馈收集
    2. 反馈分析机制
    3. 反馈响应策略
    4. 反馈历史记录
    """
    def __init__(self):
        # 反馈历史记录
        self.feedback_history = {
            'performance': [],      # 性能指标历史
            'resource_usage': [],   # 资源使用历史
            'state_stability': [],  # 状态稳定性历史
            'coordination': []      # 层级协同效果历史
        }
        
        # 性能指标
        self.performance_metrics = {
            'accuracy': [],         # 准确性历史
            'efficiency': [],       # 效率历史
            'adaptability': [],     # 适应性历史
            'stability': []         # 稳定性历史
        }
        
        # 分析窗口大小
        self.analysis_windows = {
            'short_term': 5,        # 短期趋势窗口
            'medium_term': 20,      # 中期趋势窗口
            'long_term': 50         # 长期趋势窗口
        }
        
        # 异常检测阈值
        self.anomaly_thresholds = {
            'performance_drop': 0.3,    # 性能下降阈值
            'resource_spike': 0.8,      # 资源使用峰值阈值
            'stability_loss': 0.4       # 稳定性损失阈值
        }
        
    def collect_feedback(self, 
                        feedback_data: Dict[str, Any],
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """收集多维度反馈数据
        
        Args:
            feedback_data: 包含各维度反馈的字典
            context: 可选的上下文信息
            
        Returns:
            处理后的反馈数据
        """
        # 1. 提取基础指标
        performance = feedback_data.get('performance', 0.0)
        resource_usage = feedback_data.get('resource_usage', {})
        state_info = feedback_data.get('state_info', {})
        
        # 2. 计算复合指标
        processed_feedback = {
            'performance': performance,
            'resource_efficiency': self._calculate_resource_efficiency(
                performance, 
                resource_usage
            ),
            'state_stability': self._calculate_stability(state_info),
            'coordination_score': self._evaluate_coordination(
                feedback_data.get('coordination_info', {})
            )
        }
        
        # 3. 更新历史记录
        for key, value in processed_feedback.items():
            if key in self.feedback_history:
                self.feedback_history[key].append(value)
                # 维护历史长度
                if len(self.feedback_history[key]) > self.analysis_windows['long_term']:
                    self.feedback_history[key] = self.feedback_history[key][-self.analysis_windows['long_term']:]
        
        # 4. 更新性能指标
        self._update_performance_metrics(processed_feedback)
        
        return processed_feedback
        
    def analyze_feedback(self) -> Dict[str, Any]:
        """分析反馈数据，生成综合分析报告
        
        Returns:
            分析结果报告
        """
        analysis_report = {
            'trends': self._analyze_trends(),
            'anomalies': self._detect_anomalies(),
            'bottlenecks': self._identify_bottlenecks(),
            'suggestions': self._generate_suggestions()
        }
        
        return analysis_report
    
    def generate_response(self, 
                         analysis: Dict[str, Any],
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """根据分析结果生成响应策略
        
        Args:
            analysis: 分析报告
            context: 可选的上下文信息
            
        Returns:
            响应策略
        """
        response = {
            'resource_adjustments': self._generate_resource_adjustments(
                analysis['bottlenecks']
            ),
            'state_transitions': self._generate_state_transitions(
                analysis['trends']
            ),
            'optimization_directives': self._generate_optimization_directives(
                analysis
            ),
            'architecture_adjustments': self._generate_architecture_adjustments(
                analysis['anomalies']
            )
        }
        
        return response
    
    def _calculate_resource_efficiency(self,
                                    performance: float,
                                    resource_usage: Dict[str, float]) -> float:
        """计算资源使用效率
        
        Args:
            performance: 性能指标
            resource_usage: 资源使用情况
            
        Returns:
            资源效率得分
        """
        if not resource_usage:
            return 0.0
            
        total_usage = sum(resource_usage.values())
        if total_usage == 0:
            return 0.0
            
        return performance / total_usage
    
    def _calculate_stability(self, state_info: Dict[str, Any]) -> float:
        """计算系统稳定性
        
        Args:
            state_info: 状态信息
            
        Returns:
            稳定性得分
        """
        if not state_info:
            return 1.0
            
        # 计算状态变化幅度
        state_changes = state_info.get('changes', [])
        if not state_changes:
            return 1.0
            
        # 使用最近的状态变化计算稳定性
        recent_changes = state_changes[-self.analysis_windows['short_term']:]
        avg_change = sum(recent_changes) / len(recent_changes)
        
        # 将变化幅度映射到稳定性分数
        return max(0.0, 1.0 - avg_change)
    
    def _evaluate_coordination(self, coordination_info: Dict[str, Any]) -> float:
        """评估层级间的协同效果
        
        Args:
            coordination_info: 协同相关信息
            
        Returns:
            协同效果得分
        """
        if not coordination_info:
            return 0.5
            
        # 评估各个协同指标
        message_efficiency = coordination_info.get('message_efficiency', 0.5)
        resource_sharing = coordination_info.get('resource_sharing', 0.5)
        conflict_rate = coordination_info.get('conflict_rate', 0.5)
        
        # 综合评分
        return (message_efficiency + resource_sharing + (1 - conflict_rate)) / 3
    
    def _update_performance_metrics(self, feedback: Dict[str, float]) -> None:
        """更新性能指标
        
        Args:
            feedback: 处理后的反馈数据
        """
        # 更新准确性
        self.performance_metrics['accuracy'].append(
            feedback['performance']
        )
        
        # 更新效率
        self.performance_metrics['efficiency'].append(
            feedback['resource_efficiency']
        )
        
        # 更新适应性 (基于性能变化趋势)
        if len(self.performance_metrics['accuracy']) >= 2:
            perf_change = self.performance_metrics['accuracy'][-1] - \
                         self.performance_metrics['accuracy'][-2]
            adaptability = 1.0 / (1.0 + np.abs(perf_change))  # 映射到[0,1]
            self.performance_metrics['adaptability'].append(adaptability)
        else:
            self.performance_metrics['adaptability'].append(0.5)
        
        # 更新稳定性
        self.performance_metrics['stability'].append(
            feedback['state_stability']
        )
        
        # 维护历史长度
        for metric_list in self.performance_metrics.values():
            if len(metric_list) > self.analysis_windows['long_term']:
                metric_list.pop(0)
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """分析各指标的趋势
        
        Returns:
            趋势分析结果
        """
        trends = {}
        
        # 分析各个时间窗口的趋势
        for window_name, window_size in self.analysis_windows.items():
            window_trends = {}
            
            # 分析每个指标
            for metric_name, metric_values in self.performance_metrics.items():
                if len(metric_values) >= window_size:
                    recent_values = metric_values[-window_size:]
                    
                    # 计算趋势
                    trend = sum(b - a for a, b in zip(recent_values[:-1], recent_values[1:])) / (window_size - 1)
                    
                    # 计算波动性
                    volatility = np.std(recent_values)
                    
                    window_trends[metric_name] = {
                        'direction': trend,
                        'volatility': volatility,
                        'current': recent_values[-1],
                        'average': np.mean(recent_values)
                    }
            
            trends[window_name] = window_trends
        
        return trends
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常情况
        
        Returns:
            检测到的异常列表
        """
        anomalies = []
        
        # 检查性能急剧下降
        if len(self.performance_metrics['accuracy']) >= 2:
            perf_drop = self.performance_metrics['accuracy'][-2] - \
                       self.performance_metrics['accuracy'][-1]
            if perf_drop > self.anomaly_thresholds['performance_drop']:
                anomalies.append({
                    'type': 'performance_drop',
                    'severity': perf_drop,
                    'timestamp': time.time()
                })
        
        # 检查资源使用异常
        if self.feedback_history['resource_usage']:
            latest_usage = self.feedback_history['resource_usage'][-1]
            if any(usage > self.anomaly_thresholds['resource_spike'] 
                  for usage in latest_usage.values()):
                anomalies.append({
                    'type': 'resource_spike',
                    'severity': max(latest_usage.values()),
                    'timestamp': time.time()
                })
        
        # 检查稳定性问题
        if self.performance_metrics['stability']:
            latest_stability = self.performance_metrics['stability'][-1]
            if latest_stability < self.anomaly_thresholds['stability_loss']:
                anomalies.append({
                    'type': 'stability_loss',
                    'severity': 1.0 - latest_stability,
                    'timestamp': time.time()
                })
        
        return anomalies
    
    def _identify_bottlenecks(self) -> Dict[str, Any]:
        """识别系统瓶颈
        
        Returns:
            识别出的瓶颈信息
        """
        bottlenecks = {
            'resource_bottlenecks': [],
            'performance_bottlenecks': [],
            'coordination_bottlenecks': []
        }
        
        # 分析资源瓶颈
        if self.feedback_history['resource_usage']:
            recent_usage = self.feedback_history['resource_usage'][-self.analysis_windows['short_term']:]
            avg_usage = {
                resource: np.mean([usage[resource] for usage in recent_usage])
                for resource in recent_usage[0].keys()
            }
            
            # 识别高负载资源
            for resource, usage in avg_usage.items():
                if usage > 0.8:  # 资源使用率超过80%
                    bottlenecks['resource_bottlenecks'].append({
                        'resource': resource,
                        'usage': usage,
                        'severity': (usage - 0.8) / 0.2  # 映射到[0,1]
                    })
        
        # 分析性能瓶颈
        for metric_name, metric_values in self.performance_metrics.items():
            if len(metric_values) >= self.analysis_windows['short_term']:
                recent_values = metric_values[-self.analysis_windows['short_term']:]
                avg_value = np.mean(recent_values)
                
                if avg_value < 0.3:  # 性能指标过低
                    bottlenecks['performance_bottlenecks'].append({
                        'metric': metric_name,
                        'value': avg_value,
                        'severity': (0.3 - avg_value) / 0.3  # 映射到[0,1]
                    })
        
        # 分析协同瓶颈
        if self.feedback_history['coordination']:
            recent_coord = self.feedback_history['coordination'][-self.analysis_windows['short_term']:]
            avg_coord = np.mean(recent_coord)
            
            if avg_coord < 0.4:  # 协同效果不佳
                bottlenecks['coordination_bottlenecks'].append({
                    'type': 'general_coordination',
                    'value': avg_coord,
                    'severity': (0.4 - avg_coord) / 0.4  # 映射到[0,1]
                })
        
        return bottlenecks
    
    def _generate_suggestions(self) -> List[Dict[str, Any]]:
        """生成优化建议
        
        Returns:
            优化建议列表
        """
        suggestions = []
        
        # 分析趋势
        trends = self._analyze_trends()
        
        # 基于短期趋势生成建议
        if 'short_term' in trends:
            for metric_name, trend_info in trends['short_term'].items():
                if trend_info['direction'] < 0:  # 指标下降
                    suggestions.append({
                        'type': 'performance_optimization',
                        'target': metric_name,
                        'priority': abs(trend_info['direction']),
                        'suggestion': f'Optimize {metric_name} to reverse negative trend'
                    })
        
        # 基于异常检测生成建议
        anomalies = self._detect_anomalies()
        for anomaly in anomalies:
            suggestions.append({
                'type': 'anomaly_resolution',
                'target': anomaly['type'],
                'priority': anomaly['severity'],
                'suggestion': f'Resolve {anomaly["type"]} anomaly'
            })
        
        # 基于瓶颈分析生成建议
        bottlenecks = self._identify_bottlenecks()
        for bottleneck_type, bottleneck_list in bottlenecks.items():
            for bottleneck in bottleneck_list:
                suggestions.append({
                    'type': 'bottleneck_mitigation',
                    'target': bottleneck_type,
                    'priority': bottleneck.get('severity', 0.5),
                    'suggestion': f'Mitigate {bottleneck_type} bottleneck'
                })
        
        # 对建议进行优先级排序
        suggestions.sort(key=lambda x: x['priority'], reverse=True)
        
        return suggestions
    
    def _generate_resource_adjustments(self, 
                                     bottlenecks: Dict[str, Any]) -> Dict[str, float]:
        """生成资源调整建议
        
        Args:
            bottlenecks: 系统瓶颈信息
            
        Returns:
            资源调整方案
        """
        adjustments = {}
        
        # 处理资源瓶颈
        for bottleneck in bottlenecks.get('resource_bottlenecks', []):
            resource = bottleneck['resource']
            severity = bottleneck['severity']
            
            # 根据瓶颈严重程度生成调整建议
            adjustments[resource] = max(0.2, min(0.8, 1.0 - severity))
        
        return adjustments
    
    def _generate_state_transitions(self, 
                                  trends: Dict[str, Any]) -> Dict[str, Any]:
        """生成状态转换建议
        
        Args:
            trends: 趋势分析结果
            
        Returns:
            状态转换建议
        """
        transitions = {
            'target_state': {},
            'transition_speed': 0.1,
            'priority': 0.5
        }
        
        # 基于短期趋势调整目标状态
        if 'short_term' in trends:
            st_trends = trends['short_term']
            
            # 根据性能趋势调整
            if 'accuracy' in st_trends:
                acc_trend = st_trends['accuracy']
                if acc_trend['direction'] < 0:  # 性能下降
                    transitions['target_state']['plasticity'] = 0.7  # 增加可塑性
                    transitions['transition_speed'] = 0.2  # 加快转换
                    transitions['priority'] = 0.8
                elif acc_trend['direction'] > 0:  # 性能提升
                    transitions['target_state']['stability'] = 0.7  # 增加稳定性
                    transitions['transition_speed'] = 0.1  # 减缓转换
                    transitions['priority'] = 0.6
        
        return transitions
    
    def _generate_optimization_directives(self, 
                                        analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成优化指令
        
        Args:
            analysis: 分析报告
            
        Returns:
            优化指令列表
        """
        directives = []
        
        # 基于建议生成优化指令
        for suggestion in analysis.get('suggestions', []):
            if suggestion['type'] == 'performance_optimization':
                directives.append({
                    'type': 'optimization',
                    'target': suggestion['target'],
                    'action': 'adjust_parameters',
                    'parameters': {
                        'learning_rate': 0.1,
                        'exploration_rate': 0.2
                    }
                })
            elif suggestion['type'] == 'bottleneck_mitigation':
                directives.append({
                    'type': 'mitigation',
                    'target': suggestion['target'],
                    'action': 'reallocate_resources',
                    'parameters': {
                        'allocation_priority': 0.8,
                        'reallocation_ratio': 0.3
                    }
                })
        
        return directives
    
    def _generate_architecture_adjustments(self, 
                                         anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成架构调整建议
        
        Args:
            anomalies: 检测到的异常
            
        Returns:
            架构调整建议列表
        """
        adjustments = []
        
        # 基于异常生成架构调整建议
        for anomaly in anomalies:
            if anomaly['type'] == 'performance_drop':
                adjustments.append({
                    'type': 'architecture_optimization',
                    'target': 'processing_pipeline',
                    'action': 'add_capacity',
                    'parameters': {
                        'capacity_increase': 0.3,
                        'target_component': 'processing_units'
                    }
                })
            elif anomaly['type'] == 'resource_spike':
                adjustments.append({
                    'type': 'architecture_optimization',
                    'target': 'resource_management',
                    'action': 'optimize_allocation',
                    'parameters': {
                        'allocation_strategy': 'dynamic',
                        'buffer_size': 0.2
                    }
                })
        
        return adjustments

class ThinkingLevel(nn.Module):
    """思维层次基类重构
    
    TODO:
    1. 集成资源管理
    2. 集成状态控制
    3. 集成反馈系统
    4. 实现层次间协同机制
    """
    def __init__(self, level_id):
        super().__init__()
        self.level_id = level_id
        self.resource_manager = ResourceManager()
        self.state_controller = StateController()
        self.feedback_system = FeedbackSystem()
        
    def process(self, input_data: torch.Tensor, context: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """处理输入数据
        
        Args:
            input_data: 输入数据
            context: 上下文信息
            
        Returns:
            处理结果和置信度
        """
        start_time = time.time()
        
        # 获取资源分配
        resources = self.resource_manager.allocate(
            demand={
                'computation': 0.5,
                'memory': 0.3,
                'attention': 0.2
            },
            context=context
        )
        
        # 获取当前状态
        state_feedback = self.state_controller.monitor_stability()
        
        # 处理数据
        output = self._process_impl(input_data, context)
        
        # 计算置信度
        confidence = self._calculate_confidence(output)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 更新状态
        self.state = ThinkingState(
            level=self.level_id,
            confidence=confidence,
            attention_weights=None,  # 由子类设置
            memory_context=None,     # 由子类设置
            configuration=context.get('config'),
            resource_usage=sum(resources.values()),
            processing_time=processing_time,
            output_norm=torch.norm(output).item()
        )
        
        return output, confidence
        
    def _process_impl(self, input_data: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """实际的处理逻辑
        
        由子类实现
        
        Args:
            input_data: 输入数据
            context: 上下文信息
            
        Returns:
            处理结果
        """
        raise NotImplementedError
        
    def _calculate_confidence(self, output: torch.Tensor) -> float:
        """计算置信度
        
        由子类实现
        
        Args:
            output: 处理结果
            
        Returns:
            置信度
        """
        return 0.5  # 默认中等置信度
        
    def update(self, feedback: float) -> None:
        """更新处理策略
        
        Args:
            feedback: 外部反馈
        """
        # 收集反馈
        self.feedback_system.collect_feedback({
            'performance': feedback,
            'resource_usage': self.state.resource_usage,
            'processing_time': self.state.processing_time
        })
        
        # 分析反馈
        analysis = self.feedback_system.analyze_feedback()
        
        # 调整状态
        self.state_controller.adjust_state(
            feedback={'performance': feedback},
            context=analysis
        )
        
        # 优化资源分配
        self.resource_manager.optimize_allocation()

class FastIntuitionLevel(ThinkingLevel):
    """快速直觉处理层
    
    这一层负责快速、直觉性的处理，特点是:
    1. 结构简单，处理速度快
    2. 适合处理模式识别等直觉性任务
    3. 使用浅层网络以保证响应速度
    """
    
    def __init__(self, level_id: int, config: SystemConfig):
        super().__init__(level_id)
        
        # 使用简单的前馈网络
        self.network = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.input_size)  # 确保输出维度与输入相同
        )
        
        # 置信度评估器
        self.confidence_net = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 性能历史记录
        self.performance_history = []
        self.confidence_history = []
        
        # 优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.network.parameters(), 'lr': config.base_learning_rate},
            {'params': self.confidence_net.parameters(), 'lr': config.base_learning_rate * 0.5}
        ])
        
        # 配置状态管理器
        self.config_manager = ConfigStateManager()
        
    def _process_impl(self, input_data: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """快速处理输入数据
        
        维度流转:
        1. 输入: [batch_size, input_size] 或 [batch_size, 1, input_size]
        2. 快速处理: [batch_size, input_size]
        3. 输出: [batch_size, input_size]
        
        Args:
            input_data: 输入张量 [batch_size, input_size] 或 [batch_size, 1, input_size]
            context: 上下文信息
            
        Returns:
            处理后的输出 [batch_size, input_size]
        """
        # 如果输入是3维的,压缩到2维
        if input_data.dim() == 3:
            input_data = input_data.squeeze(1)  # [batch_size, 1, input_size] -> [batch_size, input_size]
        elif input_data.dim() != 2:
            raise ValueError(f"输入维度错误: 期望2维或3维,实际是{input_data.dim()}维")
        
        # 快速前向处理
        output = self.network(input_data)  # [batch_size, input_size]
        
        return output
        
    def _calculate_confidence(self, output: torch.Tensor) -> float:
        """计算置信度
        
        Args:
            output: 处理结果
            
        Returns:
            置信度
        """
        # 计算置信度
        batch_confidences = self.confidence_net(output)  # [batch_size, 1]
        confidence = torch.mean(batch_confidences).item()  # 平均置信度
        
        # 记录置信度
        self.confidence_history.append(confidence)
        
        return confidence

class AnalyticalLevel(ThinkingLevel):
    """分析处理层
    
    负责深入分析和处理，特点是:
    1. 使用注意力机制关注重要信息
    2. 多层次信息处理
    3. 动态资源分配
    """
    
    def __init__(self, level_id: int, config: SystemConfig):
        super().__init__(level_id)
        
        # 核心网络组件
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.attention_heads
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=config.dropout
        )
        
        # 特征提取网络
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 分析网络
        self.analysis_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size, self.input_size)
        )
        
        # 置信度评估器
        self.confidence_net = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.base_learning_rate)
        
        # 状态记录
        self.attention_weights = None
        self.feature_importance = None
    
    def _process_impl(self, input_data: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """处理输入数据
        
        维度流转:
        1. 输入: [batch_size, input_size] 或 [batch_size, 1, input_size]
        2. 特征: [batch_size, hidden_size]
        3. 注意力: [seq_len, batch_size, hidden_size]
        4. 输出: [batch_size, input_size]
        
        Args:
            input_data: 输入数据
            context: 上下文信息
            
        Returns:
            处理后的输出
        """
        # 1. 确保输入是2维的
        if input_data.dim() == 3:
            input_data = input_data.squeeze(1)  # [batch_size, 1, input_size] -> [batch_size, input_size]
        elif input_data.dim() != 2:
            raise ValueError(f"输入维度错误: 期望2维或3维(中间维度为1), 实际是{input_data.dim()}维")
            
        batch_size = input_data.size(0)
        
        # 2. 特征提取
        features = self.feature_net(input_data)  # [batch_size, hidden_size]
        
        # 3. 准备注意力输入
        # 调整维度顺序为 [seq_len, batch_size, hidden_size]
        query = features.unsqueeze(0)  # [1, batch_size, hidden_size]
        key = query                    # [1, batch_size, hidden_size]
        value = query                  # [1, batch_size, hidden_size]
        
        # 4. 应用注意力机制
        attended_features, attention_weights = self.attention(
            query=query,
            key=key,
            value=value,
            need_weights=True
        )  # attended_features: [1, batch_size, hidden_size]
           # attention_weights: [batch_size, 1, 1]
        
        # 5. 保存注意力权重用于置信度计算
        # 确保attention_weights的维度是 [batch_size, 1]
        if attention_weights.dim() == 3:
            self.attention_weights = attention_weights.squeeze(1)  # [batch_size, 1]
        else:
            self.attention_weights = attention_weights.unsqueeze(1)  # [batch_size, 1]
        
        # 6. 处理注意力输出
        # 确保attended_features的维度是 [batch_size, hidden_size]
        if attended_features.dim() == 3:
            attended_features = attended_features.squeeze(0)  # [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        
        # 7. 分析处理
        output = self.analysis_net(attended_features)  # [batch_size, input_size]
        
        # 8. 验证输出维度
        assert output.dim() == 2 and output.size(0) == batch_size and output.size(1) == self.input_size, \
            f"输出维度错误: 期望[{batch_size}, {self.input_size}], 实际是{output.size()}"
        
        return output
        
    def _calculate_confidence(self, output: torch.Tensor) -> float:
        """计算处理结果的置信度
        
        维度流转:
        1. 输入: [batch_size, input_size]
        2. 注意力权重: [batch_size, 1]
        3. 注意力上下文: [batch_size, input_size]
        4. 置信度输入: [batch_size, input_size * 2]
        5. 置信度输出: [batch_size, 1]
        
        Args:
            output: 处理结果 [batch_size, input_size]
            
        Returns:
            置信度值 (0.0 ~ 1.0)
        """
        batch_size = output.size(0)
        
        # 1. 确保输入维度正确
        assert output.dim() == 2 and output.size(1) == self.input_size, \
            f"输入维度错误: 期望[batch_size, {self.input_size}], 实际是{output.size()}"
        
        # 2. 确保注意力权重维度正确
        if self.attention_weights is None:
            self.attention_weights = torch.ones(batch_size, 1, device=output.device)
        elif self.attention_weights.size(0) != batch_size:
            # 如果batch_size不匹配，调整注意力权重
            self.attention_weights = self.attention_weights.expand(batch_size, 1)
        
        # 3. 计算注意力上下文
        attention_context = output * self.attention_weights  # [batch_size, input_size]
        
        # 4. 组合特征用于置信度评估
        confidence_features = torch.cat([
            output,            # [batch_size, input_size]
            attention_context  # [batch_size, input_size]
        ], dim=-1)  # [batch_size, input_size * 2]
        
        # 5. 计算置信度
        confidence = self.confidence_net(confidence_features)  # [batch_size, 1]
        
        # 6. 返回平均置信度
        return torch.mean(confidence).item()
        
    def update(self, feedback: float) -> None:
        """根据反馈更新处理策略
        
        Args:
            feedback: 外部反馈值 (-1.0 ~ 1.0)
        """
        # 1. 收集性能指标
        performance_metrics = {
            'feedback': feedback,
            'confidence': self.state.confidence,
            'resource_usage': self.state.resource_usage,
            'processing_time': self.state.processing_time
        }
        
        # 2. 分析反馈
        feedback_analysis = self.feedback_system.collect_feedback(performance_metrics)
        
        # 3. 根据分析结果调整状态
        self.state_controller.adjust_state(
            feedback={'performance': feedback},
            context=feedback_analysis
        )
        
        # 4. 优化资源分配
        self.resource_manager.optimize_allocation()
        
        # 5. 根据反馈调整学习率
        if feedback < 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.9  # 降低学习率
        elif feedback > 0.8:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 1.1  # 提高学习率

class AbstractReasoningLevel(ThinkingLevel):
    """抽象推理层
    
    这一层负责高层次的抽象推理，特点是:
    1. 层次化的概念表示
    2. 逻辑关系推导能力
    3. 规则学习和应用
    4. 概念迁移能力
    """
    
    def __init__(self, level_id: int, config: LevelConfig):
        super().__init__(level_id)
        self.config = config
        
        # 概念编码器
        self.concept_encoder = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 关系推理模块
        self.relation_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ) for _ in range(3)  # 多层关系推理
        ])
        
        # 规则生成器
        self.rule_generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
        # 推理应用器
        self.reasoning_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.input_size)
        )
        
        # 置信度评估器
        self.confidence_net = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 规则库
        self.rule_memory = nn.Parameter(
            torch.randn(config.hidden_size, config.hidden_size)
        )
        
        # 性能历史记录
        self.performance_history = []
        self.confidence_history = []
        self.rule_usage_history = []
        
        # 优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.concept_encoder.parameters(), 'lr': 1e-3},
            {'params': self.relation_net.parameters(), 'lr': 1e-3},
            {'params': self.rule_generator.parameters(), 'lr': 5e-4},
            {'params': self.reasoning_net.parameters(), 'lr': 5e-4},
            {'params': self.confidence_net.parameters(), 'lr': 5e-4},
            {'params': self.rule_memory, 'lr': 1e-4}
        ])
        
        # 配置状态管理器
        self.config_manager = ConfigStateManager()
        
        # 添加层级间通信缓冲区
        self.communication_buffer = {
            'bottom_up': [],    # 来自低层的信息
            'top_down': [],     # 来自高层的控制信号
            'lateral': []       # 同层交互信息
        }
        
        # 添加资源监控
        self.resource_usage = {
            'computation': [],  # 计算资源使用
            'memory': [],      # 内存使用
            'attention': []    # 注意力资源使用
        }
        
        # 添加性能指标
        self.metrics = {
            'rule_efficiency': [],     # 规则使用效率
            'concept_clarity': [],     # 概念表示清晰度
            'inference_quality': [],   # 推理质量
            'resource_efficiency': []  # 资源使用效率
        }
    
    def _process_impl(self, input_data: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """处理输入数据（增加资源跟踪）"""
        start_time = time.time()
        
        # 获取当前可用资源
        available_resources = context.get('resources', {
            'computation': 1.0,
            'memory': 1.0,
            'attention': 1.0
        })
        
        # 处理来自其他层级的信息
        self._process_communications()
        
        # 原有的处理逻辑...
        output, confidence = super().process(input_data, context)
        
        # 更新资源使用情况
        processing_time = time.time() - start_time
        self.update_resource_usage({
            'computation': processing_time / context.get('time_budget', 1.0),
            'memory': sys.getsizeof(self.rule_memory.data) / (1024 * 1024),  # MB
            'attention': torch.mean(torch.abs(self.rule_memory)).item()
        })
        
        # 计算性能指标
        self.calculate_metrics()
        
        return output, confidence
        
    def _process_communications(self) -> None:
        """处理层级间的通信"""
        # 处理自下而上的信息
        if self.communication_buffer['bottom_up']:
            bottom_up_info = self.communication_buffer['bottom_up'][-1]
            # 根据低层信息调整概念编码器
            if 'feature_importance' in bottom_up_info:
                self._adjust_concept_encoding(bottom_up_info['feature_importance'])
                
        # 处理自上而下的控制信号
        if self.communication_buffer['top_down']:
            top_down_control = self.communication_buffer['top_down'][-1]
            # 根据高层控制调整推理策略
            if 'strategy_adjustment' in top_down_control:
                self._adjust_reasoning_strategy(top_down_control['strategy_adjustment'])
                
        # 处理同层交互
        if self.communication_buffer['lateral']:
            lateral_info = self.communication_buffer['lateral'][-1]
            # 与同层交换规则信息
            if 'shared_rules' in lateral_info:
                self._exchange_rules(lateral_info['shared_rules'])
                
    def _adjust_concept_encoding(self, feature_importance: torch.Tensor) -> None:
        """根据特征重要性调整概念编码"""
        with torch.no_grad():
            # 调整输入层权重
            first_layer = next(self.concept_encoder.parameters())
            importance_mask = feature_importance.unsqueeze(0).expand_as(first_layer)
            first_layer.data *= importance_mask
            
    def _adjust_reasoning_strategy(self, strategy: Dict[str, Any]) -> None:
        """根据高层控制调整推理策略"""
        if 'rule_generation_threshold' in strategy:
            # 调整规则生成阈值
            self.rule_generation_threshold = strategy['rule_generation_threshold']
            
        if 'inference_temperature' in strategy:
            # 调整推理温度
            self.inference_temperature = strategy['inference_temperature']
            
    def _exchange_rules(self, shared_rules: torch.Tensor) -> None:
        """与同层交换规则信息
        
        Args:
            shared_rules: 共享规则张量,维度为 [batch_size, hidden_size]
        """
        with torch.no_grad():
            # 确保规则内存和共享规则的维度匹配
            if len(self.rule_memory.shape) == 2:
                rule_memory = self.rule_memory
            else:
                rule_memory = self.rule_memory.squeeze(0)  # [1, hidden_size] -> [hidden_size]
                
            if len(shared_rules.shape) == 2:
                rules = shared_rules
            else:
                rules = shared_rules.squeeze(0)  # [1, hidden_size] -> [hidden_size]
                
            # 计算规则相似度
            rule_similarity = torch.matmul(rule_memory.unsqueeze(0), rules.T)  # [1, batch_size]
            
            # 评估规则有用性
            useful_rules = torch.max(rule_similarity, dim=1)[0] < 0.8
            
            if torch.any(useful_rules):
                # 整合有用的规则
                useful_shared_rules = rules[useful_rules]
                self.rule_memory = torch.cat([
                    rule_memory,
                    useful_shared_rules
                ], dim=0).unsqueeze(0)  # 保持 [1, hidden_size] 维度
                
                # 限制规则库大小
                if self.rule_memory.size(1) > self.max_rules:
                    self.rule_memory = self.rule_memory[:, :self.max_rules]
    
    def calculate_metrics(self) -> Dict[str, float]:
        """计算当前性能指标"""
        if not self.rule_usage_history:
            return {key: 0.0 for key in self.metrics}
            
        # 计算规则使用效率
        recent_usage = torch.stack(self.rule_usage_history[-5:])
        rule_efficiency = torch.mean(torch.max(recent_usage, dim=-1)[0]).item()
        
        # 计算概念表示清晰度
        concept_vectors = self.concept_encoder(torch.randn(10, self.config.input_size))
        rule_importance = recent_usage.mean(dim=[0, 1])  # 计算规则重要性
        
        # 计算规则使用频率和有效性
        rule_frequency = torch.sum(recent_usage > 0.5, dim=[0, 1]) / (recent_usage.size(0) * recent_usage.size(1))
        rule_effectiveness = rule_importance * performance
        
        # 更新规则权重
        with torch.no_grad():
            # 增强高效规则
            effective_rules = rule_effectiveness > rule_effectiveness.mean()
            self.rule_memory.data[effective_rules] *= 1.1
            
            # 弱化低效规则
            ineffective_rules = (rule_effectiveness < rule_effectiveness.mean() * 0.5) & (rule_frequency > 0.1)
            self.rule_memory.data[ineffective_rules] *= 0.9
            
            # 重新初始化未使用的规则
            unused_rules = rule_frequency < 0.05
            if torch.any(unused_rules):
                self.rule_memory.data[unused_rules] = torch.randn_like(
                    self.rule_memory.data[unused_rules]
                ) * 0.1
        
        # 分析性能趋势
        if len(self.performance_history) >= 3:
            recent_perf = self.performance_history[-3:]
            trend = sum(b - a for a, b in zip(recent_perf[:-1], recent_perf[1:]))
            
            # 根据趋势调整网络
            if trend < 0:  # 性能下降
                # 增加学习力度
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 1.2
                    
                # 增加规则生成的多样性
                with torch.no_grad():
                    # 计算规则相似度矩阵
                    rule_similarity = torch.matmul(
                        self.rule_memory, 
                        self.rule_memory.T
                    )
                    # 找出高度相似的规则
                    similar_pairs = torch.where(
                        (rule_similarity > 0.9) & 
                        (torch.eye(rule_similarity.size(0), device=rule_similarity.device) == 0)
                    )
                    # 对高度相似的规则添加差异性
                    if similar_pairs[0].size(0) > 0:
                        noise = torch.randn_like(
                            self.rule_memory[similar_pairs[0]]
                        ) * 0.2
                        self.rule_memory.data[similar_pairs[0]] += noise
                    
            elif trend > 0:  # 性能提升
                # 保持当前策略，但适当降低学习率以稳定表现
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.98
            else:  # 性能稳定
                # 适当降低学习力度
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.95
        
        # 分析置信度
        if len(self.confidence_history) >= 5:
            recent_conf = torch.tensor(self.confidence_history[-5:])
            conf_mean = torch.mean(recent_conf).item()
            conf_std = torch.std(recent_conf).item()
            
            # 如果置信度波动较大，增加稳定性
            if conf_std > 0.2:
                current_config = self.config_manager.get_current_config()
                for module in self.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = min(module.p + 0.05, current_config.hidden_dropout + 0.1)
            
            # 动态调整规则库
            if conf_mean < 0.3:  # 置信度持续较低
                with torch.no_grad():
                    # 计算规则覆盖度
                    rule_coverage = torch.sum(
                        torch.abs(self.rule_memory), 
                        dim=1
                    )
                    # 找出覆盖度低的规则
                    low_coverage = rule_coverage < rule_coverage.mean() * 0.5
                    
                    if torch.any(low_coverage):
                        # 移除覆盖度低的规则
                        high_coverage_rules = self.rule_memory.data[~low_coverage]
                        
                        # 生成新规则
                        new_rules = torch.randn(
                            low_coverage.sum(),
                            self.rule_memory.size(1)
                        ).to(self.rule_memory.device) * 0.1
                        
                        # 更新规则库
                        self.rule_memory.data = torch.cat([
                            high_coverage_rules,
                            new_rules
                        ], dim=0)
            elif conf_mean > 0.8:  # 置信度持续较高
                # 尝试优化规则间的关系
                with torch.no_grad():
                    # 计算规则间的相关性
                    rule_correlation = torch.corrcoef(self.rule_memory.T)
                    # 找出高度相关的规则对
                    high_correlation = torch.where(
                        (torch.abs(rule_correlation) > 0.9) & 
                        (torch.eye(rule_correlation.size(0), device=rule_correlation.device) == 0)
                    )
                    # 合并高度相关的规则
                    if high_correlation[0].size(0) > 0:
                        unique_rules = []
                        merged = set()
                        for i, j in zip(*high_correlation):
                            i, j = i.item(), j.item()
                            if i not in merged and j not in merged:
                                # 合并规则
                                merged_rule = (self.rule_memory.data[i] + self.rule_memory.data[j]) / 2
                                unique_rules.append(merged_rule)
                                merged.add(i)
                                merged.add(j)
                            elif i not in merged:
                                unique_rules.append(self.rule_memory.data[i])
                                merged.add(i)
                            elif j not in merged:
                                unique_rules.append(self.rule_memory.data[j])
                                merged.add(j)
                        
                        # 添加一些新规则以保持规则库大小
                        remaining_count = self.rule_memory.size(0) - len(unique_rules)
                        if remaining_count > 0:
                            new_rules = torch.randn(
                                remaining_count,
                                self.rule_memory.size(1)
                            ).to(self.rule_memory.device) * 0.1
                            unique_rules.extend([new_rules[i] for i in range(remaining_count)])
                        
                        # 更新规则库
                        self.rule_memory.data = torch.stack(unique_rules)
        
        # 清理历史记录（保留最近50条）
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
        if len(self.confidence_history) > 50:
            self.confidence_history = self.confidence_history[-50:]
        if len(self.rule_usage_history) > 50:
            self.rule_usage_history = self.rule_usage_history[-50:]

class MetaCognitiveLevel(ThinkingLevel):
    """元认知控制层
    
    这一层负责整体的认知控制，特点是:
    1. 监控和调节其他层次的行为
    2. 负责资源分配和策略选择
    3. 具有全局视角的决策能力
    """
    
    def __init__(self, level_id: int, config: LevelConfig):
        super().__init__(level_id)
        self.config = config
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 策略网络
        self.strategy_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.input_size)
        )
        
        # 资源分配网络
        self.resource_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 3)  # 为其他三层分配资源
        )
        
        # 置信度评估器
        self.confidence_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 性能历史记录
        self.performance_history = []
        self.confidence_history = []
        
        # 优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.state_encoder.parameters(), 'lr': 1e-3},
            {'params': self.strategy_net.parameters(), 'lr': 5e-4},
            {'params': self.resource_net.parameters(), 'lr': 5e-4},
            {'params': self.confidence_net.parameters(), 'lr': 5e-4}
        ])
        
        # 配置状态管理器
        self.config_manager = ConfigStateManager()
        
    def process(self,
                input_data: torch.Tensor,
                context: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """执行元认知控制
        
        维度流转:
        1. 输入: [batch_size, input_size] 或 [batch_size, 1, input_size]
        2. 统一到: [batch_size, input_size]
        3. 状态编码: [batch_size, hidden_size]
        4. 历史编码: [batch_size, hidden_size]
        5. 输出: [batch_size, input_size]
        
        Args:
            input_data: 输入张量 [batch_size, input_size] 或 [batch_size, 1, input_size]
            context: 上下文信息（包含其他层的状态）
            
        Returns:
            控制决策和置信度
        """
        # 获取当前配置
        current_config = self.config_manager.get_current_config()
        
        # 1. 确保输入是2维 [batch_size, input_size]
        if input_data.dim() == 3:
            input_data = input_data.squeeze(1)  # [batch_size, 1, input_size] -> [batch_size, input_size]
        elif input_data.dim() != 2:
            raise ValueError(f"输入维度错误: 期望2维或3维，实际是{input_data.dim()}维")
        
        # 2. 编码当前状态
        state_encoding = self.state_encoder(input_data)  # [batch_size, hidden_size]
        
        # 3. 获取历史状态信息
        history_encoding = torch.zeros_like(state_encoding)  # [batch_size, hidden_size]
        if context.get("history_states"):
            history_states = torch.stack(context["history_states"])  # [num_states, batch_size, input_size]
            if history_states.dim() == 4:  # 如果历史状态是3维的
                history_states = history_states.squeeze(2)  # [num_states, batch_size, input_size]
            history_encoding = self.state_encoder(
                history_states.mean(dim=0)  # [batch_size, input_size]
            )  # [batch_size, hidden_size]
        
        # 4. 生成策略决策
        strategy_input = torch.cat([
            state_encoding,     # [batch_size, hidden_size]
            history_encoding    # [batch_size, hidden_size]
        ], dim=-1)  # [batch_size, hidden_size * 2]
        output = self.strategy_net(strategy_input)  # [batch_size, input_size]
        
        # 5. 分配资源
        resource_allocation = torch.softmax(
            self.resource_net(state_encoding),  # [batch_size, 3]
            dim=-1
        )
        
        # 6. 计算置信度
        confidence_input = torch.cat([
            state_encoding,     # [batch_size, hidden_size]
            history_encoding    # [batch_size, hidden_size]
        ], dim=-1)  # [batch_size, hidden_size * 2]
        confidence = torch.mean(self.confidence_net(confidence_input)).item()
        
        # 记录置信度
        self.confidence_history.append(confidence)
        
        # 更新状态
        self.state = ThinkingState(
            level=self.level_id,
            confidence=confidence,
            attention_weights=None,
            memory_context={"resource_allocation": resource_allocation},
            configuration=current_config
        )
        
        return output, confidence
        
    def update(self, performance: float) -> None:
        """根据性能反馈更新控制策略
        
        这个方法实现元认知层的自适应调整，包括:
        1. 策略评估和优化
        2. 资源分配策略调整
        3. 控制参数动态调整
        
        Args:
            performance: 性能评分 (0.0 ~ 1.0)
        """
        # 记录性能
        self.performance_history.append(performance)
        self.config_manager.record_performance(performance)
        
        # 检查是否需要状态转换
        if self.config_manager.should_transition():
            next_state = self.config_manager.get_next_state()
            if next_state != self.config_manager.current_state:
                self.config_manager.current_state = next_state
                new_config = self.config_manager.get_current_config()
                
                # 更新dropout
                for module in self.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = new_config.hidden_dropout
                
                # 更新优化器学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_config.learning_rate
        
        # 分析性能趋势
        if len(self.performance_history) >= 3:
            recent_perf = self.performance_history[-3:]
            trend = sum(b - a for a, b in zip(recent_perf[:-1], recent_perf[1:]))
            
            # 根据趋势调整策略网络
            if trend < 0:  # 性能下降
                # 增加探索性
                for param_group in self.optimizer.param_groups:
                    if 'strategy_net' in str(param_group['params'][0].shape):
                        param_group['lr'] *= 1.2
                
                # 增加资源分配的灵活性
                with torch.no_grad():
                    for param in self.resource_net.parameters():
                        if param.requires_grad:
                            param.data *= 1.1
                            
            elif trend > 0:  # 性能提升
                # 保持当前策略，但适当降低学习率以稳定表现
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.98
                    
            else:  # 性能稳定
                # 适当降低学习力度
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.95
                    
        # 分析置信度
        if len(self.confidence_history) >= 5:
            recent_conf = torch.tensor(self.confidence_history[-5:])
            conf_mean = torch.mean(recent_conf).item()
            conf_std = torch.std(recent_conf).item()
            
            # 如果置信度波动较大，增加稳定性
            if conf_std > 0.2:
                current_config = self.config_manager.get_current_config()
                for module in self.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = min(module.p + 0.05, current_config.hidden_dropout + 0.1)
            
            # 根据平均置信度调整策略
            if conf_mean < 0.3:  # 置信度持续较低
                # 增强状态编码器的表达能力
                with torch.no_grad():
                    for param in self.state_encoder.parameters():
                        if param.requires_grad:
                            param.data *= 1.1
                            
                # 增加资源分配的确定性
                with torch.no_grad():
                    for param in self.resource_net.parameters():
                        if param.requires_grad:
                            param.data *= 0.9
                            
            elif conf_mean > 0.8:  # 置信度持续较高
                # 尝试优化资源使用
                with torch.no_grad():
                    # 计算资源分配的熵
                    resource_dist = torch.softmax(
                        self.resource_net(torch.randn(1, self.config.hidden_size)),
                        dim=-1
                    )
                    entropy = -torch.sum(resource_dist * torch.log(resource_dist + 1e-10))
                    
                    # 如果资源分配过于分散，增加集中度
                    if entropy > 0.8:
                        for param in self.resource_net.parameters():
                            if param.requires_grad:
                                param.data *= 0.95
                                
        # 优化策略生成
        if len(self.performance_history) >= 5:
            recent_perf = self.performance_history[-5:]
            avg_perf = sum(recent_perf) / len(recent_perf)
            
            # 如果性能持续较低，增强策略网络的容量
            if avg_perf < 0.3:
                with torch.no_grad():
                    # 增强策略网络中间层的表达能力
                    for module in self.strategy_net:
                        if isinstance(module, nn.Linear):
                            # 增加权重范围
                            module.weight.data *= 1.1
                            if module.bias is not None:
                                module.bias.data *= 1.1
            
            # 如果性能稳定在高水平，优化策略的效率
            elif avg_perf > 0.8:
                with torch.no_grad():
                    # 计算策略网络参数的稀疏度
                    total_params = 0
                    small_params = 0
                    for module in self.strategy_net:
                        if isinstance(module, nn.Linear):
                            total_params += module.weight.numel()
                            small_params += torch.sum(torch.abs(module.weight) < 0.01).item()
                    
                    sparsity = small_params / total_params
                    
                    # 如果网络不够稀疏，增加L1正则化效果
                    if sparsity < 0.3:
                        for module in self.strategy_net:
                            if isinstance(module, nn.Linear):
                                module.weight.data *= torch.where(
                                    torch.abs(module.weight.data) < 0.01,
                                    torch.tensor(0.9),
                                    torch.tensor(1.0)
                                )
        
        # 清理历史记录（保留最近50条）
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
        if len(self.confidence_history) > 50:
            self.confidence_history = self.confidence_history[-50:]

# TODO: 系统集成
class ThinkingSystem:
    """整体思维系统
    
    负责协调各个思维层次的协同工作，优化资源分配，
    并通过反馈系统实现整体的自适应调节。基于神经科学中的
    分层控制和动态平衡原理构建。
    """
    def __init__(self, config: SystemConfig):
        # 基础组件
        self.resource_manager = ResourceManager()
        self.state_controller = StateController()
        self.feedback_system = FeedbackSystem()
        
        # 思维层次初始化
        self.levels = {
            'fast_intuition': FastIntuitionLevel(0, LevelConfig(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                attention_heads=config.attention_heads
            )),
            'analytical': AnalyticalLevel(1, LevelConfig(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                attention_heads=config.attention_heads
            )),
            'abstract_reasoning': AbstractReasoningLevel(2, LevelConfig(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                attention_heads=config.attention_heads
            )),
            'metacognitive': MetaCognitiveLevel(3, LevelConfig(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                attention_heads=config.attention_heads
            ))
        }
        
        # 层次间通信通道
        self.communication_channels = {
            'bottom_up': [],    # 自下而上的信息流
            'top_down': [],     # 自上而下的控制流
            'lateral': []       # 同层交互
        }
        
        # 系统状态
        self.system_state = {
            'active_levels': set(),       # 当前活跃的层次
            'processing_depth': 0,        # 当前处理深度
            'resource_distribution': {},   # 资源分布
            'performance_metrics': {}     # 性能指标
        }
        
        # 协同参数
        self.coordination_params = {
            'activation_threshold': 0.3,   # 层次激活阈值
            'inhibition_strength': 0.2,    # 抑制强度
            'feedback_strength': 0.5,      # 反馈强度
            'integration_rate': 0.1        # 整合速率
        }
    
    def process(self, 
                input_data: torch.Tensor,
                context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """处理输入数据
        
        维度处理:
        1. 输入: [batch_size, input_size]
        2. 内部处理: [batch_size, 1, input_size]
        3. 输出: [batch_size, input_size]
        
        实现多层次的协同处理，包括:
        1. 动态资源分配
        2. 层次间的信息流动
        3. 结果整合与优化
        
        Args:
            input_data: 输入数据 [batch_size, input_size]
            context: 上下文信息
            
        Returns:
            处理结果和处理状态
        """
        context = context or {}
        processing_results = {}
        
        # 确保输入是2维的
        if input_data.dim() == 3:
            input_data = input_data.squeeze(1)
        elif input_data.dim() != 2:
            raise ValueError(f"输入维度错误: 期望2维或3维，实际是{input_data.dim()}维")
        
        # 1. 准备阶段
        self._prepare_processing(input_data, context)
        
        # 2. 自下而上的处理
        bottom_up_results = self._bottom_up_processing(input_data, context)
        
        # 3. 自上而下的调控
        top_down_control = self._top_down_control(bottom_up_results, context)
        
        # 4. 整合结果
        final_result = self._integrate_results(bottom_up_results, top_down_control)
        
        # 确保输出是2维的
        if final_result.dim() == 3:
            final_result = final_result.squeeze(1)
        
        # 5. 更新系统状态
        self._update_system_state(final_result, bottom_up_results)
        
        return final_result, self.system_state
    
    def _prepare_processing(self, 
                          input_data: torch.Tensor,
                          context: Dict[str, Any]) -> None:
        """准备处理阶段
        
        配置系统状态，分配初始资源
        """
        # 评估任务复杂度
        task_complexity = self._evaluate_task_complexity(input_data)
        
        # 初始资源分配
        initial_resources = self.resource_manager.allocate(
            demand={
                'computation': task_complexity,
                'memory': task_complexity * 0.8,
                'attention': task_complexity * 0.6
            },
            context=context
        )
        
        # 更新系统状态
        self.system_state['resource_distribution'] = initial_resources
        self.system_state['processing_depth'] = task_complexity
        
        # 激活相关层次
        self._activate_relevant_levels(task_complexity)
    
    def _bottom_up_processing(self,
                            input_data: torch.Tensor,
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """自下而上的处理流程
        
        维度处理:
        1. 输入: [batch_size, input_size]
        2. 内部处理: [batch_size, 1, input_size]
        3. 每层输出: [batch_size, input_size]
        4. 传递给下一层前: [batch_size, 1, input_size]
        
        Args:
            input_data: 输入数据 [batch_size, input_size]
            context: 上下文信息
            
        Returns:
            各层处理结果
        """
        results = {}
        
        # 确保输入是2维的
        if input_data.dim() == 3:
            current_input = input_data.squeeze(1)
        elif input_data.dim() == 2:
            current_input = input_data
        else:
            raise ValueError(f"输入维度错误: 期望2维或3维，实际是{input_data.dim()}维")
        
        # 逐层处理
        for level_name, level in self.levels.items():
            if level_name not in self.system_state['active_levels']:
                continue
                
            # 分配资源
            level_resources = self._allocate_level_resources(level_name)
            
            # 处理数据 - 确保输入是3维的
            level_input = current_input.unsqueeze(1) if current_input.dim() == 2 else current_input
            output, confidence = level.process(
                level_input,
                {**context, 'resources': level_resources}
            )
            
            # 记录结果
            results[level_name] = {
                'output': output,
                'confidence': confidence,
                'state': level.state
            }
            
            # 更新输入 - 确保是2维的用于下一层处理
            current_input = output.squeeze(1) if output.dim() == 3 else output
            
            # 收集反馈
            self._collect_level_feedback(level_name, results[level_name])
        
        return results
    
    def _top_down_control(self,
                         bottom_up_results: Dict[str, Any],
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """自上而下的控制流程
        
        基于高层结果对低层进行调控
        """
        control_signals = {}
        
        # 从高层到低层进行控制
        for level_name in reversed(self.levels.keys()):
            if level_name not in self.system_state['active_levels']:
                continue
                
            # 获取当前层的结果
            level_result = bottom_up_results.get(level_name, {})
            
            # 生成控制信号
            control = self._generate_control_signals(
                level_name,
                level_result,
                context
            )
            
            # 应用控制
            self._apply_control_signals(level_name, control)
            
            control_signals[level_name] = control
        
        return control_signals
    
    def _integrate_results(self,
                         bottom_up_results: Dict[str, Any],
                         top_down_control: Dict[str, Any]) -> torch.Tensor:
        """整合各层结果
        
        使用加权整合方法组合各层输出，确保维度一致性
        """
        weighted_outputs = []
        weights = []
        target_size = None
        
        # 1. 首先确定目标维度
        for level_name, result in bottom_up_results.items():
            if level_name not in self.system_state['active_levels']:
                continue
            
            output = result['output']
            if target_size is None:
                target_size = output.size()
            elif output.size() != target_size:
                # 调整输出维度以匹配目标维度
                if len(output.size()) < len(target_size):
                    # 如果维度较少，添加维度
                    output = output.unsqueeze(1).expand(-1, target_size[1], -1)
                elif len(output.size()) > len(target_size):
                    # 如果维度较多，取平均降维
                    output = output.mean(dim=1)
                
                # 确保最后一个维度匹配
                if output.size(-1) != target_size[-1]:
                    # 使用线性投影调整特征维度
                    projection = nn.Linear(
                        output.size(-1), 
                        target_size[-1],
                        device=output.device
                    )
                    output = projection(output)
            
            # 获取权重
            confidence = result['confidence']
            control = top_down_control.get(level_name, {})
            weight = confidence * control.get('weight', 1.0)
            
            weighted_outputs.append(output * weight)
            weights.append(weight)
        
        # 2. 整合结果
        total_weight = sum(weights) + 1e-10  # 避免除零
        integrated_output = sum(weighted_outputs) / total_weight
        
        return integrated_output
    
    def update(self, feedback: Dict[str, float]) -> None:
        """更新系统状态和参数
        
        基于外部反馈调整系统行为
        """
        # 1. 收集系统级反馈
        system_feedback = self.feedback_system.collect_feedback(
            performance=feedback.get('performance', 0.5),
            resource_usage=self.system_state['resource_distribution']
        )
        
        # 2. 分析反馈
        analysis = self.feedback_system.analyze_feedback()
        
        # 3. 调整系统参数
        self._adjust_system_parameters(analysis)
        
        # 4. 更新各层
        self._update_levels(analysis)
    
    def _evaluate_task_complexity(self, input_data: torch.Tensor) -> float:
        """评估任务复杂度"""
        # 基于输入特征的统计特性评估复杂度
        feature_complexity = torch.std(input_data).item()
        
        # 考虑输入维度
        dimensional_complexity = np.log(input_data.numel()) / 10
        
        return min(1.0, feature_complexity * 0.7 + dimensional_complexity * 0.3)
    
    def _activate_relevant_levels(self, task_complexity: float) -> None:
        """激活相关的处理层次"""
        # 清除当前激活状态
        self.system_state['active_levels'].clear()
        
        # 基于任务复杂度激活层次
        if task_complexity > self.coordination_params['activation_threshold']:
            self.system_state['active_levels'].add('fast_intuition')
            
        if task_complexity > self.coordination_params['activation_threshold'] * 2:
            self.system_state['active_levels'].add('analytical')
            
        if task_complexity > self.coordination_params['activation_threshold'] * 3:
            self.system_state['active_levels'].add('abstract_reasoning')
            
        # 元认知层始终保持激活
        self.system_state['active_levels'].add('metacognitive')
    
    def _allocate_level_resources(self, level_name: str) -> Dict[str, float]:
        """为特定层次分配资源"""
        base_allocation = {
            'fast_intuition': 0.2,
            'analytical': 0.3,
            'abstract_reasoning': 0.4,
            'metacognitive': 0.1
        }
        
        # 获取基础分配比例
        base_ratio = base_allocation[level_name]
        
        # 考虑当前系统状态调整分配
        adjusted_ratio = base_ratio * (1 + self.system_state['processing_depth'])
        
        # 分配资源
        return {
            resource: amount * adjusted_ratio
            for resource, amount in self.system_state['resource_distribution'].items()
        }
    
    def _collect_level_feedback(self,
                              level_name: str,
                              level_result: Dict[str, Any]) -> None:
        """收集层次级别的反馈"""
        # 添加到通信通道
        self.communication_channels['bottom_up'].append({
            'level': level_name,
            'confidence': level_result['confidence'],
            'state': level_result['state']
        })
        
        # 维护通道长度
        if len(self.communication_channels['bottom_up']) > 50:
            self.communication_channels['bottom_up'] = self.communication_channels['bottom_up'][-50:]
    
    def _generate_control_signals(self,
                                level_name: str,
                                level_result: Dict[str, Any],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """生成控制信号"""
        control = {
            'weight': 1.0,
            'inhibition': 0.0,
            'modulation': {}
        }
        
        # 基于置信度调整权重
        if 'confidence' in level_result:
            control['weight'] *= level_result['confidence']
        
        # 基于性能调整抑制
        if level_name in self.system_state['performance_metrics']:
            metrics = self.system_state['performance_metrics'][level_name]
            if metrics:  # 确保有性能记录
                latest_metric = metrics[-1]  # 获取最新的性能指标
                if latest_metric['confidence'] < 0.3:  # 使用置信度作为性能指标
                    control['inhibition'] = self.coordination_params['inhibition_strength']
        
        # 添加调制信号
        control['modulation'] = {
            'learning_rate': context.get('learning_rate', 0.1),
            'attention_focus': context.get('attention_focus', None)
        }
        
        return control
    
    def _apply_control_signals(self,
                             level_name: str,
                             control: Dict[str, Any]) -> None:
        """应用控制信号"""
        # 添加到通信通道
        self.communication_channels['top_down'].append({
            'level': level_name,
            'control': control,
            'timestamp': time.time()
        })
        
        # 维护通道长度
        if len(self.communication_channels['top_down']) > 50:
            self.communication_channels['top_down'] = self.communication_channels['top_down'][-50:]
    
    def _adjust_system_parameters(self, analysis: Dict[str, Any]) -> None:
        """调整系统参数"""
        # 处理警告信号
        for alert in analysis.get('alert_signals', []):
            if alert['type'] == 'performance_drop':
                # 增加反馈强度
                self.coordination_params['feedback_strength'] *= 1.2
            elif alert['type'] == 'resource_depletion':
                # 提高激活阈值
                self.coordination_params['activation_threshold'] *= 1.1
        
        # 应用调整建议
        for suggestion in analysis.get('adjustment_suggestions', {}).values():
            if suggestion['action'] == 'increase_resources':
                self.coordination_params['integration_rate'] *= 1.1
            elif suggestion['action'] == 'optimize_processing':
                self.coordination_params['inhibition_strength'] *= 0.9
    
    def _update_levels(self, analysis: Dict[str, Any]) -> None:
        """更新各个层次"""
        for level_name, level in self.levels.items():
            if level_name not in self.system_state['active_levels']:
                continue
                
            # 获取层级性能
            level_performance = self.system_state['performance_metrics'].get(
                level_name,
                0.5
            )
            
            # 更新层级
            level.update(level_performance)

    def _update_system_state(self, 
                           final_result: torch.Tensor,
                           level_results: Dict[str, Any]) -> None:
        """更新系统状态
        
        Args:
            final_result: 最终处理结果
            level_results: 各层处理结果
        """
        # 更新性能指标
        for level_name, result in level_results.items():
            if level_name not in self.system_state['performance_metrics']:
                self.system_state['performance_metrics'][level_name] = []
                
            self.system_state['performance_metrics'][level_name].append({
                'confidence': result['confidence'],
                'output_norm': torch.norm(result['output']).item()
            })
            
            # 保持历史长度
            if len(self.system_state['performance_metrics'][level_name]) > 50:
                self.system_state['performance_metrics'][level_name] = \
                    self.system_state['performance_metrics'][level_name][-50:]
        
        # 更新资源分布
        self.system_state['resource_distribution'] = \
            self.resource_manager.monitor_usage()
            
        # 更新处理深度
        self.system_state['processing_depth'] = \
            len(self.system_state['active_levels']) / len(self.levels) 