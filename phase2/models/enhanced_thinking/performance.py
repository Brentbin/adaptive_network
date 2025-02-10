"""性能监控模块

实现系统性能的监控、分析和优化建议生成。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import numpy as np
import torch
from collections import deque

@dataclass
class PerformanceMetrics:
    """性能指标"""
    accuracy: float           # 准确率
    efficiency: float        # 效率
    resource_usage: float    # 资源使用率
    response_time: float     # 响应时间
    stability: float         # 稳定性
    adaptability: float      # 适应性
    timestamp: float         # 时间戳

class PerformanceMonitor:
    """性能监控器
    
    负责:
    1. 性能指标收集
    2. 性能趋势分析
    3. 性能瓶颈识别
    4. 优化建议生成
    """
    
    def __init__(self, window_size: int = 50):
        # 性能历史
        self.performance_history = deque(maxlen=window_size)
        
        # 层级性能
        self.level_performance = {
            'fast_intuition': deque(maxlen=window_size),
            'analytical': deque(maxlen=window_size),
            'abstract_reasoning': deque(maxlen=window_size),
            'metacognitive': deque(maxlen=window_size)
        }
        
        # 性能阈值
        self.thresholds = {
            'accuracy': 0.8,
            'efficiency': 0.7,
            'resource_usage': 0.8,
            'response_time': 0.5,  # 秒
            'stability': 0.7,
            'adaptability': 0.6
        }
        
        # 分析窗口
        self.analysis_windows = {
            'short_term': 5,
            'medium_term': 20,
            'long_term': 50
        }
        
        # 性能警报历史
        self.alert_history = []
        
    def record_performance(self, 
                          metrics: PerformanceMetrics,
                          level: Optional[str] = None) -> None:
        """记录性能指标
        
        Args:
            metrics: 性能指标
            level: 可选的层级名称
        """
        # 记录整体性能
        self.performance_history.append(metrics)
        
        # 记录层级性能
        if level and level in self.level_performance:
            self.level_performance[level].append(metrics)
            
        # 检查是否需要发出警报
        self._check_alerts(metrics, level)
        
    def analyze_performance(self) -> Dict[str, Any]:
        """分析性能
        
        Returns:
            性能分析报告
        """
        if not self.performance_history:
            return {}
            
        analysis = {
            'trends': self._analyze_trends(),
            'bottlenecks': self._identify_bottlenecks(),
            'level_analysis': self._analyze_level_performance(),
            'suggestions': self._generate_suggestions()
        }
        
        return analysis
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计
        
        Returns:
            性能统计信息
        """
        if not self.performance_history:
            return {}
            
        recent_metrics = list(self.performance_history)[-self.analysis_windows['short_term']:]
        
        stats = {
            'accuracy': np.mean([m.accuracy for m in recent_metrics]),
            'efficiency': np.mean([m.efficiency for m in recent_metrics]),
            'resource_usage': np.mean([m.resource_usage for m in recent_metrics]),
            'response_time': np.mean([m.response_time for m in recent_metrics]),
            'stability': np.mean([m.stability for m in recent_metrics]),
            'adaptability': np.mean([m.adaptability for m in recent_metrics])
        }
        
        return stats
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """获取性能警报
        
        Returns:
            警报列表
        """
        return self.alert_history[-10:]  # 返回最近10条警报
    
    def _check_alerts(self, 
                     metrics: PerformanceMetrics,
                     level: Optional[str] = None) -> None:
        """检查是否需要发出警报
        
        Args:
            metrics: 性能指标
            level: 可选的层级名称
        """
        # 检查各项指标
        for metric_name, threshold in self.thresholds.items():
            metric_value = getattr(metrics, metric_name)
            
            if metric_value < threshold:
                alert = {
                    'timestamp': time.time(),
                    'metric': metric_name,
                    'value': metric_value,
                    'threshold': threshold,
                    'level': level,
                    'severity': (threshold - metric_value) / threshold
                }
                
                self.alert_history.append(alert)
                
        # 维护警报历史长度
        if len(self.alert_history) > 50:
            self.alert_history = self.alert_history[-50:]
            
    def _analyze_trends(self) -> Dict[str, Any]:
        """分析性能趋势
        
        Returns:
            趋势分析结果
        """
        trends = {}
        
        # 分析不同时间窗口的趋势
        for window_name, window_size in self.analysis_windows.items():
            window_metrics = list(self.performance_history)[-window_size:]
            if not window_metrics:
                continue
                
            window_trends = {}
            for metric_name in vars(window_metrics[0]).keys():
                if metric_name == 'timestamp':
                    continue
                    
                values = [getattr(m, metric_name) for m in window_metrics]
                
                # 计算趋势
                if len(values) >= 2:
                    trend = sum(b - a for a, b in zip(values[:-1], values[1:])) / (len(values) - 1)
                else:
                    trend = 0
                    
                # 计算波动性
                volatility = np.std(values) if len(values) > 1 else 0
                
                window_trends[metric_name] = {
                    'direction': trend,
                    'volatility': volatility,
                    'current': values[-1],
                    'average': np.mean(values)
                }
                
            trends[window_name] = window_trends
            
        return trends
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """识别性能瓶颈
        
        Returns:
            瓶颈列表
        """
        bottlenecks = []
        
        if not self.performance_history:
            return bottlenecks
            
        recent_metrics = list(self.performance_history)[-self.analysis_windows['short_term']:]
        
        # 检查各项指标
        for metric_name in vars(recent_metrics[0]).keys():
            if metric_name == 'timestamp':
                continue
                
            values = [getattr(m, metric_name) for m in recent_metrics]
            avg_value = np.mean(values)
            
            # 如果指标低于阈值,认为是瓶颈
            threshold = self.thresholds.get(metric_name, 0.7)
            if avg_value < threshold:
                bottleneck = {
                    'metric': metric_name,
                    'current_value': avg_value,
                    'threshold': threshold,
                    'severity': (threshold - avg_value) / threshold
                }
                bottlenecks.append(bottleneck)
                
        # 按严重程度排序
        bottlenecks.sort(key=lambda x: x['severity'], reverse=True)
        
        return bottlenecks
    
    def _analyze_level_performance(self) -> Dict[str, Any]:
        """分析各层级的性能
        
        Returns:
            层级性能分析结果
        """
        level_analysis = {}
        
        for level_name, history in self.level_performance.items():
            if not history:
                continue
                
            recent_metrics = list(history)[-self.analysis_windows['short_term']:]
            
            # 计算平均性能指标
            level_stats = {}
            for metric_name in vars(recent_metrics[0]).keys():
                if metric_name == 'timestamp':
                    continue
                    
                values = [getattr(m, metric_name) for m in recent_metrics]
                level_stats[metric_name] = {
                    'average': np.mean(values),
                    'std': np.std(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values)
                }
                
            # 评估层级状态
            overall_performance = np.mean([
                level_stats[metric]['average']
                for metric in level_stats
            ])
            
            status = 'good' if overall_performance >= 0.8 else \
                     'moderate' if overall_performance >= 0.6 else \
                     'poor'
                     
            level_analysis[level_name] = {
                'stats': level_stats,
                'overall_performance': overall_performance,
                'status': status
            }
            
        return level_analysis
    
    def _generate_suggestions(self) -> List[Dict[str, Any]]:
        """生成优化建议
        
        Returns:
            建议列表
        """
        suggestions = []
        
        # 基于瓶颈生成建议
        bottlenecks = self._identify_bottlenecks()
        for bottleneck in bottlenecks:
            suggestion = self._generate_bottleneck_suggestion(bottleneck)
            if suggestion:
                suggestions.append(suggestion)
                
        # 基于层级分析生成建议
        level_analysis = self._analyze_level_performance()
        for level_name, analysis in level_analysis.items():
            if analysis['status'] == 'poor':
                suggestion = self._generate_level_suggestion(level_name, analysis)
                if suggestion:
                    suggestions.append(suggestion)
                    
        # 基于趋势分析生成建议
        trends = self._analyze_trends()
        if 'short_term' in trends:
            for metric_name, trend_info in trends['short_term'].items():
                if trend_info['direction'] < 0:  # 下降趋势
                    suggestion = {
                        'target': metric_name,
                        'type': 'trend_optimization',
                        'priority': abs(trend_info['direction']),
                        'description': f'Optimize {metric_name} to reverse negative trend',
                        'actions': [
                            f'Review recent changes affecting {metric_name}',
                            f'Adjust parameters related to {metric_name}',
                            'Consider temporary resource reallocation'
                        ]
                    }
                    suggestions.append(suggestion)
                    
        # 按优先级排序
        suggestions.sort(key=lambda x: x['priority'], reverse=True)
        
        return suggestions
    
    def _generate_bottleneck_suggestion(self, 
                                      bottleneck: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """根据瓶颈生成建议
        
        Args:
            bottleneck: 瓶颈信息
            
        Returns:
            建议信息
        """
        metric = bottleneck['metric']
        severity = bottleneck['severity']
        
        suggestion = {
            'target': metric,
            'type': 'bottleneck_mitigation',
            'priority': severity,
            'description': f'Address performance bottleneck in {metric}',
            'actions': []
        }
        
        # 根据不同指标生成具体建议
        if metric == 'accuracy':
            suggestion['actions'] = [
                'Increase model capacity',
                'Review training data quality',
                'Adjust learning rate'
            ]
        elif metric == 'efficiency':
            suggestion['actions'] = [
                'Optimize resource allocation',
                'Review processing pipeline',
                'Consider batch processing'
            ]
        elif metric == 'resource_usage':
            suggestion['actions'] = [
                'Implement resource pooling',
                'Review resource allocation strategy',
                'Consider scaling resources'
            ]
        elif metric == 'response_time':
            suggestion['actions'] = [
                'Optimize critical path',
                'Implement caching',
                'Review parallel processing opportunities'
            ]
        elif metric == 'stability':
            suggestion['actions'] = [
                'Increase regularization',
                'Implement gradual updates',
                'Review state management'
            ]
        elif metric == 'adaptability':
            suggestion['actions'] = [
                'Increase exploration rate',
                'Review adaptation mechanism',
                'Adjust learning parameters'
            ]
            
        return suggestion
    
    def _generate_level_suggestion(self,
                                 level_name: str,
                                 analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """根据层级分析生成建议
        
        Args:
            level_name: 层级名称
            analysis: 层级分析结果
            
        Returns:
            建议信息
        """
        suggestion = {
            'target': level_name,
            'type': 'level_optimization',
            'priority': 1 - analysis['overall_performance'],
            'description': f'Optimize performance of {level_name} level',
            'actions': []
        }
        
        # 根据不同层级生成具体建议
        if level_name == 'fast_intuition':
            suggestion['actions'] = [
                'Optimize pattern recognition',
                'Review feature extraction',
                'Adjust response threshold'
            ]
        elif level_name == 'analytical':
            suggestion['actions'] = [
                'Review analysis depth',
                'Optimize attention mechanism',
                'Adjust processing granularity'
            ]
        elif level_name == 'abstract_reasoning':
            suggestion['actions'] = [
                'Review abstraction mechanism',
                'Optimize rule generation',
                'Adjust reasoning depth'
            ]
        elif level_name == 'metacognitive':
            suggestion['actions'] = [
                'Review control strategy',
                'Optimize resource allocation',
                'Adjust feedback mechanism'
            ]
            
        return suggestion 