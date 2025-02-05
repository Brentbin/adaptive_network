"""监控和可视化模块

负责系统运行时的状态监控和可视化。
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

from .system_logger import SystemLogger

class SystemMonitor:
    """系统监控器"""
    def __init__(self, 
                 logger: SystemLogger,
                 save_dir: str):
        self.logger = logger
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 监控指标
        self.metrics_history = {
            'performance': [],
            'resource_usage': [],
            'state_changes': [],
            'error_rates': []
        }
        
    def update_metrics(self,
                      metrics: Dict[str, Any]) -> None:
        """更新监控指标
        
        Args:
            metrics: 新的指标数据
        """
        timestamp = datetime.now()
        
        for category, values in metrics.items():
            if category in self.metrics_history:
                self.metrics_history[category].append({
                    'timestamp': timestamp,
                    'values': values
                })
                
    def plot_performance_trends(self,
                              save_path: Optional[str] = None) -> None:
        """绘制性能趋势图
        
        Args:
            save_path: 图表保存路径（可选）
        """
        if not self.metrics_history['performance']:
            return
            
        # 提取数据
        timestamps = [
            record['timestamp'] 
            for record in self.metrics_history['performance']
        ]
        values = [
            record['values']
            for record in self.metrics_history['performance']
        ]
        
        # 创建图表
        plt.figure(figsize=(12, 12))
        
        # 获取所有指标
        all_metrics = set()
        for v in values:
            all_metrics.update(v.keys())
        
        # 按层级分组
        level_metrics = {
            'fast_intuition': [],
            'analytical': [],
            'abstract_reasoning': [],
            'metacognitive': []
        }
        
        for metric in all_metrics:
            for level in level_metrics:
                if metric.startswith(level):
                    level_metrics[level].append(metric)
                    
        # 为每个层级创建子图
        for i, (level, metrics) in enumerate(level_metrics.items(), 1):
            if not metrics:
                continue
                
            plt.subplot(len(level_metrics), 1, i)
            for metric in metrics:
                try:
                    metric_values = [v.get(metric, 0) for v in values]
                    plt.plot(
                        timestamps,
                        metric_values,
                        marker='o',
                        label=metric
                    )
                except KeyError:
                    continue
                    
            plt.title(f'{level} 性能趋势')
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            save_path = os.path.join(
                self.save_dir,
                f'performance_trends_{datetime.now():%Y%m%d_%H%M%S}.png'
            )
            plt.savefig(save_path)
            
        plt.close()
        
    def plot_resource_usage(self,
                           save_path: Optional[str] = None) -> None:
        """绘制资源使用图
        
        Args:
            save_path: 图表保存路径（可选）
        """
        if not self.metrics_history['resource_usage']:
            return
            
        # 提取数据
        timestamps = [
            record['timestamp']
            for record in self.metrics_history['resource_usage']
        ]
        values = [
            record['values']
            for record in self.metrics_history['resource_usage']
        ]
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 堆叠面积图
        resources = list(values[0].keys())
        data = np.array([[v[r] for v in values] for r in resources])
        
        plt.stackplot(
            timestamps,
            data,
            labels=resources,
            alpha=0.7
        )
        
        plt.title('资源使用情况')
        plt.xlabel('时间')
        plt.ylabel('使用率')
        plt.legend(loc='upper left')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            save_path = os.path.join(
                self.save_dir,
                f'resource_usage_{datetime.now():%Y%m%d_%H%M%S}.png'
            )
            plt.savefig(save_path)
            
        plt.close()
        
    def plot_state_transitions(self,
                             save_path: Optional[str] = None) -> None:
        """绘制状态转换图
        
        Args:
            save_path: 图表保存路径（可选）
        """
        if not self.metrics_history['state_changes']:
            return
            
        # 提取数据
        timestamps = [
            record['timestamp']
            for record in self.metrics_history['state_changes']
        ]
        values = [
            record['values']
            for record in self.metrics_history['state_changes']
        ]
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 为每个状态维度创建子图
        dimensions = list(values[0].keys())
        for i, dim in enumerate(dimensions, 1):
            plt.subplot(len(dimensions), 1, i)
            plt.plot(
                timestamps,
                [v[dim] for v in values],
                marker='o'
            )
            plt.title(f'{dim} 状态变化')
            plt.grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            save_path = os.path.join(
                self.save_dir,
                f'state_transitions_{datetime.now():%Y%m%d_%H%M%S}.png'
            )
            plt.savefig(save_path)
            
        plt.close()
        
    def plot_error_distribution(self,
                              save_path: Optional[str] = None) -> None:
        """绘制错误分布图
        
        Args:
            save_path: 图表保存路径（可选）
        """
        if not self.metrics_history['error_rates']:
            return
            
        # 提取数据
        values = [
            record['values']
            for record in self.metrics_history['error_rates']
        ]
        
        # 统计错误类型
        error_types = {}
        for v in values:
            for error_type, count in v.items():
                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += count
                
        # 创建饼图
        plt.figure(figsize=(10, 10))
        plt.pie(
            list(error_types.values()),
            labels=list(error_types.keys()),
            autopct='%1.1f%%',
            startangle=90
        )
        plt.title('错误类型分布')
        
        if save_path:
            plt.savefig(save_path)
        else:
            save_path = os.path.join(
                self.save_dir,
                f'error_distribution_{datetime.now():%Y%m%d_%H%M%S}.png'
            )
            plt.savefig(save_path)
            
        plt.close()
        
    def generate_monitoring_report(self,
                                 output_path: str) -> None:
        """生成监控报告
        
        Args:
            output_path: 报告保存路径
        """
        # 创建报告目录
        report_dir = os.path.dirname(output_path)
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成图表
        plots_dir = os.path.join(report_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        self.plot_performance_trends(
            os.path.join(plots_dir, 'performance_trends.png')
        )
        self.plot_resource_usage(
            os.path.join(plots_dir, 'resource_usage.png')
        )
        self.plot_state_transitions(
            os.path.join(plots_dir, 'state_transitions.png')
        )
        self.plot_error_distribution(
            os.path.join(plots_dir, 'error_distribution.png')
        )
        
        # 生成统计数据
        stats = {
            'timestamp': datetime.now().isoformat(),
            'performance_summary': self._calculate_performance_summary(),
            'resource_usage_summary': self._calculate_resource_summary(),
            'state_changes_summary': self._calculate_state_summary(),
            'error_summary': self._calculate_error_summary()
        }
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
            
    def _calculate_performance_summary(self) -> Dict[str, Any]:
        """计算性能统计信息"""
        if not self.metrics_history['performance']:
            return {}
            
        recent_records = self.metrics_history['performance'][-10:]
        
        summary = {}
        metrics = list(recent_records[0]['values'].keys())
        
        for metric in metrics:
            values = [r['values'][metric] for r in recent_records]
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'trend': sum(b - a for a, b in zip(values[:-1], values[1:]))
            }
            
        return summary
        
    def _calculate_resource_summary(self) -> Dict[str, Any]:
        """计算资源使用统计信息"""
        if not self.metrics_history['resource_usage']:
            return {}
            
        recent_records = self.metrics_history['resource_usage'][-10:]
        
        summary = {}
        resources = list(recent_records[0]['values'].keys())
        
        for resource in resources:
            values = [r['values'][resource] for r in recent_records]
            summary[resource] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            
        return summary
        
    def _calculate_state_summary(self) -> Dict[str, Any]:
        """计算状态变化统计信息"""
        if not self.metrics_history['state_changes']:
            return {}
            
        recent_records = self.metrics_history['state_changes'][-10:]
        
        summary = {}
        dimensions = list(recent_records[0]['values'].keys())
        
        for dim in dimensions:
            values = [r['values'][dim] for r in recent_records]
            summary[dim] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'stability': 1.0 - np.std(values),
                'trend': sum(b - a for a, b in zip(values[:-1], values[1:]))
            }
            
        return summary
        
    def _calculate_error_summary(self) -> Dict[str, Any]:
        """计算错误统计信息"""
        if not self.metrics_history['error_rates']:
            return {}
            
        recent_records = self.metrics_history['error_rates'][-10:]
        
        # 统计错误类型
        error_counts = {}
        for record in recent_records:
            for error_type, count in record['values'].items():
                if error_type not in error_counts:
                    error_counts[error_type] = 0
                error_counts[error_type] += count
                
        total_errors = sum(error_counts.values())
        
        return {
            'total_errors': total_errors,
            'error_types': error_counts,
            'error_rate': total_errors / len(recent_records)
        }
        
    def clear_history(self) -> None:
        """清除历史记录"""
        for category in self.metrics_history:
            self.metrics_history[category].clear() 