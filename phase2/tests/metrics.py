"""测试指标计算模块

计算各种性能指标。
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

@dataclass
class TestResult:
    """测试结果"""
    accuracy: float            # 准确率
    mse: float                # 均方误差
    response_time: float      # 响应时间
    resource_usage: float     # 资源使用率
    complexity_score: float   # 复杂度得分
    level_stats: Dict[str, Dict[str, float]]  # 各层级统计
    description: str          # 结果描述

class MetricsCalculator:
    """指标计算器
    
    计算各种性能指标，包括:
    1. 准确率指标
    2. 效率指标
    3. 资源指标
    4. 复杂度指标
    5. 层级指标
    """
    
    def __init__(self):
        # 指标阈值
        self.thresholds = {
            'accuracy': 0.8,
            'mse': 0.2,
            'response_time': 1.0,  # 秒
            'resource_usage': 0.8
        }
        
        # 评分权重
        self.weights = {
            'accuracy': 0.4,
            'efficiency': 0.3,
            'resource': 0.2,
            'complexity': 0.1
        }
        
    def calculate_metrics(self,
                        output: torch.Tensor,
                        target: torch.Tensor,
                        system_state: Dict[str, Any],
                        processing_time: float,
                        case_complexity: float) -> TestResult:
        """计算测试指标
        
        Args:
            output: 系统输出
            target: 期望输出
            system_state: 系统状态
            processing_time: 处理时间
            case_complexity: 测试用例复杂度
            
        Returns:
            测试结果
        """
        # 1. 计算准确率指标
        accuracy = self._calculate_accuracy(output, target)
        mse = self._calculate_mse(output, target)
        
        # 2. 计算资源使用率
        resource_usage = self._calculate_resource_usage(system_state)
        
        # 3. 计算层级统计
        level_stats = self._calculate_level_stats(system_state)
        
        # 4. 计算复杂度得分
        complexity_score = self._calculate_complexity_score(
            case_complexity,
            accuracy,
            processing_time,
            resource_usage
        )
        
        # 5. 生成描述
        description = self._generate_description(
            accuracy,
            mse,
            processing_time,
            resource_usage,
            complexity_score,
            level_stats
        )
        
        return TestResult(
            accuracy=accuracy,
            mse=mse,
            response_time=processing_time,
            resource_usage=resource_usage,
            complexity_score=complexity_score,
            level_stats=level_stats,
            description=description
        )
        
    def aggregate_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """聚合多个测试结果
        
        Args:
            results: 测试结果列表
            
        Returns:
            聚合统计
        """
        if not results:
            return {}
            
        # 过滤掉失败的测试结果
        valid_results = [r for r in results if r.level_stats]
        if not valid_results:
            return {
                'average_metrics': {
                    'accuracy': 0.0,
                    'mse': float('inf'),
                    'response_time': float('inf'),
                    'resource_usage': 1.0,
                    'complexity_score': 0.0
                },
                'std_metrics': {
                    'accuracy': 0.0,
                    'mse': 0.0,
                    'response_time': 0.0,
                    'resource_usage': 0.0,
                    'complexity_score': 0.0
                },
                'level_statistics': {},
                'overall_score': 0.0,
                'performance_grade': 'D',
                'num_tests': len(results),
                'num_valid_tests': 0
            }
            
        # 1. 计算平均指标
        avg_metrics = {
            'accuracy': np.mean([r.accuracy for r in valid_results]),
            'mse': np.mean([r.mse for r in valid_results]),
            'response_time': np.mean([r.response_time for r in valid_results]),
            'resource_usage': np.mean([r.resource_usage for r in valid_results]),
            'complexity_score': np.mean([r.complexity_score for r in valid_results])
        }
        
        # 2. 计算标准差
        std_metrics = {
            'accuracy': np.std([r.accuracy for r in valid_results]),
            'mse': np.std([r.mse for r in valid_results]),
            'response_time': np.std([r.response_time for r in valid_results]),
            'resource_usage': np.std([r.resource_usage for r in valid_results]),
            'complexity_score': np.std([r.complexity_score for r in valid_results])
        }
        
        # 3. 计算层级统计
        level_stats = {}
        # 找出所有层级
        all_levels = set()
        for r in valid_results:
            all_levels.update(r.level_stats.keys())
            
        # 对每个层级计算统计
        for level in all_levels:
            # 找出包含该层级的结果
            level_results = [r for r in valid_results if level in r.level_stats]
            if not level_results:
                continue
                
            # 找出所有指标
            all_metrics = set()
            for r in level_results:
                all_metrics.update(r.level_stats[level].keys())
                
            # 计算每个指标的平均值
            level_stats[level] = {}
            for metric in all_metrics:
                values = []
                for r in level_results:
                    if metric in r.level_stats[level]:
                        values.append(r.level_stats[level][metric])
                if values:
                    level_stats[level][metric] = np.mean(values)
                else:
                    level_stats[level][metric] = 0.0
            
        # 4. 计算总体得分
        overall_score = self._calculate_overall_score(avg_metrics)
        
        # 5. 生成性能等级
        performance_grade = self._assign_performance_grade(overall_score)
        
        return {
            'average_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'level_statistics': level_stats,
            'overall_score': overall_score,
            'performance_grade': performance_grade,
            'num_tests': len(results),
            'num_valid_tests': len(valid_results)
        }
        
    def _calculate_accuracy(self,
                          output: torch.Tensor,
                          target: torch.Tensor) -> float:
        """计算准确率
        
        Args:
            output: 系统输出
            target: 期望输出
            
        Returns:
            准确率得分
        """
        # 计算相对误差
        relative_error = torch.abs(output - target) / (torch.abs(target) + 1e-10)
        accuracy = 1.0 - torch.mean(relative_error).item()
        
        return max(0.0, min(1.0, accuracy))
        
    def _calculate_mse(self,
                      output: torch.Tensor,
                      target: torch.Tensor) -> float:
        """计算均方误差
        
        Args:
            output: 系统输出
            target: 期望输出
            
        Returns:
            均方误差
        """
        return torch.mean((output - target) ** 2).item()
        
    def _calculate_resource_usage(self, system_state: Dict[str, Any]) -> float:
        """计算资源使用率
        
        Args:
            system_state: 系统状态
            
        Returns:
            资源使用率
        """
        if 'resource_distribution' not in system_state:
            return 0.0
            
        # 计算总资源使用率
        total_usage = sum(system_state['resource_distribution'].values())
        
        return min(1.0, total_usage)
        
    def _calculate_level_stats(self,
                             system_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """计算层级统计
        
        Args:
            system_state: 系统状态
            
        Returns:
            层级统计信息
        """
        stats = {}
        
        if 'performance_metrics' not in system_state:
            return stats
            
        # 计算每个层级的统计信息
        for level, metrics in system_state['performance_metrics'].items():
            stats[level] = {
                'confidence': metrics.get('confidence', 0.0),
                'output_norm': metrics.get('output_norm', 0.0),
                'resource_ratio': system_state['resource_distribution'].get(level, 0.0)
            }
            
        return stats
        
    def _calculate_complexity_score(self,
                                  case_complexity: float,
                                  accuracy: float,
                                  processing_time: float,
                                  resource_usage: float) -> float:
        """计算复杂度得分
        
        Args:
            case_complexity: 测试用例复杂度
            accuracy: 准确率
            processing_time: 处理时间
            resource_usage: 资源使用率
            
        Returns:
            复杂度得分
        """
        # 计算处理效率
        efficiency = 1.0 / (1.0 + processing_time)
        
        # 计算资源效率
        resource_efficiency = accuracy / (resource_usage + 1e-10)
        
        # 综合评分
        score = (
            accuracy * 0.4 +
            efficiency * 0.3 +
            resource_efficiency * 0.3
        )
        
        # 根据任务复杂度调整
        adjusted_score = score * (1.0 + case_complexity) / 2
        
        return max(0.0, min(1.0, adjusted_score))
        
    def _calculate_overall_score(self, avg_metrics: Dict[str, float]) -> float:
        """计算总体得分
        
        Args:
            avg_metrics: 平均指标
            
        Returns:
            总体得分
        """
        # 计算准确性得分
        accuracy_score = avg_metrics['accuracy']
        
        # 计算效率得分
        efficiency_score = 1.0 / (1.0 + avg_metrics['response_time'])
        
        # 计算资源得分
        resource_score = 1.0 - avg_metrics['resource_usage']
        
        # 计算复杂度得分
        complexity_score = avg_metrics['complexity_score']
        
        # 加权平均
        overall_score = (
            accuracy_score * self.weights['accuracy'] +
            efficiency_score * self.weights['efficiency'] +
            resource_score * self.weights['resource'] +
            complexity_score * self.weights['complexity']
        )
        
        return max(0.0, min(1.0, overall_score))
        
    def _assign_performance_grade(self, overall_score: float) -> str:
        """分配性能等级
        
        Args:
            overall_score: 总体得分
            
        Returns:
            性能等级
        """
        if overall_score >= 0.9:
            return 'S'
        elif overall_score >= 0.8:
            return 'A'
        elif overall_score >= 0.7:
            return 'B'
        elif overall_score >= 0.6:
            return 'C'
        else:
            return 'D'
            
    def _generate_description(self,
                            accuracy: float,
                            mse: float,
                            processing_time: float,
                            resource_usage: float,
                            complexity_score: float,
                            level_stats: Dict[str, Dict[str, float]]) -> str:
        """生成结果描述
        
        Args:
            accuracy: 准确率
            mse: 均方误差
            processing_time: 处理时间
            resource_usage: 资源使用率
            complexity_score: 复杂度得分
            level_stats: 层级统计
            
        Returns:
            结果描述
        """
        # 评估各项指标
        accuracy_status = 'good' if accuracy >= self.thresholds['accuracy'] else 'poor'
        mse_status = 'good' if mse <= self.thresholds['mse'] else 'poor'
        time_status = 'good' if processing_time <= self.thresholds['response_time'] else 'poor'
        resource_status = 'good' if resource_usage <= self.thresholds['resource_usage'] else 'poor'
        
        # 生成描述
        description = f"Test completed with {accuracy_status} accuracy ({accuracy:.3f}), "
        description += f"{mse_status} MSE ({mse:.3f}), "
        description += f"{time_status} response time ({processing_time:.3f}s), "
        description += f"{resource_status} resource usage ({resource_usage:.3f}). "
        description += f"Complexity score: {complexity_score:.3f}."
        
        # 添加层级信息
        if level_stats:
            description += "\nLevel performance:"
            for level, stats in level_stats.items():
                description += f"\n- {level}: "
                description += f"confidence={stats['confidence']:.3f}, "
                description += f"resource_ratio={stats['resource_ratio']:.3f}"
                
        return description 