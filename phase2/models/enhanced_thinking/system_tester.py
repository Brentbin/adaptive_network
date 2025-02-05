"""测试框架模块

负责系统的测试和评估，包括单元测试、集成测试和性能测试。
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os

from .thinking_levels import ThinkingSystem
from .config_manager import SystemConfig
from .system_logger import SystemLogger

@dataclass
class TestCase:
    """测试用例"""
    input_data: torch.Tensor
    expected_output: torch.Tensor
    description: str
    level: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
@dataclass
class TestResult:
    """测试结果"""
    case: TestCase
    actual_output: torch.Tensor
    success: bool
    error: Optional[Exception] = None
    metrics: Optional[Dict[str, float]] = None
    duration: float = 0.0

class SystemTester:
    """系统测试器"""
    def __init__(self, 
                 system: ThinkingSystem,
                 logger: SystemLogger):
        self.system = system
        self.logger = logger
        self.test_cases: List[TestCase] = []
        self.test_results: List[TestResult] = []
        
    def add_test_case(self,
                      input_data: torch.Tensor,
                      expected_output: torch.Tensor,
                      description: str,
                      level: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> None:
        """添加测试用例
        
        Args:
            input_data: 输入数据
            expected_output: 期望输出
            description: 测试描述
            level: 指定测试层次（可选）
            context: 测试上下文（可选）
        """
        case = TestCase(
            input_data=input_data,
            expected_output=expected_output,
            description=description,
            level=level,
            context=context or {}
        )
        self.test_cases.append(case)
        
    def run_tests(self) -> Dict[str, Any]:
        """运行测试
        
        Returns:
            测试结果统计
        """
        self.test_results.clear()
        
        for case in self.test_cases:
            try:
                # 记录开始时间
                start_time = datetime.now()
                
                # 运行测试
                actual_output, state = self.system.process(
                    case.input_data,
                    case.context
                )
                
                # 计算持续时间
                duration = (datetime.now() - start_time).total_seconds()
                
                # 计算性能指标
                metrics = self._calculate_metrics(
                    case.expected_output,
                    actual_output
                )
                
                # 判断测试是否成功
                success = self._check_success(metrics)
                
                # 记录结果
                result = TestResult(
                    case=case,
                    actual_output=actual_output,
                    success=success,
                    metrics=metrics,
                    duration=duration
                )
                
            except Exception as e:
                # 记录错误
                result = TestResult(
                    case=case,
                    actual_output=None,
                    success=False,
                    error=e,
                    duration=(datetime.now() - start_time).total_seconds()
                )
                
                # 记录错误日志
                self.logger.log_error(
                    level=case.level or 'system',
                    error=e,
                    context={'test_case': case.description}
                )
                
            self.test_results.append(result)
            
            # 记录性能指标
            if result.metrics:
                self.logger.log_performance(
                    level=case.level or 'system',
                    metrics=result.metrics
                )
                
        return self.analyze_results()
        
    def _calculate_metrics(self,
                         expected: torch.Tensor,
                         actual: torch.Tensor) -> Dict[str, float]:
        """计算性能指标
        
        Args:
            expected: 期望输出
            actual: 实际输出
            
        Returns:
            性能指标字典
        """
        # 确保张量在CPU上
        expected = expected.detach().cpu()
        actual = actual.detach().cpu()
        
        # 计算各种指标
        mse = torch.mean((expected - actual) ** 2).item()
        mae = torch.mean(torch.abs(expected - actual)).item()
        
        # 计算相关系数
        if expected.numel() > 1:
            correlation = torch.corrcoef(
                torch.stack([expected.flatten(), actual.flatten()])
            )[0, 1].item()
        else:
            correlation = 1.0 if torch.allclose(expected, actual) else 0.0
            
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation
        }
        
    def _check_success(self, metrics: Dict[str, float]) -> bool:
        """检查测试是否成功
        
        Args:
            metrics: 性能指标
            
        Returns:
            是否成功
        """
        # 设定阈值
        thresholds = {
            'mse': 0.1,
            'mae': 0.1,
            'correlation': 0.9
        }
        
        # 检查是否满足所有阈值
        return all(
            metrics[metric] <= threshold if 'error' in metric.lower()
            else metrics[metric] >= threshold
            for metric, threshold in thresholds.items()
        )
        
    def analyze_results(self) -> Dict[str, Any]:
        """分析测试结果
        
        Returns:
            测试结果统计
        """
        if not self.test_results:
            return {}
            
        # 基础统计
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - successful_tests
        
        # 性能指标统计
        metrics_summary = {}
        for result in self.test_results:
            if result.metrics:
                for metric, value in result.metrics.items():
                    if metric not in metrics_summary:
                        metrics_summary[metric] = []
                    metrics_summary[metric].append(value)
                    
        # 计算平均指标
        avg_metrics = {}
        for metric, values in metrics_summary.items():
            avg_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            
        # 时间统计
        durations = [r.duration for r in self.test_results]
        time_stats = {
            'total_time': sum(durations),
            'avg_time': np.mean(durations),
            'max_time': np.max(durations),
            'min_time': np.min(durations)
        }
        
        # 错误统计
        errors = [r for r in self.test_results if r.error is not None]
        error_types = {}
        for error in errors:
            error_type = type(error.error).__name__
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
            
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests / total_tests,
            'metrics_summary': avg_metrics,
            'time_stats': time_stats,
            'error_summary': {
                'total_errors': len(errors),
                'error_types': error_types
            }
        }
        
    def generate_report(self, 
                       output_path: str) -> None:
        """生成测试报告
        
        Args:
            output_path: 报告输出路径
        """
        # 获取测试结果
        results = self.analyze_results()
        
        # 添加时间戳
        results['timestamp'] = datetime.now().isoformat()
        
        # 添加详细测试用例结果
        case_results = []
        for result in self.test_results:
            case_result = {
                'description': result.case.description,
                'level': result.case.level,
                'success': result.success,
                'duration': result.duration
            }
            
            if result.metrics:
                case_result['metrics'] = result.metrics
                
            if result.error:
                case_result['error'] = {
                    'type': type(result.error).__name__,
                    'message': str(result.error)
                }
                
            case_results.append(case_result)
            
        results['test_cases'] = case_results
        
        # 保存报告
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
    def clear_test_cases(self) -> None:
        """清除所有测试用例"""
        self.test_cases.clear()
        self.test_results.clear() 