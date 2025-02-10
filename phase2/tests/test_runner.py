"""测试运行模块

运行测试并收集结果。
"""

import torch
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import os
from datetime import datetime

from .data_generator import DataGenerator, TestCase
from .metrics import MetricsCalculator, TestResult
from ..models.enhanced_thinking.system import ThinkingSystem
from ..models.enhanced_thinking.config_manager import SystemConfig

@dataclass
class TestSuite:
    """测试套件"""
    name: str                 # 测试名称
    description: str          # 测试描述
    cases: List[TestCase]     # 测试用例
    config: SystemConfig      # 系统配置

class TestRunner:
    """测试运行器
    
    负责:
    1. 运行测试用例
    2. 收集测试结果
    3. 生成测试报告
    4. 保存测试数据
    """
    
    def __init__(self, 
                 output_dir: str = 'test_results',
                 save_results: bool = True):
        self.output_dir = output_dir
        self.save_results = save_results
        
        # 创建输出目录
        if save_results and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 初始化组件
        self.data_generator = DataGenerator()
        self.metrics_calculator = MetricsCalculator()
        
        # 测试历史
        self.test_history = []
        
    def run_suite(self, suite: TestSuite) -> Dict[str, Any]:
        """运行测试套件
        
        Args:
            suite: 测试套件
            
        Returns:
            测试结果
        """
        print(f"\nRunning test suite: {suite.name}")
        print(f"Description: {suite.description}")
        print(f"Number of test cases: {len(suite.cases)}")
        
        # 初始化系统
        system = ThinkingSystem(suite.config)
        
        # 运行测试用例
        results = []
        start_time = time.time()
        
        for i, case in enumerate(suite.cases):
            print(f"\nRunning test case {i+1}/{len(suite.cases)}")
            print(f"Description: {case.description}")
            
            # 运行测试用例
            result = self._run_case(system, case)
            results.append(result)
            
            # 打印结果
            print(result.description)
            
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 聚合结果
        aggregated_results = self.metrics_calculator.aggregate_results(results)
        aggregated_results['total_time'] = total_time
        aggregated_results['suite_name'] = suite.name
        aggregated_results['timestamp'] = datetime.now().isoformat()
        
        # 保存结果
        if self.save_results:
            self._save_results(aggregated_results, results)
            
        # 更新历史
        self.test_history.append(aggregated_results)
        
        return aggregated_results
        
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """运行所有测试
        
        Returns:
            所有测试结果
        """
        results = []
        
        # 1. 模式识别测试
        pattern_suite = TestSuite(
            name="Pattern Recognition Tests",
            description="Testing pattern recognition capabilities",
            cases=self.data_generator.generate_pattern_recognition_cases(),
            config=SystemConfig(input_size=256, hidden_size=512)
        )
        results.append(self.run_suite(pattern_suite))
        
        # 2. 序列分析测试
        sequence_suite = TestSuite(
            name="Sequence Analysis Tests",
            description="Testing sequence analysis capabilities",
            cases=self.data_generator.generate_sequence_analysis_cases(),
            config=SystemConfig(input_size=256, hidden_size=512)
        )
        results.append(self.run_suite(sequence_suite))
        
        # 3. 抽象推理测试
        reasoning_suite = TestSuite(
            name="Abstract Reasoning Tests",
            description="Testing abstract reasoning capabilities",
            cases=self.data_generator.generate_abstract_reasoning_cases(),
            config=SystemConfig(input_size=256, hidden_size=512)
        )
        results.append(self.run_suite(reasoning_suite))
        
        # 4. 多任务测试
        multi_task_suite = TestSuite(
            name="Multi-task Tests",
            description="Testing multi-task processing capabilities",
            cases=self.data_generator.generate_multi_task_cases(),
            config=SystemConfig(input_size=256, hidden_size=512)
        )
        results.append(self.run_suite(multi_task_suite))
        
        return results
        
    def _run_case(self, 
                  system: ThinkingSystem,
                  case: TestCase) -> TestResult:
        """运行单个测试用例
        
        Args:
            system: 思维系统
            case: 测试用例
            
        Returns:
            测试结果
        """
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 运行系统
            output, state = system.process(case.input_data, case.context)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 计算指标
            result = self.metrics_calculator.calculate_metrics(
                output=output,
                target=case.expected_output,
                system_state=state,
                processing_time=processing_time,
                case_complexity=case.complexity
            )
            
            return result
            
        except Exception as e:
            print(f"Error running test case: {str(e)}")
            # 返回失败结果
            return TestResult(
                accuracy=0.0,
                mse=float('inf'),
                response_time=float('inf'),
                resource_usage=1.0,
                complexity_score=0.0,
                level_stats={},
                description=f"Test failed with error: {str(e)}"
            )
            
    def _save_results(self,
                     aggregated_results: Dict[str, Any],
                     detailed_results: List[TestResult]) -> None:
        """保存测试结果
        
        Args:
            aggregated_results: 聚合结果
            detailed_results: 详细结果
        """
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建结果目录
        result_dir = os.path.join(
            self.output_dir,
            f"{aggregated_results['suite_name']}_{timestamp}"
        )
        os.makedirs(result_dir)
        
        # 保存聚合结果
        with open(os.path.join(result_dir, 'summary.json'), 'w') as f:
            json.dump(aggregated_results, f, indent=2)
            
        # 保存详细结果
        detailed_data = [
            {
                'accuracy': r.accuracy,
                'mse': r.mse,
                'response_time': r.response_time,
                'resource_usage': r.resource_usage,
                'complexity_score': r.complexity_score,
                'level_stats': r.level_stats,
                'description': r.description
            }
            for r in detailed_results
        ]
        
        with open(os.path.join(result_dir, 'details.json'), 'w') as f:
            json.dump(detailed_data, f, indent=2)
            
    def get_test_history(self) -> List[Dict[str, Any]]:
        """获取测试历史
        
        Returns:
            测试历史记录
        """
        return self.test_history
        
    def clear_history(self) -> None:
        """清除测试历史"""
        self.test_history.clear() 