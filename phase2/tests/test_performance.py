"""性能测试模块

测试系统在不同场景下的性能表现。
"""

import unittest
import torch
import time
import os
import yaml
import numpy as np
from typing import Dict, Any, List

from models.enhanced_thinking import (
    ThinkingSystem,
    SystemConfig,
    ConfigManager,
    SystemLogger,
    SystemTester,
    SystemMonitor
)

class TestSystemPerformance(unittest.TestCase):
    """性能测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试前准备"""
        # 加载测试配置
        config_path = os.path.join(
            os.path.dirname(__file__),
            '../configs/test_config.yaml'
        )
        with open(config_path, 'r', encoding='utf-8') as f:
            cls.test_config = yaml.safe_load(f)
            
        # 创建系统组件
        cls.config_manager = ConfigManager(config_path)
        cls.logger = SystemLogger(
            os.path.join(
                os.path.dirname(__file__),
                '../logs/performance'
            )
        )
        cls.monitor = SystemMonitor(
            cls.logger,
            os.path.join(
                os.path.dirname(__file__),
                '../logs/performance/monitor'
            )
        )
        cls.system = ThinkingSystem(cls.config_manager.config)
        cls.tester = SystemTester(cls.system, cls.logger)
        
    def setUp(self):
        """每个测试前准备"""
        self.logger.clear_history()
        self.monitor.clear_history()
        self.tester.clear_test_cases()
        
    def test_processing_speed(self):
        """测试处理速度"""
        # 准备测试数据
        batch_sizes = [1, 8, 32, 128]
        times = []
        
        for batch_size in batch_sizes:
            input_data = torch.randn(
                batch_size,
                self.test_config['input_size']
            )
            
            # 预热
            self.system.process(input_data)
            
            # 计时
            start_time = time.time()
            for _ in range(10):
                self.system.process(input_data)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            times.append(avg_time)
            
            # 记录性能指标
            self.monitor.update_metrics({
                'performance': {
                    'batch_size': batch_size,
                    'processing_time': avg_time
                }
            })
            
        # 验证性能
        self.assertLess(
            times[0],
            0.1,
            "单样本处理时间应小于0.1秒"
        )
        
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        process = psutil.Process()
        
        # 记录初始内存
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 处理大量数据
        batch_size = 256
        num_batches = 10
        for _ in range(num_batches):
            input_data = torch.randn(
                batch_size,
                self.test_config['input_size']
            )
            self.system.process(input_data)
            
        # 记录最终内存
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 记录内存使用
        self.monitor.update_metrics({
            'resource_usage': {
                'memory_increase': memory_increase,
                'final_memory': final_memory
            }
        })
        
        # 验证内存使用
        self.assertLess(
            memory_increase,
            1000,  # 1GB
            "内存增长应该在合理范围内"
        )
        
    def test_resource_efficiency(self):
        """测试资源使用效率"""
        # 准备测试数据
        input_data = torch.randn(
            self.test_config['batch_size'],
            self.test_config['input_size']
        )
        
        # 记录资源使用
        initial_resources = self.system.resource_manager.global_resources.copy()
        
        # 连续处理
        num_iterations = 50
        resource_usage = []
        
        for _ in range(num_iterations):
            self.system.process(input_data)
            current_resources = self.system.resource_manager.global_resources
            resource_usage.append({
                k: v / initial_resources[k]
                for k, v in current_resources.items()
            })
            
        # 计算资源使用效率
        efficiency = {}
        for resource in initial_resources:
            values = [usage[resource] for usage in resource_usage]
            efficiency[resource] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            
        # 记录效率指标
        self.monitor.update_metrics({
            'resource_usage': efficiency
        })
        
        # 验证资源效率
        for resource, stats in efficiency.items():
            self.assertLess(
                stats['std'],
                0.2,
                f"{resource}使用应该稳定"
            )
            
    def test_scalability(self):
        """测试可扩展性"""
        # 测试不同规模的输入
        input_sizes = [64, 128, 256, 512]
        processing_times = []
        
        for size in input_sizes:
            # 创建新的系统实例，以适应不同的输入大小
            config = self.config_manager.get_config()
            config.input_size = size
            config.hidden_size = size * 2
            system = ThinkingSystem(config)
            
            input_data = torch.randn(
                self.test_config['batch_size'],
                size
            )
            
            # 预热
            system.process(input_data)
            
            # 计时
            start_time = time.time()
            for _ in range(5):
                system.process(input_data)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 5
            processing_times.append(avg_time)
            
        # 计算扩展性指标
        scalability = np.polyfit(
            input_sizes,
            processing_times,
            1
        )[0]
        
        # 记录扩展性指标
        self.monitor.update_metrics({
            'performance': {
                'scalability': scalability,
                'input_sizes': input_sizes,
                'processing_times': processing_times
            }
        })
        
        # 验证扩展性
        self.assertLess(
            scalability,
            0.001,
            "处理时间增长应该次线性"
        )
        
    def test_stability(self):
        """测试稳定性"""
        # 准备测试数据
        input_data = torch.randn(
            self.test_config['batch_size'],
            self.test_config['input_size']
        )
        
        # 连续处理
        num_iterations = 100
        outputs = []
        states = []
        
        for _ in range(num_iterations):
            output, state = self.system.process(input_data)
            outputs.append(output.detach().mean().item())
            states.append(self.system.state_controller.working_point.copy())
            
        # 计算稳定性指标
        output_stability = np.std(outputs)
        state_stability = {
            k: np.std([s[k] for s in states])
            for k in states[0]
        }
        
        # 记录稳定性指标
        self.monitor.update_metrics({
            'stability': {
                'output_stability': output_stability,
                'state_stability': state_stability
            }
        })
        
        # 验证稳定性
        self.assertLess(
            output_stability,
            0.1,
            "输出应该稳定"
        )
        for k, v in state_stability.items():
            self.assertLess(
                v,
                0.2,
                f"{k}状态应该稳定"
            )
            
    def test_adaptation_speed(self):
        """测试适应速度"""
        # 准备测试数据
        input_data = torch.randn(
            self.test_config['batch_size'],
            self.test_config['input_size']
        )
        
        # 记录初始状态
        initial_state = self.system.state_controller.working_point.copy()
        
        # 施加扰动
        perturbation = {
            'performance': 0.2,
            'efficiency': 0.3,
            'plasticity': 0.4
        }
        
        # 记录适应过程
        num_iterations = 50
        state_changes = []
        
        for _ in range(num_iterations):
            self.system.update(perturbation)
            current_state = self.system.state_controller.working_point.copy()
            state_changes.append({
                k: abs(current_state[k] - initial_state[k])
                for k in initial_state
            })
            
        # 计算适应速度
        adaptation_speed = {}
        for k in initial_state:
            changes = [change[k] for change in state_changes]
            # 找到稳定点
            stable_idx = next(
                (i for i in range(1, len(changes))
                 if abs(changes[i] - changes[i-1]) < 0.01),
                len(changes)
            )
            adaptation_speed[k] = stable_idx
            
        # 记录适应指标
        self.monitor.update_metrics({
            'adaptation': {
                'speed': adaptation_speed,
                'state_changes': state_changes
            }
        })
        
        # 验证适应速度
        for k, speed in adaptation_speed.items():
            self.assertLess(
                speed,
                20,
                f"{k}应该快速适应"
            )
            
    def test_long_term_stability(self):
        """测试长期稳定性"""
        # 准备测试数据
        input_data = torch.randn(
            self.test_config['batch_size'],
            self.test_config['input_size']
        )
        
        # 长期运行
        num_iterations = 200
        performance_history = []
        resource_usage = []
        state_history = []
        
        for _ in range(num_iterations):
            # 处理数据
            output, _ = self.system.process(input_data)
            
            # 记录性能
            performance = torch.mean(output).item()
            performance_history.append(performance)
            
            # 记录资源使用
            resources = self.system.resource_manager.global_resources.copy()
            resource_usage.append(resources)
            
            # 记录状态
            state = self.system.state_controller.working_point.copy()
            state_history.append(state)
            
        # 计算长期稳定性指标
        stability_metrics = {
            'performance': {
                'mean': np.mean(performance_history),
                'std': np.std(performance_history)
            },
            'resources': {
                k: {
                    'mean': np.mean([r[k] for r in resource_usage]),
                    'std': np.std([r[k] for r in resource_usage])
                }
                for k in resource_usage[0]
            },
            'state': {
                k: {
                    'mean': np.mean([s[k] for s in state_history]),
                    'std': np.std([s[k] for s in state_history])
                }
                for k in state_history[0]
            }
        }
        
        # 记录稳定性指标
        self.monitor.update_metrics({
            'long_term_stability': stability_metrics
        })
        
        # 验证长期稳定性
        self.assertLess(
            stability_metrics['performance']['std'],
            0.1,
            "长期性能应该稳定"
        )
        
        for resource, stats in stability_metrics['resources'].items():
            self.assertLess(
                stats['std'],
                0.2,
                f"{resource}使用应该长期稳定"
            )
            
        for state_var, stats in stability_metrics['state'].items():
            self.assertLess(
                stats['std'],
                0.2,
                f"{state_var}状态应该长期稳定"
            )
            
if __name__ == '__main__':
    unittest.main() 