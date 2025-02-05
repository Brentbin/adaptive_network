"""实验运行脚本

运行系统的完整实验，包括：
1. 基础功能测试
2. 性能测试
3. 长期稳定性测试
4. 适应性测试
"""

import os
import torch
import yaml
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any

from models.enhanced_thinking import (
    ThinkingSystem,
    ConfigManager,
    SystemLogger,
    SystemMonitor
)
from data.data_generator import DataGenerator, DataConfig

class ExperimentEncoder(json.JSONEncoder):
    """自定义JSON编码器,处理特殊类型的序列化"""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return super().default(obj)

def setup_experiment(config_path: str) -> Dict[str, Any]:
    """设置实验环境
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        实验组件字典
    """
    # 加载配置
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', 'experiments', timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化组件
    logger = SystemLogger(log_dir)
    monitor = SystemMonitor(logger, os.path.join(log_dir, 'monitor'))
    system = ThinkingSystem(config)
    
    # 创建数据生成器
    data_generator = DataGenerator(DataConfig(
        batch_size=config.batch_size,
        input_size=config.input_size,
        num_patterns=10,
        num_concepts=5
    ))
    
    return {
        'config': config,
        'logger': logger,
        'monitor': monitor,
        'system': system,
        'data_generator': data_generator,
        'log_dir': log_dir
    }

def run_basic_test(components: Dict[str, Any]) -> Dict[str, Any]:
    """运行基础测试
    
    维度处理:
    1. 输入数据: [batch_size, input_size]
    2. 目标数据: [batch_size, input_size]
    3. 系统输出: [batch_size, input_size]
    
    Args:
        components: 实验组件
        
    Returns:
        测试结果
    """
    system = components['system']
    data_generator = components['data_generator']
    monitor = components['monitor']
    
    results = {}
    
    # 测试各个层次
    test_data = data_generator.generate_test_batch()
    
    # 层级名称映射
    level_name_map = {
        'fast_intuition': 'fast_intuition',
        'analytical': 'analytical',
        'abstract_reasoning': 'abstract_reasoning',
        'metacognitive': 'metacognitive'
    }
    
    for level_name, (inputs, targets) in test_data.items():
        # 确保输入是2维的 [batch_size, input_size]
        if inputs.dim() == 3:
            inputs = inputs.squeeze(1)
        if targets.dim() == 3:
            targets = targets.squeeze(1)
        
        # 处理数据
        output, state = system.process(inputs)
        
        # 确保输出是2维的 [batch_size, input_size]
        if output.dim() == 3:
            output = output.squeeze(1)
        
        # 计算性能指标
        mse = torch.mean((output - targets) ** 2).item()
        accuracy = torch.mean((torch.abs(output - targets) < 0.1).float()).item()
        
        results[level_name] = {
            'mse': mse,
            'accuracy': accuracy,
            'state': state
        }
        
        # 记录性能
        monitor.update_metrics({
            'performance': {
                f'{level_name_map[level_name]}_mse': mse,
                f'{level_name_map[level_name]}_accuracy': accuracy
            }
        })
        
    return results

def run_performance_test(components: Dict[str, Any]) -> Dict[str, Any]:
    """运行性能测试
    
    维度处理:
    1. 输入数据: [batch_size, input_size]
    2. 目标数据: [batch_size, input_size]
    3. 系统输出: [batch_size, input_size]
    
    Args:
        components: 实验组件
        
    Returns:
        测试结果
    """
    system = components['system']
    data_generator = components['data_generator']
    monitor = components['monitor']
    
    results = {}
    
    # 测试不同批量大小
    batch_sizes = [1, 8, 32, 128]
    for batch_size in batch_sizes:
        # 生成数据
        inputs, targets = data_generator.generate_pattern_recognition_data()
        # 确保输入是2维的 [batch_size, input_size]
        if inputs.dim() == 3:
            inputs = inputs[:, -1, :]  # 取最后一个时间步
        if targets.dim() == 3:
            targets = targets[:, -1, :]
        
        # 测试处理速度
        start_time = datetime.now()
        output, state = system.process(inputs)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        results[f'batch_{batch_size}'] = {
            'processing_time': processing_time,
            'samples_per_second': batch_size / processing_time
        }
        
        # 记录性能
        monitor.update_metrics({
            'performance': {
                f'batch_{batch_size}_time': processing_time,
                f'batch_{batch_size}_throughput': batch_size / processing_time
            }
        })
        
    return results

def run_stability_test(components: Dict[str, Any]) -> Dict[str, Any]:
    """运行长期稳定性测试
    
    Args:
        components: 实验组件
        
    Returns:
        测试结果
    """
    system = components['system']
    data_generator = components['data_generator']
    monitor = components['monitor']
    
    results = {
        'performance_history': [],
        'resource_usage': [],
        'state_changes': []
    }
    
    # 长期运行
    num_iterations = 100
    for i in range(num_iterations):
        # 生成数据
        test_data = data_generator.generate_test_batch()
        
        # 处理每个层次的数据
        for level_name, (inputs, targets) in test_data.items():
            output, state = system.process(inputs)
            
            # 计算性能
            performance = {
                'mse': torch.mean((output - targets) ** 2).item(),
                'accuracy': torch.mean((torch.abs(output - targets) < 0.1).float()).item()
            }
            
            # 记录结果
            results['performance_history'].append(performance)
            results['resource_usage'].append(state['resource_distribution'])
            results['state_changes'].append(state['processing_depth'])
            
            # 更新监控
            monitor.update_metrics({
                'performance': performance,
                'resource_usage': state['resource_distribution'],
                'state_changes': {'depth': state['processing_depth']}
            })
            
    return results

def run_adaptation_test(components: Dict[str, Any]) -> Dict[str, Any]:
    """运行适应性测试
    
    维度处理:
    1. 输入数据: [batch_size, input_size]
    2. 目标数据: [batch_size, input_size]
    3. 系统输出: [batch_size, input_size]
    
    Args:
        components: 实验组件
        
    Returns:
        测试结果
    """
    system = components['system']
    data_generator = components['data_generator']
    monitor = components['monitor']
    
    results = {
        'adaptation_speed': [],
        'stability_recovery': []
    }
    
    # 测试不同扰动
    perturbations = [
        {'performance': 0.2},  # 性能下降
        {'resource_usage': 0.9},  # 资源压力
        {'confidence': 0.3}  # 置信度降低
    ]
    
    for perturbation in perturbations:
        # 记录初始状态
        initial_state = system.state_controller.working_point.copy()
        
        # 施加扰动
        system.update(perturbation)
        
        # 记录适应过程
        adaptation_history = []
        for _ in range(20):  # 观察20步
            # 生成数据
            inputs, targets = data_generator.generate_pattern_recognition_data()
            # 确保输入是2维的 [batch_size, input_size]
            if inputs.dim() == 3:
                inputs = inputs[:, -1, :]  # 取最后一个时间步
            if targets.dim() == 3:
                targets = targets[:, -1, :]
            
            # 处理数据
            output, state = system.process(inputs)
            
            # 计算性能
            performance = torch.mean((output - targets) ** 2).item()
            
            # 记录状态
            adaptation_history.append({
                'state': system.state_controller.working_point.copy(),
                'performance': performance
            })
            
            # 更新监控
            monitor.update_metrics({
                'adaptation': {
                    'state_distance': sum(
                        abs(v - initial_state[k])
                        for k, v in system.state_controller.working_point.items()
                    ),
                    'performance': performance
                }
            })
            
        results['adaptation_speed'].append(adaptation_history)
        
    return results

def save_results(results: Dict[str, Any], save_path: str):
    """保存实验结果
    
    Args:
        results: 实验结果
        save_path: 保存路径
    """
    # 处理结果
    processed_results = {}
    for test_name, test_results in results.items():
        processed_results[test_name] = {}
        for metric_name, metric_value in test_results.items():
            if isinstance(metric_value, (torch.Tensor, np.ndarray)):
                processed_results[test_name][metric_name] = metric_value.tolist()
            else:
                processed_results[test_name][metric_name] = metric_value
    
    # 保存为JSON
    with open(f'{save_path}.json', 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=4, ensure_ascii=False, cls=ExperimentEncoder)

def main():
    """运行实验"""
    # 设置实验
    components = setup_experiment('configs/test_config.yaml')
    
    # 运行测试
    results = {
        'basic_test': run_basic_test(components),
        'performance_test': run_performance_test(components),
        'stability_test': run_stability_test(components),
        'adaptation_test': run_adaptation_test(components)
    }
    
    # 保存结果
    save_results(results, os.path.join(components['log_dir'], 'results'))
    
    # 生成监控报告
    components['monitor'].generate_monitoring_report(
        os.path.join(components['log_dir'], 'monitoring_report.json')
    )
    
    print("实验完成！结果已保存到:", components['log_dir'])

if __name__ == '__main__':
    main() 