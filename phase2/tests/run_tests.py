"""测试运行脚本

运行系统的基础测试和性能测试。
"""

import os
import unittest
import torch
from datetime import datetime

from models.enhanced_thinking import (
    ThinkingSystem,
    ConfigManager,
    SystemLogger,
    SystemMonitor
)

def run_basic_tests():
    """运行基础测试"""
    # 设置测试环境
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(test_dir)
    
    # 创建日志目录
    log_dir = os.path.join(project_dir, 'logs', 'test')
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化组件
    config_manager = ConfigManager(
        os.path.join(project_dir, 'configs', 'test_config.yaml')
    )
    logger = SystemLogger(log_dir)
    monitor = SystemMonitor(
        logger,
        os.path.join(log_dir, 'monitor')
    )
    
    # 创建系统
    system = ThinkingSystem(config_manager.get_config())
    
    # 运行测试
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(test_dir)
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 运行测试并生成报告
    runner = unittest.TextTestRunner(verbosity=2)
    test_results = runner.run(test_suite)
    
    # 记录结束时间
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 记录测试结果
    logger.log_performance(
        'system',
        {
            'tests_run': test_results.testsRun,
            'tests_failed': len(test_results.failures),
            'tests_errors': len(test_results.errors),
            'duration': duration
        }
    )
    
    # 生成监控报告
    monitor.generate_monitoring_report(
        os.path.join(log_dir, 'test_report.json')
    )
    
    return test_results

if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    
    # 运行测试
    results = run_basic_tests()
    
    # 输出总结
    print("\n测试总结:")
    print(f"运行测试: {results.testsRun}")
    print(f"失败: {len(results.failures)}")
    print(f"错误: {len(results.errors)}")
    
    # 设置退出码
    exit_code = 0 if results.wasSuccessful() else 1
    exit(exit_code) 