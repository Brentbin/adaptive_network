"""测试运行脚本

运行所有测试并显示结果。
"""

import os
import json
from typing import Dict, List, Any
import argparse
from datetime import datetime
import torch

from .test_runner import TestRunner
from ..models.enhanced_thinking.config_manager import SystemConfig

def setup_args() -> argparse.Namespace:
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='Run system tests')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_results',
        help='Directory to save test results'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to system configuration file'
    )
    
    parser.add_argument(
        '--num-cases',
        type=int,
        default=100,
        help='Number of test cases per suite'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()

def load_config(config_file: str) -> SystemConfig:
    """加载系统配置
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        系统配置
    """
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        print("Using default configuration")
        return SystemConfig(
            input_size=256,
            hidden_size=512
        )
        
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
        
    return SystemConfig.from_dict(config_dict)

def print_results(results: List[Dict[str, Any]]) -> None:
    """打印测试结果
    
    Args:
        results: 测试结果列表
    """
    print("\n=== Test Results ===")
    
    for suite_result in results:
        print(f"\nTest Suite: {suite_result['suite_name']}")
        print(f"Timestamp: {suite_result['timestamp']}")
        print(f"Total Time: {suite_result['total_time']:.2f}s")
        print(f"Number of Tests: {suite_result['num_tests']}")
        print(f"Overall Score: {suite_result['overall_score']:.3f}")
        print(f"Performance Grade: {suite_result['performance_grade']}")
        
        print("\nAverage Metrics:")
        for metric, value in suite_result['average_metrics'].items():
            print(f"- {metric}: {value:.3f}")
            
        print("\nStandard Deviations:")
        for metric, value in suite_result['std_metrics'].items():
            print(f"- {metric}: {value:.3f}")
            
        print("\nLevel Statistics:")
        for level, stats in suite_result['level_statistics'].items():
            print(f"\n{level}:")
            for metric, value in stats.items():
                print(f"- {metric}: {value:.3f}")
                
def save_summary(results: List[Dict[str, Any]], output_dir: str) -> None:
    """保存测试总结
    
    Args:
        results: 测试结果列表
        output_dir: 输出目录
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_suites': len(results),
        'total_cases': sum(r['num_tests'] for r in results),
        'total_time': sum(r['total_time'] for r in results),
        'suite_results': results
    }
    
    # 计算总体统计
    all_scores = [r['overall_score'] for r in results]
    summary['overall_stats'] = {
        'min_score': min(all_scores),
        'max_score': max(all_scores),
        'avg_score': sum(all_scores) / len(all_scores)
    }
    
    # 保存总结
    summary_file = os.path.join(output_dir, 'test_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
def main() -> None:
    """主函数"""
    # 解析参数
    args = setup_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        
    # 加载配置
    config = load_config(args.config_file) if args.config_file else None
    
    # 创建测试运行器
    runner = TestRunner(
        output_dir=args.output_dir,
        save_results=True
    )
    
    # 运行测试
    print("Starting tests...")
    results = runner.run_all_tests()
    
    # 打印结果
    print_results(results)
    
    # 保存总结
    save_summary(results, args.output_dir)
    print(f"\nTest summary saved to {args.output_dir}/test_summary.json")
    
if __name__ == '__main__':
    main() 