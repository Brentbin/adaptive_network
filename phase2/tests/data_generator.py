"""测试数据生成模块

生成不同场景的测试数据。
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import random

@dataclass
class TestCase:
    """测试用例"""
    input_data: torch.Tensor      # 输入数据
    expected_output: torch.Tensor # 期望输出
    context: Dict[str, Any]       # 上下文信息
    complexity: float             # 任务复杂度
    description: str              # 测试描述

class DataGenerator:
    """测试数据生成器
    
    生成不同类型的测试数据，包括:
    1. 基础模式识别
    2. 序列分析
    3. 抽象推理
    4. 多任务处理
    """
    
    def __init__(self, 
                 input_size: int = 256,
                 hidden_size: int = 512,
                 seed: Optional[int] = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
        # 生成基础模式
        self.base_patterns = self._generate_base_patterns()
        
    def generate_pattern_recognition_cases(self, 
                                        num_cases: int = 100) -> List[TestCase]:
        """生成模式识别测试用例
        
        Args:
            num_cases: 测试用例数量
            
        Returns:
            测试用例列表
        """
        cases = []
        for i in range(num_cases):
            # 随机选择基础模式
            pattern_idx = random.randint(0, len(self.base_patterns) - 1)
            base_pattern = self.base_patterns[pattern_idx]
            
            # 添加噪声
            noise_level = random.uniform(0.1, 0.3)
            input_data = base_pattern + torch.randn_like(base_pattern) * noise_level
            
            # 生成测试用例
            case = TestCase(
                input_data=input_data,
                expected_output=base_pattern,
                context={
                    'pattern_type': f'pattern_{pattern_idx}',
                    'noise_level': noise_level
                },
                complexity=0.3 + noise_level,
                description=f'Pattern recognition test with noise level {noise_level:.2f}'
            )
            cases.append(case)
            
        return cases
        
    def generate_sequence_analysis_cases(self,
                                      num_cases: int = 100) -> List[TestCase]:
        """生成序列分析测试用例
        
        Args:
            num_cases: 测试用例数量
            
        Returns:
            测试用例列表
        """
        cases = []
        for i in range(num_cases):
            # 生成序列长度
            seq_length = random.randint(5, 10)
            
            # 生成序列模式
            pattern_type = random.choice(['linear', 'exponential', 'periodic'])
            input_seq, output_seq = self._generate_sequence(
                seq_length,
                pattern_type
            )
            
            # 整理为所需形状
            input_data = self._reshape_sequence(input_seq)
            expected_output = self._reshape_sequence(output_seq)
            
            # 生成测试用例
            case = TestCase(
                input_data=input_data,
                expected_output=expected_output,
                context={
                    'sequence_type': pattern_type,
                    'sequence_length': seq_length
                },
                complexity=0.5 + seq_length / 20,
                description=f'Sequence analysis test with {pattern_type} pattern'
            )
            cases.append(case)
            
        return cases
        
    def generate_abstract_reasoning_cases(self,
                                       num_cases: int = 100) -> List[TestCase]:
        """生成抽象推理测试用例
        
        Args:
            num_cases: 测试用例数量
            
        Returns:
            测试用例列表
        """
        cases = []
        for i in range(num_cases):
            # 生成规则
            rule_type = random.choice(['transformation', 'analogy', 'composition'])
            input_data, expected_output = self._generate_reasoning_case(rule_type)
            
            # 生成测试用例
            case = TestCase(
                input_data=input_data,
                expected_output=expected_output,
                context={
                    'rule_type': rule_type,
                    'difficulty': random.uniform(0.6, 0.9)
                },
                complexity=0.7 + random.uniform(0.1, 0.2),
                description=f'Abstract reasoning test with {rule_type} rule'
            )
            cases.append(case)
            
        return cases
        
    def generate_multi_task_cases(self,
                                num_cases: int = 100) -> List[TestCase]:
        """生成多任务测试用例
        
        Args:
            num_cases: 测试用例数量
            
        Returns:
            测试用例列表
        """
        cases = []
        for i in range(num_cases):
            # 随机选择任务组合
            tasks = random.sample([
                'pattern',
                'sequence',
                'reasoning'
            ], random.randint(2, 3))
            
            # 生成组合输入
            input_parts = []
            output_parts = []
            for task in tasks:
                if task == 'pattern':
                    case = self.generate_pattern_recognition_cases(1)[0]
                elif task == 'sequence':
                    case = self.generate_sequence_analysis_cases(1)[0]
                else:
                    case = self.generate_abstract_reasoning_cases(1)[0]
                    
                input_parts.append(case.input_data)
                output_parts.append(case.expected_output)
                
            # 合并输入输出
            input_data = torch.cat(input_parts, dim=-1)
            expected_output = torch.cat(output_parts, dim=-1)
            
            # 生成测试用例
            case = TestCase(
                input_data=input_data,
                expected_output=expected_output,
                context={
                    'tasks': tasks,
                    'num_tasks': len(tasks)
                },
                complexity=0.8 + len(tasks) * 0.1,
                description=f'Multi-task test with {", ".join(tasks)}'
            )
            cases.append(case)
            
        return cases
        
    def _generate_base_patterns(self) -> List[torch.Tensor]:
        """生成基础模式
        
        Returns:
            基础模式列表,每个模式形状为[1, input_size]
        """
        patterns = []
        
        # 1. 简单形状
        for _ in range(5):
            pattern = torch.zeros(1, self.input_size)
            start = random.randint(0, self.input_size - 50)
            length = random.randint(10, 50)
            pattern[0, start:start+length] = 1.0
            patterns.append(pattern)
            
        # 2. 周期模式
        for _ in range(5):
            x = torch.linspace(0, 4*np.pi, self.input_size)
            freq = random.uniform(0.5, 2.0)
            phase = random.uniform(0, 2*np.pi)
            pattern = torch.sin(freq * x + phase).unsqueeze(0)
            patterns.append(pattern)
            
        # 3. 高斯模式
        for _ in range(5):
            x = torch.linspace(-3, 3, self.input_size)
            mu = random.uniform(-2, 2)
            sigma = random.uniform(0.5, 1.5)
            pattern = torch.exp(-(x - mu)**2 / (2*sigma**2)).unsqueeze(0)
            patterns.append(pattern)
            
        return patterns
        
    def _generate_sequence(self,
                         length: int,
                         pattern_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成序列数据
        
        Args:
            length: 序列长度
            pattern_type: 模式类型
            
        Returns:
            输入序列和输出序列,形状为[1, length]
        """
        if pattern_type == 'linear':
            # 生成线性序列
            slope = random.uniform(0.5, 2.0)
            intercept = random.uniform(-1.0, 1.0)
            x = torch.linspace(0, 1, length)
            y = slope * x + intercept
            
            # 添加噪声
            noise = torch.randn(1, length) * 0.1
            input_seq = y.unsqueeze(0) + noise
            output_seq = y.unsqueeze(0)
            
        elif pattern_type == 'exponential':
            # 生成指数序列
            base = random.uniform(1.1, 1.5)
            x = torch.linspace(0, 1, length)
            y = torch.pow(base, x)
            
            # 归一化
            y = (y - y.min()) / (y.max() - y.min())
            
            # 添加噪声
            noise = torch.randn(1, length) * 0.1
            input_seq = y.unsqueeze(0) + noise
            output_seq = y.unsqueeze(0)
            
        else:  # periodic
            # 生成周期序列
            freq = random.uniform(1.0, 3.0)
            phase = random.uniform(0, 2*np.pi)
            x = torch.linspace(0, 2*np.pi, length)
            y = torch.sin(freq * x + phase)
            
            # 添加噪声
            noise = torch.randn(1, length) * 0.1
            input_seq = y.unsqueeze(0) + noise
            output_seq = y.unsqueeze(0)
            
        return input_seq, output_seq
        
    def _reshape_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """重整序列形状
        
        Args:
            seq: 输入序列 [1, length]
            
        Returns:
            重整后的序列 [1, input_size]
        """
        # 填充到目标维度
        padded = torch.zeros(1, self.input_size)
        padded[0, :seq.size(1)] = seq
        return padded
        
    def _generate_reasoning_case(self,
                               rule_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成推理测试用例
        
        Args:
            rule_type: 规则类型
            
        Returns:
            输入数据和期望输出,形状为[1, input_size]
        """
        if rule_type == 'transformation':
            # 生成变换规则
            base = torch.randn(1, self.input_size)
            transform_type = random.choice(['shift', 'scale', 'reverse'])
            
            if transform_type == 'shift':
                shift = random.uniform(-1.0, 1.0)
                input_data = base
                expected_output = base + shift
            elif transform_type == 'scale':
                scale = random.uniform(0.5, 2.0)
                input_data = base
                expected_output = base * scale
            else:  # reverse
                input_data = base
                expected_output = torch.flip(base, [1])
                
        elif rule_type == 'analogy':
            # 生成类比规则
            a = torch.randn(1, self.input_size)
            b = torch.randn(1, self.input_size)
            relation = b - a
            
            c = torch.randn(1, self.input_size)
            d = c + relation
            
            input_data = torch.cat([a, b, c], dim=1)
            expected_output = d
            
        else:  # composition
            # 生成组合规则
            parts = [torch.randn(1, self.input_size // 3) for _ in range(3)]
            input_data = torch.cat(parts, dim=1)
            
            # 应用组合规则
            operation = random.choice(['sum', 'product', 'max'])
            if operation == 'sum':
                result = sum(parts)
            elif operation == 'product':
                result = parts[0] * parts[1] * parts[2]
            else:
                result = torch.max(torch.stack(parts, dim=2), dim=2)[0]
                
            expected_output = result
            
        return input_data, expected_output 