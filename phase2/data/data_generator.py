"""实验数据生成模块

负责生成不同类型的测试数据，包括：
1. 基础模式识别数据
2. 分析推理数据
3. 抽象概念数据
4. 元认知控制数据
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

@dataclass
class DataConfig:
    """数据配置"""
    batch_size: int
    input_size: int
    num_patterns: int = 10
    num_concepts: int = 5
    noise_level: float = 0.1
    sequence_length: int = 20

class DataGenerator:
    """数据生成器"""
    def __init__(self, config: DataConfig):
        self.config = config
        
        # 生成基础模式
        self.patterns = torch.randn(
            config.num_patterns,
            config.input_size
        )
        
        # 生成概念原型
        self.concepts = torch.randn(
            config.num_concepts,
            config.input_size
        )
        
        # 生成规则模板
        self.rules = self._generate_rules()
        
    def _generate_rules(self) -> List[Dict[str, Any]]:
        """生成规则模板"""
        rules = []
        
        # 1. 简单组合规则
        rules.append({
            'type': 'combination',
            'inputs': [0, 1],
            'weights': [0.7, 0.3]
        })
        
        # 2. 序列模式规则
        rules.append({
            'type': 'sequence',
            'pattern': [0, 1, 2],
            'interval': 2
        })
        
        # 3. 层次关系规则
        rules.append({
            'type': 'hierarchy',
            'parent': 0,
            'children': [1, 2],
            'inheritance': 0.5
        })
        
        return rules
        
    def generate_pattern_recognition_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成模式识别数据
        
        生成包含基础模式的序列数据，适合FastIntuitionLevel测试
        
        维度设计:
        1. 输入序列: [batch_size, sequence_length, input_size]
        2. 目标输出: [batch_size, input_size]
        
        Returns:
            输入数据和目标输出
        """
        batch_inputs = []
        batch_targets = []
        
        for _ in range(self.config.batch_size):
            # 生成序列
            sequence = []
            # 随机选择一个主要模式
            main_pattern_idx = torch.randint(0, self.config.num_patterns, (1,)).item()
            
            # 生成序列中的每个时间步
            for _ in range(self.config.sequence_length):
                # 有80%的概率使用主要模式，20%的概率使用随机模式
                if torch.rand(1).item() < 0.8:
                    pattern_idx = main_pattern_idx
                else:
                    pattern_idx = torch.randint(0, self.config.num_patterns, (1,)).item()
                
                # 获取模式并添加噪声
                pattern = self.patterns[pattern_idx].clone()
                noise = torch.randn_like(pattern) * self.config.noise_level
                sequence.append(pattern + noise)
            
            # 组合序列
            input_sequence = torch.stack(sequence)  # [sequence_length, input_size]
            batch_inputs.append(input_sequence)
            
            # 目标是主要模式的无噪声版本
            batch_targets.append(self.patterns[main_pattern_idx])
        
        # 最终维度:
        # inputs: [batch_size, sequence_length, input_size]
        # targets: [batch_size, input_size]
        return torch.stack(batch_inputs), torch.stack(batch_targets)
        
    def generate_analytical_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成分析数据
        
        生成需要注意力机制的数据，适合AnalyticalLevel测试
        
        维度设计:
        1. 输入序列: [batch_size, sequence_length, input_size]
        2. 目标输出: [batch_size, input_size]
        
        Returns:
            输入数据和目标输出
        """
        # 生成序列数据
        sequence_length = 5
        batch_inputs = []
        batch_targets = []
        
        for _ in range(self.config.batch_size):
            # 随机选择规则
            rule = self.rules[1]  # 使用序列模式规则
            
            # 生成序列
            sequence = []
            for i in range(sequence_length):
                pattern_idx = rule['pattern'][i % len(rule['pattern'])]
                pattern = self.patterns[pattern_idx]
                
                # 添加位置编码
                position = torch.zeros_like(pattern)
                position[i * (len(position) // sequence_length): (i + 1) * (len(position) // sequence_length)] = 1
                
                sequence.append(pattern + position)
                
            # 组合序列
            input_sequence = torch.stack(sequence)  # [sequence_length, input_size]
            
            # 预测下一个模式
            next_pattern_idx = rule['pattern'][(sequence_length) % len(rule['pattern'])]
            target = self.patterns[next_pattern_idx]  # [input_size]
            
            batch_inputs.append(input_sequence)
            batch_targets.append(target)
            
        # 最终维度:
        # inputs: [batch_size, sequence_length, input_size]
        # targets: [batch_size, input_size]
        return torch.stack(batch_inputs), torch.stack(batch_targets)
        
    def generate_abstract_reasoning_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成抽象推理数据
        
        生成需要概念推理的数据，适合AbstractReasoningLevel测试
        
        Returns:
            输入数据和目标输出
        """
        batch_inputs = []
        batch_targets = []
        
        for _ in range(self.config.batch_size):
            # 随机选择规则
            rule = self.rules[2]  # 使用层次关系规则
            
            # 获取概念
            parent_concept = self.concepts[rule['parent']]
            child_concepts = self.concepts[rule['children']]
            
            # 生成混合概念
            input_concept = (
                parent_concept * rule['inheritance'] +
                torch.mean(child_concepts, dim=0) * (1 - rule['inheritance'])
            )
            
            # 添加噪声
            noise = torch.randn_like(input_concept) * self.config.noise_level
            input_concept += noise
            
            # 目标是纯净的混合概念
            target_concept = (
                parent_concept * rule['inheritance'] +
                torch.mean(child_concepts, dim=0) * (1 - rule['inheritance'])
            )
            
            batch_inputs.append(input_concept)
            batch_targets.append(target_concept)
            
        return torch.stack(batch_inputs), torch.stack(batch_targets)
        
    def generate_metacognitive_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成元认知数据
        
        生成需要控制决策的数据，适合MetaCognitiveLevel测试
        
        维度设计:
        - 输入: [batch_size, input_size]
        - 目标: [batch_size, input_size]
        
        Returns:
            输入数据和目标输出
        """
        batch_inputs = []
        batch_targets = []
        
        for _ in range(self.config.batch_size):
            # 随机初始状态
            state = {
                'performance': np.random.uniform(0.3, 0.7),
                'resource_usage': np.random.uniform(0.2, 0.8),
                'confidence': np.random.uniform(0.4, 0.6)
            }
            
            # 理想控制
            control = {
                'learning_rate': 0.1 if state['performance'] < 0.5 else 0.05,
                'resource_allocation': 0.8 if state['resource_usage'] > 0.7 else 0.5,
                'attention_focus': 1.0 if state['confidence'] < 0.5 else 0.5
            }
            
            # 生成输入特征
            input_features = []
            # 系统状态特征
            input_features.extend([
                state['performance'],
                state['resource_usage'],
                state['confidence']
            ])
            # 填充到input_size
            input_features.extend([0.0] * (self.config.input_size - len(input_features)))
            
            # 生成目标特征
            target_features = []
            # 控制信号
            target_features.extend([
                control['learning_rate'],
                control['resource_allocation'],
                control['attention_focus']
            ])
            # 填充到input_size
            target_features.extend([0.0] * (self.config.input_size - len(target_features)))
            
            batch_inputs.append(torch.tensor(input_features))
            batch_targets.append(torch.tensor(target_features))
        
        return torch.stack(batch_inputs), torch.stack(batch_targets)
        
    def generate_test_batch(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """生成测试批次
        
        为每个层次生成适合的测试数据
        
        维度设计:
        1. FastIntuitionLevel: 
           - 输入: [batch_size, input_size]
           - 目标: [batch_size, input_size]
        2. AnalyticalLevel:
           - 输入: [batch_size, input_size]
           - 目标: [batch_size, input_size]
        3. AbstractReasoningLevel:
           - 输入: [batch_size, input_size]
           - 目标: [batch_size, input_size]
        4. MetaCognitiveLevel:
           - 输入: [batch_size, input_size]
           - 目标: [batch_size, input_size]
        
        Returns:
            各层次的测试数据
        """
        test_data = {}
        
        # 1. FastIntuitionLevel数据
        inputs, targets = self.generate_pattern_recognition_data()
        # 取序列的最后一个时间步
        test_data['fast_intuition'] = (
            inputs[:, -1, :],  # [batch_size, input_size]
            targets            # [batch_size, input_size]
        )
        
        # 2. AnalyticalLevel数据
        inputs, targets = self.generate_analytical_data()
        # 取序列的最后一个时间步
        test_data['analytical'] = (
            inputs[:, -1, :],  # [batch_size, input_size]
            targets            # [batch_size, input_size]
        )
        
        # 3. AbstractReasoningLevel数据
        inputs, targets = self.generate_abstract_reasoning_data()
        test_data['abstract_reasoning'] = (inputs, targets)  # 已经是正确的维度
        
        # 4. MetaCognitiveLevel数据
        inputs, targets = self.generate_metacognitive_data()
        test_data['metacognitive'] = (inputs, targets)  # 已经是正确的维度
        
        return test_data 