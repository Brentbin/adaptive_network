"""数据生成器"""

import numpy as np
from pathlib import Path
import json
import sys

root_path = "/Users/libin/SAGE/src/sage/models/adaptive_network"
sys.path.append(str(root_path))
from experiments.exp2.config import CONFIG

class SequenceTaskGenerator:
    def __init__(self):
        self.num_samples = CONFIG['num_samples']
        self.test_ratio = CONFIG['test_ratio']
        self.seq_length = CONFIG['seq_length']
        self.input_window = CONFIG['input_window']
        self.noise_level = CONFIG['noise_level']
        self.num_patterns = CONFIG['num_patterns']
    
    def generate_addition_sequence(self):
        """生成加法序列数据
        每个序列中的下一个数是前两个数的和加上一些噪声
        """
        sequences = []
        for _ in range(self.num_samples):
            # 生成初始值
            seq = [np.random.uniform(-5, 5)]
            seq.append(np.random.uniform(-5, 5))
            
            # 生成序列
            for i in range(2, self.seq_length):
                next_val = seq[i-1] + seq[i-2]
                noise = np.random.normal(0, self.noise_level * abs(next_val))
                seq.append(next_val + noise)
            
            sequences.append(seq)
        
        return np.array(sequences)
    
    def generate_multiplication_sequence(self):
        """生成乘法序列数据
        每个序列中的下一个数是前两个数的积加上一些噪声
        初始值范围避开0，使用[0.1, 0.9]或[-0.9, -0.1]
        """
        sequences = []
        for _ in range(self.num_samples):
            # 生成初始值，随机选择正负区间
            ranges = [(0.1, 0.9), (-0.9, -0.1)]
            range1 = ranges[np.random.randint(2)]
            range2 = ranges[np.random.randint(2)]
            seq = [np.random.uniform(*range1)]
            seq.append(np.random.uniform(*range2))
            
            # 生成序列
            for i in range(2, self.seq_length):
                next_val = seq[i-1] * seq[i-2]
                noise = np.random.normal(0, self.noise_level * abs(next_val))
                seq.append(next_val + noise)
            
            sequences.append(seq)
        
        return np.array(sequences)
    
    def generate_pattern_sequence(self):
        """生成模式序列数据
        包含三种不同的模式：正弦、二次和指数
        """
        patterns = [
            lambda x: np.sin(x),           # 正弦模式
            lambda x: x**2,                # 二次模式
            lambda x: np.exp(x/5)          # 指数模式
        ]
        
        sequences = []
        samples_per_pattern = self.num_samples // len(patterns)
        
        for pattern_func in patterns:
            for _ in range(samples_per_pattern):
                # 生成基础序列
                x = np.linspace(-5, 5, self.seq_length)
                seq = pattern_func(x)
                
                # 添加噪声
                noise = np.random.normal(0, self.noise_level * np.abs(seq))
                seq = seq + noise
                
                sequences.append(seq)
        
        return np.array(sequences)
    
    def prepare_xy(self, sequences):
        """准备输入输出对"""
        X, y = [], []
        for seq in sequences:
            for i in range(len(seq) - self.input_window):
                X.append(seq[i:i+self.input_window])
                y.append(seq[i+self.input_window])
        
        return np.array(X), np.array(y)
    
    def train_test_split(self, sequences):
        """划分训练测试集"""
        num_test = int(len(sequences) * self.test_ratio)
        indices = np.random.permutation(len(sequences))
        
        train_idx = indices[num_test:]
        test_idx = indices[:num_test]
        
        return sequences[train_idx], sequences[test_idx]
    
    def save_data(self, data_dict, filename):
        """保存数据"""
        save_path = CONFIG['data_dir'] / 'processed' / filename
        np.savez(save_path, **data_dict)
        print(f"数据已保存到: {save_path}")

def main():
    """主函数"""
    # 初始化生成器
    generator = SequenceTaskGenerator()
    
    # 生成三种任务的序列数据
    tasks_data = {
        'addition': generator.generate_addition_sequence(),
        'multiplication': generator.generate_multiplication_sequence(),
        'pattern': generator.generate_pattern_sequence()
    }
    
    # 处理每个任务的数据
    for task_name, sequences in tasks_data.items():
        # 划分训练测试集
        train_seqs, test_seqs = generator.train_test_split(sequences)
        
        # 准备输入输出对
        X_train, y_train = generator.prepare_xy(train_seqs)
        X_test, y_test = generator.prepare_xy(test_seqs)
        
        # 保存数据
        data_dict = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_seqs': train_seqs,
            'test_seqs': test_seqs
        }
        generator.save_data(data_dict, f'{task_name}_data.npz')
    
    # 保存数据参数
    params = {
        'num_samples': CONFIG['num_samples'],
        'test_ratio': CONFIG['test_ratio'],
        'seq_length': CONFIG['seq_length'],
        'input_window': CONFIG['input_window'],
        'noise_level': CONFIG['noise_level'],
        'num_patterns': CONFIG['num_patterns']
    }
    
    params_path = CONFIG['data_dir'] / 'processed' / 'params.json'
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    print("所有数据生成完成！")

if __name__ == "__main__":
    main() 