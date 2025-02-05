"""
数据生成器 - 生成实验所需的序列数据

TODO:
1. 实现三种不同规则的序列生成:
   - 算术序列 (2,4,6,8...)
   - 几何序列 (2,4,8,16...)
   - 倍数变化序列 (3,6,12,24...)
2. 添加数据集划分功能
3. 实现数据保存和加载
"""

import numpy as np
import os
from pathlib import Path
import json

class SequenceGenerator:
    """序列数据生成器"""
    def __init__(self, 
                 seq_length=10,           # 序列长度
                 input_window=5,          # 输入窗口大小
                 train_samples=1000,      # 训练样本数
                 test_samples=200,        # 测试样本数
                 noise_level=0.01):       # 噪声水平
        
        self.seq_length = seq_length
        self.input_window = input_window
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.noise_level = noise_level
        
        # 确保数据保存路径存在
        self.data_path = Path("../data")
        self.raw_path = self.data_path / "raw"
        self.processed_path = self.data_path / "processed"
        
        for path in [self.data_path, self.raw_path, self.processed_path]:
            path.mkdir(parents=True, exist_ok=True)

    def generate_arithmetic(self, start=2, step=2):
        """生成算术序列: an = a1 + (n-1)d"""
        sequences = []
        for _ in range(self.train_samples + self.test_samples):
            # 生成基础序列
            seq = np.arange(start, start + self.seq_length * step, step)
            # 添加噪声
            noise = np.random.normal(0, self.noise_level, seq.shape)
            seq = seq + noise
            sequences.append(seq)
        
        return np.array(sequences)

    def generate_geometric(self, start=2, ratio=2):
        """生成几何序列: an = a1 * r^(n-1)"""
        sequences = []
        for _ in range(self.train_samples + self.test_samples):
            # 生成基础序列
            seq = start * np.power(ratio, np.arange(self.seq_length))
            # 添加噪声
            noise = np.random.normal(0, self.noise_level * seq, seq.shape)
            seq = seq + noise
            sequences.append(seq)
        
        return np.array(sequences)

    def generate_multiplier(self, start=3, multiplier=2):
        """生成倍数变化序列"""
        sequences = []
        for _ in range(self.train_samples + self.test_samples):
            # 生成基础序列
            seq = start * np.power(multiplier, np.arange(self.seq_length))
            # 添加噪声
            noise = np.random.normal(0, self.noise_level * seq, seq.shape)
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
        """划分训练集和测试集"""
        train = sequences[:self.train_samples]
        test = sequences[self.train_samples:]
        return train, test

    def save_data(self, data_dict, filename):
        """保存数据"""
        save_path = self.processed_path / filename
        np.savez(save_path, **data_dict)
        
        # 保存数据生成参数
        params = {
            'seq_length': self.seq_length,
            'input_window': self.input_window,
            'train_samples': self.train_samples,
            'test_samples': self.test_samples,
            'noise_level': self.noise_level
        }
        with open(self.processed_path / 'params.json', 'w') as f:
            json.dump(params, f, indent=4)

def main():
    """主函数"""
    # 初始化生成器
    generator = SequenceGenerator()
    
    # 生成三种序列
    arithmetic_seqs = generator.generate_arithmetic()
    geometric_seqs = generator.generate_geometric()
    multiplier_seqs = generator.generate_multiplier()
    
    # 处理每种序列
    for name, seqs in [
        ('arithmetic', arithmetic_seqs),
        ('geometric', geometric_seqs),
        ('multiplier', multiplier_seqs)
    ]:
        # 划分训练测试集
        train_seqs, test_seqs = generator.train_test_split(seqs)
        
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
        generator.save_data(data_dict, f'{name}_data.npz')
        
    print("数据生成完成！")

if __name__ == "__main__":
    main() 