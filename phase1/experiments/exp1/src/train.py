"""
实验一的训练脚本 - 序列预测任务
"""

import torch
# from pathlib import Path
import numpy as np
from typing import Dict, Tuple
import sys
from tqdm import tqdm
import json
from pathlib import Path

# 添加根目录到Python路径
# root_dir = Path(__file__).parent.parent.parent
root_path = "/Users/libin/SAGE/src/sage/models/adaptive_network"
sys.path.insert(0, str(root_path))
from experiments.exp1.src.network import SequencePredictionNetwork
from experiments.exp1.src.utils import load_data, calculate_metrics, save_training_log



# from experiments.exp1.src.network import SequencePredictionNetwork
# from experiments.exp1.src.utils import load_data, calculate_metrics, save_training_log

class ExperimentResults:
    def __init__(self, model_name="adaptive_network"):
        self.results = {}
        self.model_name = model_name
    
    def add_result(self, pattern_type, epoch, train_loss, test_loss):
        if pattern_type not in self.results:
            self.results[pattern_type] = {
                'train_losses': [],
                'test_losses': []
            }
        self.results[pattern_type]['train_losses'].append(train_loss)
        self.results[pattern_type]['test_losses'].append(test_loss)
    
    def get_summary_dict(self):
        summary = {}
        for pattern_type, metrics in self.results.items():
            train_losses = metrics['train_losses']
            test_losses = metrics['test_losses']
            
            summary[pattern_type] = {
                'train_loss': {
                    'initial': float(train_losses[0]),
                    'final': float(train_losses[-1]),
                    'best': float(min(train_losses)),
                    'mean': float(np.mean(train_losses))
                },
                'test_loss': {
                    'initial': float(test_losses[0]),
                    'final': float(test_losses[-1]),
                    'best': float(min(test_losses)),
                    'mean': float(np.mean(test_losses))
                },
                'history': {
                    'train_losses': [float(x) for x in train_losses],
                    'test_losses': [float(x) for x in test_losses]
                }
            }
        return summary
    
    def save_results(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整结果
        results_file = save_dir / f"{self.model_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.get_summary_dict(), f, indent=2)
        
        # 打印结果到控制台
        self.print_summary()
    
    def print_summary(self):
        print("\n=== Experiment Results Summary ===")
        for pattern_type, metrics in self.results.items():
            train_losses = metrics['train_losses']
            test_losses = metrics['test_losses']
            
            print(f"\n{pattern_type.upper()} Pattern:")
            print(f"Training Loss:")
            print(f"  Initial: {train_losses[0]:.4f}")
            print(f"  Final:   {train_losses[-1]:.4f}")
            print(f"  Best:    {min(train_losses):.4f}")
            print(f"  Mean:    {np.mean(train_losses):.4f}")
            print(f"Test Loss:")
            print(f"  Initial: {test_losses[0]:.4f}")
            print(f"  Final:   {test_losses[-1]:.4f}")
            print(f"  Best:    {min(test_losses):.4f}")
            print(f"  Mean:    {np.mean(test_losses):.4f}")

def train_epoch(
    model: SequencePredictionNetwork,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    pattern_type: str
) -> Dict:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    metrics = []
    
    for i in tqdm(range(len(X_train)), desc="训练中", leave=False):
        optimizer.zero_grad()
        
        sequence = X_train[i]
        target = y_train[i].view(-1)
        
        try:
            prediction, stats = model.predict(sequence)
            prediction = prediction.view(-1)
            
            loss = torch.nn.functional.mse_loss(prediction, target)
            
            loss.backward()
            optimizer.step()
            
            performance = 1.0 - loss.item()
            model.adapt_to_pattern(performance, pattern_type)
            
            total_loss += loss.item()
            metrics.append({**stats, 'loss': loss.item()})
            
        except Exception as e:
            print(f"训练样本 {i} 出错:")
            print(f"输入形状: {sequence.shape}")
            print(f"目标形状: {target.shape}")
            print(f"预测形状: {prediction.shape}")
            raise e
    
    return {'avg_loss': total_loss / len(X_train), 'metrics': metrics}

def evaluate(
    model: SequencePredictionNetwork,
    X_test: torch.Tensor,
    y_test: torch.Tensor
) -> Dict:
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            # 获取当前序列和目标
            sequence = X_test[i]
            target = y_test[i].view(-1)
            
            # 预测
            prediction, _ = model.predict(sequence)
            prediction = prediction.view(-1)
            
            # 计算损失
            loss = torch.nn.functional.mse_loss(prediction, target)
            total_loss += loss.item()
            predictions.append(prediction.cpu().numpy())  # 转换为numpy数组
    
    return {
        'avg_loss': total_loss / len(X_test),
        'predictions': predictions  # 现在是numpy数组的列表
    }

def main():
    # 加载数据
    data = load_data()
    results = ExperimentResults(model_name="adaptive_network")
    
    # 创建模型
    model = SequencePredictionNetwork(
        input_size=data['input_window'],
        hidden_size=64
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # 训练循环
    for pattern_type in ['arithmetic', 'geometric', 'multiplier']:
        print(f"\n开始训练 {pattern_type} 模式...")
        
        X_train = data[f'{pattern_type}_X_train']
        y_train = data[f'{pattern_type}_y_train']
        X_test = data[f'{pattern_type}_X_test']
        y_test = data[f'{pattern_type}_y_test']
        
        for epoch in tqdm(range(10), desc=f"训练轮次"):
            train_stats = train_epoch(model, X_train, y_train, optimizer, pattern_type)
            eval_stats = evaluate(model, X_test, y_test)
            
            results.add_result(pattern_type, epoch, train_stats['avg_loss'], eval_stats['avg_loss'])
            save_training_log(train_stats, eval_stats, pattern_type, epoch)
    
    # 保存实验结果
    results_dir = Path(root_path) / "experiments/exp1/results"
    results.save_results(results_dir)

if __name__ == "__main__":
    main() 